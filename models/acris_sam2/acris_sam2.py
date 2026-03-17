import os
from typing import Any, Dict, List, Tuple, Union
import random
import math

import py3_wget
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence # 用于填充多属性文本序列


import models.CLIP.clip.clip as clip

from models.sam2.modeling.sam2_utils import preprocess
from models.sam2.modeling.sam2_base import SAM2Base 
from models.acris_sam2.model_utils import BackboneOutput, DecoderOutput
from util.path_utils import SAM2_PATHS_CONFIG, SAM2_WEIGHTS_URL
from util.promptable_utils import rescale_prompt
from models.acris_sam2.adapter import Adapter

# ==========================================
#  辅助模块 (Position Encoding & Transformer)
# ==========================================

class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos: Union[Tensor, None]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, query_pos: Union[Tensor, None] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos: Union[Tensor, None]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory_key, memory_value, pos: Union[Tensor, None] = None, query_pos: Union[Tensor, None] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory_key, pos),
                                   value=memory_value)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

class VLPAlignedDecoder(nn.Module):
    def __init__(self, hidden_dim=256, num_queries=50, nheads=8, dec_layers=1):
        super().__init__()
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        
        self.num_layers = dec_layers
        self.num_queries = num_queries
        self.supp_q_feat = nn.Embedding(num_queries, hidden_dim)

        self.transformer_cross_attention_layers_0 = nn.ModuleList() 
        self.transformer_self_attention_layers_0 = nn.ModuleList()  
        self.transformer_cross_attention_layers = nn.ModuleList()   
        self.transformer_self_attention_layers = nn.ModuleList()    

        for _ in range(self.num_layers):
            self.transformer_cross_attention_layers_0.append(CrossAttentionLayer(hidden_dim, nheads))
            self.transformer_self_attention_layers_0.append(SelfAttentionLayer(hidden_dim, nheads))
            self.transformer_cross_attention_layers.append(CrossAttentionLayer(hidden_dim, nheads))
            self.transformer_self_attention_layers.append(SelfAttentionLayer(hidden_dim, nheads))

    def forward(self, query_feat, supp_feat):
        bs, C, H, W = query_feat.shape
        pos_x = self.pe_layer(query_feat, None).flatten(2).permute(2, 0, 1) 
        src_x = query_feat.flatten(2).permute(2, 0, 1)                      
        pos_x_s = self.pe_layer(supp_feat, None).flatten(2).permute(2, 0, 1)
        src_x_s = supp_feat.flatten(2).permute(2, 0, 1)                     
        
        output = self.supp_q_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        for i in range(self.num_layers):
            output = self.transformer_cross_attention_layers_0[i](output, src_x_s, src_x_s, pos=pos_x_s)
            output = self.transformer_self_attention_layers_0[i](output)
            output = self.transformer_cross_attention_layers[i](output, src_x, src_x, pos=pos_x)
            output = self.transformer_self_attention_layers[i](output)
            
        return output.transpose(0, 1)

class TextGuidedChannelAttention(nn.Module):
    def __init__(self, in_channels, text_dim=512, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(text_dim, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, text_vec):
        b, c, _, _ = x.size()
        weight = self.fc(text_vec).view(b, c, 1, 1)
        return x * weight

# ==========================================
#  SANSA 主模型
# ==========================================
class SANSA(nn.Module):
    def __init__(self, sam: SAM2Base, device: torch.device, clip_model: nn.Module = None):
        super().__init__()
        self.sam = sam
        self.device = device
        self.clip_model = clip_model 
        
        if self.clip_model is not None:
            self._init_vlp_layers()
            # VLP Decoder
            self.vlp_decoder = VLPAlignedDecoder(hidden_dim=256, num_queries=50, dec_layers=1)

    def _init_vlp_layers(self):
        self.vlp_fea_dim = 512 
        self.vlp_hidden_dim = 256
        
        self.downsample_query = nn.Sequential(
            nn.Conv2d(self.vlp_fea_dim, self.vlp_hidden_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )

        feat_multi = 3 
        input_dim = self.vlp_hidden_dim * feat_multi + 2 

        # 跨模态记忆投射 MLP (现用于投射属性序列)
        self.text_to_sam_mem = nn.Sequential(
            nn.Linear(self.vlp_fea_dim, self.vlp_hidden_dim),
            nn.LayerNorm(self.vlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.vlp_hidden_dim, self.sam.hidden_dim)
        )

        # 空间感知交叉注意力层：用于视觉特征动态查询文本序列属性
        self.text_vision_cross_attn = CrossAttentionLayer(d_model=self.sam.hidden_dim, nhead=8)

        # 简单的 1x1 卷积融合
        self.merge_1 = nn.Sequential(
            nn.Conv2d(input_dim, self.vlp_hidden_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )
        
        self.text_gate = TextGuidedChannelAttention(in_channels=self.vlp_hidden_dim, text_dim=512)

        self.aux_head = nn.Sequential(
            nn.Conv2d(self.vlp_hidden_dim, self.vlp_hidden_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, self.vlp_hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5), 
            nn.ConvTranspose2d(self.vlp_hidden_dim // 2, self.vlp_hidden_dim // 4, kernel_size=2, stride=2),
            nn.GroupNorm(1, self.vlp_hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.vlp_hidden_dim // 4, self.vlp_hidden_dim // 8, kernel_size=2, stride=2),
            nn.GroupNorm(1, self.vlp_hidden_dim // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.vlp_hidden_dim // 8, 1, kernel_size=1)
        )
        torch.nn.init.normal_(self.aux_head[-1].weight, std=0.01)
        torch.nn.init.constant_(self.aux_head[-1].bias, -2.19)

    def forward(self, samples: torch.Tensor, prompt_dict: List[Dict[str, Any]], text_prompts = None, clip_inputs=None, clip_masks=None) -> Dict[str, Any]:
        B, T, C, H, W = samples.shape
        outputs = {"masks": []}
        n_shots = prompt_dict['shots']

        # [1] N-shot VLP 特征提取与提示生成
        sparse_embeddings = None
        vlp_dense = None
        text_vec = None
        text_seq = None 
        
        if clip_inputs is not None and self.clip_model is not None and clip_masks is not None and n_shots > 0:
            support_imgs = clip_inputs[:, :n_shots] # [B, N, 3, 224, 224]
            query_img = clip_inputs[:, -1]          # [B, 3, 224, 224]
            support_masks_in = clip_masks 

            vlp_res = self.get_vlp_features(query_img, support_imgs, support_masks_in, text_prompts)
            supp_feats_1 = vlp_res["supp_feats_1"]
            
            text_vec = vlp_res["text_vec"]
            text_seq = vlp_res["text_seq"] # [B, L, 512] 

            batch_prompts = []
            aux_masks = [] 

            for i in range(n_shots):
                supp_feat_i = supp_feats_1[:, i, :, :, :] 
                proto_i = vlp_res["protos"][:, i, :, :, :]           
                pseudo_mask_i = vlp_res["pseudo_masks"][:, i, :, :, :] 
                
                H_feat, W_feat = vlp_res["query_feat_down"].shape[-2:]
                supp_feat_bin_i = proto_i.repeat(1, 1, H_feat, W_feat)
                
                merged_feat = self.merge_1(torch.cat([
                    vlp_res["query_feat_down"],
                    supp_feat_bin_i,
                    vlp_res["text_feat_down"],
                    pseudo_mask_i * 10,  
                    vlp_res["query_attn_map"] * 10
                ], dim=1))
                
                merged_feat = self.text_gate(merged_feat, text_vec)
                aux_mask_i = self.aux_head(merged_feat) 
                aux_masks.append(aux_mask_i)

                prompts_i = self.vlp_decoder(merged_feat, supp_feat_i)
                batch_prompts.append(prompts_i)
            
            sparse_embeddings = torch.cat(batch_prompts, dim=1) # [B, N*50, 256]

            avg_aux_mask = torch.stack(aux_masks, dim=1).mean(dim=1) 
            outputs["aux_pseudo_mask"] = avg_aux_mask
            vlp_dense = F.interpolate(avg_aux_mask, size=(256, 256), mode='bilinear', align_corners=False)

        # [2] SAM2 流程
        samples_bt, B, T, orig_size = self._preprocess_visual_features(samples, self.sam.image_size)
        
        vis_base = self.sam.image_encoder.trunk(samples_bt)
        feats, pos = self.sam.image_encoder.neck(vis_base)
        
        feats_input, pos_input = feats[:-1], pos[:-1]
        feats_input[0] = self.sam.sam_mask_decoder.conv_s0(feats_input[0])
        feats_input[1] = self.sam.sam_mask_decoder.conv_s1(feats_input[1])
        
        bb = {
            "vision_features": feats_input[-1], 
            "vision_pos_enc": pos_input,
            "backbone_fpn": feats_input
        }
        vision_feats, vision_pos, sizes = self.sam._prepare_backbone_features(bb)
        
        self.memory_bank = {}
        batch_pred_masks = []
        
        batched_vision_feats = [f.view(f.shape[0], B, T, f.shape[2]) for f in vision_feats]
        batched_vision_pos = [p.view(p.shape[0], B, T, p.shape[2]) for p in vision_pos]

        # ---- [SAM Memory 条件] 将 CLIP 文本序列映射 ----
        if text_seq is not None:
            text_mem_cond = self.text_to_sam_mem(text_seq).permute(1, 0, 2)
        else:
            text_mem_cond = None

        for idx in range(T):
            current_vision_feats = [f[:, :, idx, :] for f in batched_vision_feats]
            current_vision_pos = [p[:, :, idx, :] for p in batched_vision_pos]
            
            # --- [文本语义注入] ---
            if text_mem_cond is not None:
                vis_feat = current_vision_feats[-1] 
                vis_pos_embed = current_vision_pos[-1] 
                
                vis_refined = self.text_vision_cross_attn(
                    tgt=vis_feat, 
                    memory_key=text_mem_cond, 
                    memory_value=text_mem_cond, 
                    pos=None,            
                    query_pos=vis_pos_embed  
                )
                
                current_vision_feats[-1] = vis_refined

            high_res_features = [
                x.permute(1, 2, 0).view(B, x.size(2), *s) 
                for x, s in zip(current_vision_feats[:-1], sizes[:-1])
            ]
            
            if idx < n_shots:
                frame_prompts = []
                for b in range(B):
                    prompt_type = prompt_dict[b][idx]['prompt_type']
                    fp = rescale_prompt(prompt_dict[b][idx]['prompt'], prompt_type, orig_size[b], self.sam.image_size)
                    frame_prompts.append(fp)
                
                if prompt_type == 'mask':
                    batched_mask_prompt = torch.cat(frame_prompts, dim=0)
                    
                    out_scale, out_bias = 20.0, -10.0
                    mask_inputs_float = batched_mask_prompt.float()
                    high_res_masks = mask_inputs_float * out_scale + out_bias
                    low_res_masks = F.interpolate(
                        high_res_masks,
                        size=(high_res_masks.size(-2) // 4, high_res_masks.size(-1) // 4),
                        align_corners=False, mode="bilinear", antialias=True,
                    )
                    
                    if not self.sam.use_obj_ptrs_in_encoder:
                        obj_ptr = torch.zeros(B, self.sam.hidden_dim, device=self.device)
                    else:
                        fake_out = self.sam._forward_sam_heads(
                            backbone_features=current_vision_feats[-1].permute(1, 2, 0).view(B, self.sam.hidden_dim, *sizes[-1]),
                            mask_inputs=self.sam.mask_downsample(mask_inputs_float),
                            high_res_features=high_res_features,
                        )
                        obj_ptr = fake_out.obj_ptr
                    
                    is_obj_appearing = torch.any(batched_mask_prompt.flatten(1).float() > 0.0, dim=1)[..., None]
                    if self.sam.pred_obj_scores:
                        obj_ptr = obj_ptr * is_obj_appearing.float() if self.sam.fixed_no_obj_ptr else \
                                  obj_ptr + (1 - is_obj_appearing.float()) * self.sam.no_obj_ptr

                    decoder_out = DecoderOutput(
                        low_res_masks=low_res_masks,
                        high_res_masks=F.interpolate(low_res_masks, size=(self.sam.image_size, self.sam.image_size), mode="bilinear", align_corners=False),
                        obj_ptr=obj_ptr, 
                        pix_feat_with_mem=current_vision_feats[-1]
                    )
                else:
                    pass 
            else:
                vlp_sparse = sparse_embeddings 
                vlp_dense_b = vlp_dense        
                
                pix_feat_with_mem = self.sam._prepare_memory_conditioned_features(
                    frame_idx=idx,
                    current_vision_feats=current_vision_feats[-1:],
                    current_vision_pos_embeds=current_vision_pos[-1:],
                    feat_sizes=sizes[-1:],
                    num_frames=idx+1,
                    memory_bank=self.memory_bank
                )
                
                decoder_out = self.sam._forward_sam_heads(
                    backbone_features=pix_feat_with_mem, 
                    high_res_features=high_res_features, 
                    multimask_output=True if idx > 0 else False,
                    text_inputs=vlp_sparse,
                    mask_inputs=vlp_dense_b
                )

            mem_feats, mem_pos = self.sam._encode_new_memory(
                current_vision_feats=current_vision_feats, 
                feat_sizes=sizes, 
                pred_masks_high_res=decoder_out.high_res_masks, 
                is_mask_from_pts=False
            )
            self.memory_bank[idx] = {
                "maskmem_features": mem_feats, 
                "maskmem_pos_enc": mem_pos, 
                "pred_masks": decoder_out.low_res_masks, 
                "obj_ptr": decoder_out.obj_ptr
            }
            
            batch_pred_masks.append(decoder_out.high_res_masks)

        masks = torch.stack(batch_pred_masks, dim=1)
        masks = masks.view(B * T, 1, masks.size(3), masks.size(4)).squeeze(1) 
        masks = F.interpolate(masks[None], size=orig_size[0], mode='bilinear', align_corners=False)[0]
        
        return {"pred_masks": masks, "aux_pseudo_mask": outputs.get("aux_pseudo_mask", None)}

    def get_vlp_features(self, query_img, support_imgs, support_masks, text_prompts):
        B, N, _, H, W = support_imgs.shape
        with torch.no_grad():
            query_feat = self.clip_model.encode_image(query_img).float()
            H_feat, W_feat = query_feat.shape[-2:]

            supp_imgs_flat = support_imgs.reshape(B*N, 3, H, W)
            supp_feats_flat = self.clip_model.encode_image(supp_imgs_flat).float()

            # ---- [1. 多文本属性主次加权融合] ----
            batch_text_features = []
            batch_global_vecs = []
            
            if text_prompts is not None:
                for p in text_prompts:
                    if isinstance(p, (list, tuple)):
                        attrs = list(p)
                    elif isinstance(p, str):
                        attrs = [a.strip() for a in p.split(',')]
                    else:
                        attrs = [str(p)]

                    tokens = clip.tokenize(attrs).to(self.device)
                    attr_feats = self.clip_model.encode_text(tokens).float() 
                    attr_feats = attr_feats / attr_feats.norm(dim=-1, keepdim=True)
                    
                    cls_feat = attr_feats[0:1] # 类别为主 [1, 512]
                    if len(attr_feats) > 1:
                        aux_feat = attr_feats[1:].mean(dim=0, keepdim=True) # 属性为辅
                        alpha = 0.75 # 类别占主导权重
                        global_vec = alpha * cls_feat + (1 - alpha) * aux_feat
                    else:
                        global_vec = cls_feat
                        
                    global_vec = global_vec / global_vec.norm(dim=-1, keepdim=True)
                    
                    batch_text_features.append(attr_feats)
                    batch_global_vecs.append(global_vec.squeeze(0))
            else:
                for _ in range(B):
                    empty_feat = torch.zeros(1, 512, device=self.device)
                    batch_text_features.append(empty_feat)
                    batch_global_vecs.append(empty_feat.squeeze(0))

            text_feature = torch.stack(batch_global_vecs, dim=0) # [B, 512]
            text_seq = pad_sequence(batch_text_features, batch_first=True)

            # 严格对齐 VLP: 提取前先归一化
            text_feature_norm = (text_feature / text_feature.norm(dim=-1, keepdim=True)).float()

            # ---- [2. 完全严格对齐 VLP: Attn Map 的原生 API 调用] ----
            # Query Attn
            B_q, C_q, H_q, W_q = query_feat.shape
            query_features = query_feat.reshape(B_q, C_q, -1).permute(0, 2, 1) @ text_feature_norm.t()
            query_similarity_map = clip.get_similarity_map(query_features, (H_q, W_q))
            
            query_attn_map = []
            for i in range(B_q):
                query_attn_map.append(query_similarity_map[i, :, :, i])
            query_attn_map = torch.stack(query_attn_map).unsqueeze(1)

            # Support Attn (使用 repeat_interleave 对齐 N-shot 原有逻辑)
            text_feature_supp_norm = text_feature_norm.repeat_interleave(N, dim=0)
            supp_features = supp_feats_flat.reshape(B*N, C_q, -1).permute(0, 2, 1) @ text_feature_supp_norm.t()
            supp_similarity_map = clip.get_similarity_map(supp_features, (H_q, W_q))
            
            supp_attn_map = []
            for i in range(B * N):
                supp_attn_map.append(supp_similarity_map[i, :, :, i])
            supp_attn_map_flat = torch.stack(supp_attn_map).unsqueeze(1)

            # ---- [3. 完全严格对齐 VLP: 诡异的 3x Mask 插值法] ----
            supp_masks_flat = support_masks.reshape(B*N, -1, support_masks.shape[-1])
            resize_size = H_feat * 3
            # VLP的第一步：先 Nearest 插值到特征尺寸的 3 倍
            support_mask_3x = F.interpolate(supp_masks_flat.unsqueeze(1).float(), size=(resize_size, resize_size), mode='nearest')
            
            # VLP第二步：计算 Pseudo Mask
            query_feat_expanded = query_feat.unsqueeze(1).expand(-1, N, -1, -1, -1).reshape(B*N, self.vlp_fea_dim, H_feat, W_feat)
            pseudo_masks_flat = self.get_pseudo_mask_aligned(
                tmp_supp_feat=supp_feats_flat, 
                query_feat=query_feat_expanded, 
                mask_3x=support_mask_3x # 传入 3x mask
            )

        # 降维
        query_feat_down = self.downsample_query(query_feat) 
        supp_feats_down_flat = self.downsample_query(supp_feats_flat) 

        text_feat_down = self.downsample_query(text_feature.unsqueeze(-1).unsqueeze(-1))
        text_feat_down = text_feat_down.repeat(1, 1, H_feat, W_feat) 
        
        text_feature_expanded = text_feature.unsqueeze(1).expand(-1, N, -1).reshape(B*N, -1)
        text_feat_supp_down = self.downsample_query(text_feature_expanded.unsqueeze(-1).unsqueeze(-1))
        text_feat_supp_down = text_feat_supp_down.repeat(1, 1, H_feat, W_feat) 

        # ---- [4. 完全严格对齐 VLP: Protos 获取] ----
        with torch.no_grad():
            # VLP第三步：计算 Protos 时，把 3x mask 用 bilinear 缩小，且默认 align_corners=False
            support_mask_for_protos = F.interpolate(support_mask_3x, size=(H_feat, W_feat), mode='bilinear')
            protos_flat = self.mask_feature(supp_feats_down_flat, support_mask_for_protos) 
            
        supp_proto_self_flat = protos_flat.repeat(1, 1, H_feat, W_feat)
        
        # 特征融合
        supp_feats_1_flat = self.merge_1(torch.cat([
            supp_feats_down_flat,
            supp_proto_self_flat, 
            text_feat_supp_down,
            support_mask_for_protos * 10, # 注意：严格对齐要求这里使用 bilinear 降维后的软 mask
            supp_attn_map_flat * 10
        ], dim=1))
        
        supp_feats_1 = supp_feats_1_flat.view(B, N, -1, H_feat, W_feat)

        return {
            "supp_feats_1": supp_feats_1,          
            "query_feat_down": query_feat_down,    
            "text_feat_down": text_feat_down,  
            "text_vec": text_feature,               
            "text_seq": text_seq,       
            "query_attn_map": query_attn_map,
            "supp_attn_map": supp_attn_map_flat,
            "protos": protos_flat.view(B, N, -1, 1, 1),            
            "pseudo_masks": pseudo_masks_flat.view(B, N, 1, H_feat, W_feat) 
        }

    # 严格对齐的高维像素级伪掩码获取
    def get_pseudo_mask_aligned(self, tmp_supp_feat, query_feat, mask_3x):
        bsize, ch_sz, sp_sz, _ = query_feat.size()
        
        # VLP 内部强制使用 bilinear 且 align_corners=True 处理 3x mask
        tmp_mask = F.interpolate(mask_3x, size=(sp_sz, sp_sz), mode='bilinear', align_corners=True)
        tmp_supp_feat_masked = tmp_supp_feat * tmp_mask

        tmp_query = query_feat.reshape(bsize, ch_sz, -1) 
        tmp_query_norm = torch.norm(tmp_query, 2, 1, True) 

        tmp_supp = tmp_supp_feat_masked.reshape(bsize, ch_sz, -1).permute(0, 2, 1) 
        tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True) 

        cosine_eps = 1e-7
        similarity = torch.bmm(tmp_supp, tmp_query) / (torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)   
        
        # .max(1)[0] 严格复刻了像素级匹配：寻找 Query 每个像素在 Support 里的最高响应值
        similarity = similarity.max(1)[0].reshape(bsize, sp_sz * sp_sz)
        corr_query = similarity.reshape(bsize, 1, sp_sz, sp_sz)
        
        return corr_query

    def mask_feature(self, features, support_mask):
        mask = support_mask
        supp_feat = features * mask
        feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
        area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
        supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
        return supp_feat

    def _preprocess_visual_features(self, samples, image_size):
        B, T, C, H, W = samples.shape
        samples_view = samples.view(B * T, C, H, W)
        orig_size = [tuple(x.shape[-2:]) for x in samples_view]
        samples_pre = torch.stack([preprocess(x, image_size) for x in samples_view], dim=0)
        return samples_pre, B, T, orig_size

def build_acris_sam2(
    sam2_version: str = 'large', 
    adaptformer_stages: List[int] = [2, 3], 
    channel_factor: float = 0.3, 
    device: str = 'cuda',
    clip_version: str = "CS-ViT-B/16"
):
    assert sam2_version in SAM2_PATHS_CONFIG.keys()
    sam2_weights, sam2_config = SAM2_PATHS_CONFIG[sam2_version]
    if not os.path.isfile(sam2_weights):
        py3_wget.download_file(SAM2_WEIGHTS_URL[sam2_version], sam2_weights)

    print(f"Loading CLIP {clip_version}...")
    clip_model, _ = clip.load(clip_version, device=device)
    dummy_img = torch.zeros(1, 3, 224, 224, device=device)
    _ = clip_model.encode_image(dummy_img)
    for p in clip_model.parameters():
        p.requires_grad = False 

    with initialize(version_base=None, config_path=".", job_name="test_app"):
        cfg = compose(config_name=sam2_config, overrides=[
            f"++model.image_encoder.trunk.adaptformer_stages={adaptformer_stages}", 
            f"++model.image_encoder.trunk.adapt_dim={channel_factor}",
        ])
        OmegaConf.resolve(cfg)
        cfg.model.pred_obj_scores = False
        cfg.model.pred_obj_scores_mlp = False
        cfg.model.fixed_no_obj_ptr = False
        sam = instantiate(cfg.model, _recursive_=True)

    state_dict = torch.load(sam2_weights, map_location="cpu", weights_only=False)["model"]
    sam.load_state_dict(state_dict, strict=False)
    model = SANSA(sam=sam, device=torch.device(device), clip_model=clip_model)

    for name, p in model.named_parameters():
        # [修改] 已移除 in_context_fusion
        if any(k in name for k in ["adapter", "mask_decoder", "downsample_query", "merge_1", "text_gate", "vlp_decoder", "aux_head", "text_to_sam_mem", "text_vision_cross_attn"]):
            p.requires_grad = True
        else:
            p.requires_grad = False

    return model