import os
import random
import numpy as np
import PIL.Image as Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from models.CLIP import clip

# PASCAL VOC 的 20 个基础类别映射 (索引 0 是 background, 255 是 ignore)
PASCAL_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pot plant', 'sheep', 'sofa', 'train', 'television'
]

# 参考 COCO_ATTRIBUTES，为 PASCAL 的 20 个类别定义 5 个核心属性
PASCAL_ATTRIBUTES = {
    "aeroplane": ["wings", "a fuselage", "a tail fin", "jet engines", "landing gear"],
    "bicycle": ["two wheels", "handlebars", "pedals", "a metal frame", "a saddle seat"],
    "bird": ["feathers", "a beak", "wings", "two legs", "lightweight skeleton"],
    "boat": ["a hull shape", "a deck area", "floatation capability", "often a mast", "a rudder"],
    "bottle": ["a narrow neck", "a wide body", "a cap or lid", "transparent or colored material", "liquid containment function"],
    "bus": ["a large rectangular body", "multiple windows", "wheels", "doors for passengers", "seats inside"],
    "car": ["four wheels", "a metal body", "a windshield", "headlights", "doors"],
    "cat": ["a furry body", "pointy ears", "whiskers", "a tail", "retractable claws"],
    "chair": ["a seat surface", "a backrest", "four support legs", "indoor furniture design", "an ergonomic shape"],
    "cow": ["a large body", "four legs", "hooves", "an udder", "spots or solid coloring"],
    "diningtable": ["a flat surface", "support legs", "a seating capacity", "wood or metal material", "a dining function"],
    "dog": ["a furry body", "floppy or pointy ears", "a tail", "a snout", "four legs"],
    "horse": ["a large muscular body", "a mane", "a tail", "four long legs", "hooves"],
    "motorbike": ["two wheels", "handlebars", "an engine", "a metal frame", "a rider seat"],
    "person": ["a human face", "two arms", "two legs", "a head with hair", "clothing"],
    "potted plant": ["a living plant", "a decorative pot", "soil medium", "foliage coverage", "indoor decoration function"],
    "sheep": ["a woolly coat", "four legs", "hooves", "curled horns", "a docile expression"],
    "sofa": ["a long seating area", "a backrest", "armrests", "soft cushions", "upholstered surface material"],
    "train": ["multiple connected cars", "a metal structure", "wheels on tracks", "a locomotive engine", "windows along sides"],
    "television": ["a flat screen", "an electronic display", "audio speakers", "a remote control interface", "entertainment functionality"]
}

class DatasetPASCAL(Dataset):
    def __init__(self, datapath, fold, transform, clip_transform, split, shot):
        self.split = split
        self.benchmark = 'pascal_voc'
        self.shot = shot
        self.fold = fold
        self.transform = transform
        self.clip_transform = clip_transform
        self.datapath = datapath
        
        # PASCAL VOC 包含 20 个前景类（ID 1~20），背景 ID 为 0
        self.nclass = 21
        
        self.base_path = os.path.join(datapath, 'VOCdevkit2012/VOC2012')
        
        # 1. 预解析 data_list
        list_file = os.path.join(self.base_path, f'fss_list/{split}/data_list_{fold}.txt')
        with open(list_file, 'r') as f:
            f_str = f.readlines()
        self.data_list = []
        for line in f_str:
            if line.strip():
                img, mask = line.strip().split(' ')
                self.data_list.append((img, mask))
            
        # 2. 读取 sub_class_file_list
        sub_class_file = os.path.join(self.base_path, f'fss_list/{split}/sub_class_file_list_{fold}.txt')
        with open(sub_class_file, 'r') as f:
            f_str = f.read()
        self.sub_class_file_list = eval(f_str)
        
        # 获取当前 fold 包含的所有新类/基础类 ID
        self.class_ids = list(self.sub_class_file_list.keys())

    def _clean_path(self, path_str):
        if path_str.startswith('../data/'):
            path_str = path_str[8:]
        return os.path.join(self.datapath, path_str)

    def read_mask(self, mask_path):
        """ 直接读取为 Tensor，对齐 COCO 加速逻辑 """
        return torch.tensor(np.array(Image.open(mask_path)))

    def __len__(self):
        return len(self.data_list)
        # return len(self.data_list)

    def __getitem__(self, idx):
        # =======================================================
        # 1. 加载 Query 图像与标签
        # =======================================================
        image_path, label_path = self.data_list[idx]
        query_img_path = self._clean_path(image_path)
        query_mask_path = self._clean_path(label_path)
        
        query_img_pil = Image.open(query_img_path).convert('RGB')
        query_mask_raw = self.read_mask(query_mask_path)
        
        # =======================================================
        # 2. 类别提取与强制过滤
        # =======================================================
        label_class = torch.unique(query_mask_raw).tolist()
        if 0 in label_class: label_class.remove(0)
        if 255 in label_class: label_class.remove(255)
        
        new_label_class = [c for c in label_class if c in self.class_ids]
        label_class = new_label_class
        
        assert len(label_class) > 0, f"Query image {query_img_path} does not contain valid novel/base classes for this fold!"
        
        # 随机选取一个作为 Target Class
        class_chosen = label_class[random.randint(1, len(label_class)) - 1]
        class_name = PASCAL_CLASSES[class_chosen]
        
        # 对齐 COCO: 向量化二值化与过滤 query_mask
        base_query_mask = query_mask_raw.clone()
        query_mask = (query_mask_raw == class_chosen).float()
        
        valid_classes = torch.tensor(self.class_ids, device=base_query_mask.device)
        mask_to_keep = torch.isin(base_query_mask, valid_classes)
        base_query_mask[~mask_to_keep] = 0

        # =======================================================
        # 3. 严格防重的 Support 采样 
        # =======================================================
        file_class_chosen = self.sub_class_file_list[class_chosen]
        num_file = len(file_class_chosen)
        
        support_image_path_list = []
        support_label_path_list = []
        support_idx_list = []
        
        for k in range(self.shot):
            support_idx = random.randint(1, num_file) - 1
            support_image_path = image_path
            support_label_path = label_path
            
            if num_file > self.shot:
                while ((support_image_path == image_path and support_label_path == label_path) or support_idx in support_idx_list):
                    support_idx = random.randint(1, num_file) - 1
                    support_image_path, support_label_path = file_class_chosen[support_idx]
            else:
                support_image_path, support_label_path = file_class_chosen[support_idx]
                
            support_idx_list.append(support_idx)
            support_image_path_list.append(support_image_path)
            support_label_path_list.append(support_label_path)

        # 真正读取被选中的 Support 图片 (加速处理)
        support_imgs_pil = []
        support_masks_tensors = []
        base_support_masks_tensors = []
        support_names = []
        
        for i in range(self.shot):
            sup_img_p = self._clean_path(support_image_path_list[i])
            sup_mask_p = self._clean_path(support_label_path_list[i])
            
            support_imgs_pil.append(Image.open(sup_img_p).convert('RGB'))
            support_names.append(sup_img_p)
            
            mask_raw = self.read_mask(sup_mask_p)
            base_mask = mask_raw.clone()
            
            # 对齐 COCO: 向量化二值化与过滤 support_mask
            mask = (mask_raw == class_chosen).float()
            
            valid_support_mask = torch.isin(base_mask, valid_classes)
            base_mask[~valid_support_mask] = 0
            
            support_masks_tensors.append(mask)
            base_support_masks_tensors.append(base_mask)

        # =======================================================
        # 4. SAM2 与 CLIP 预处理 (对齐 COCO 插值逻辑)
        # =======================================================
        # --- SAM2 处理 ---
        sam_query_img = self.transform(query_img_pil)
        sam_query_mask = F.interpolate(query_mask.unsqueeze(0).unsqueeze(0).float(), sam_query_img.size()[-2:], mode='nearest').squeeze()
        sam_base_query = F.interpolate(base_query_mask.unsqueeze(0).unsqueeze(0).float(), sam_query_img.size()[-2:], mode='nearest').squeeze()
        
        sam_support_imgs = torch.stack([self.transform(img) for img in support_imgs_pil])
        sam_support_masks = torch.stack([
            F.interpolate(m.unsqueeze(0).unsqueeze(0).float(), sam_support_imgs.size()[-2:], mode='nearest').squeeze() 
            for m in support_masks_tensors
        ])
        sam_base_supports = torch.stack([
            F.interpolate(m.unsqueeze(0).unsqueeze(0).float(), sam_support_imgs.size()[-2:], mode='nearest').squeeze() 
            for m in base_support_masks_tensors
        ])

        # --- CLIP 处理 ---
        clip_query_img = self.clip_transform(query_img_pil)
        clip_support_imgs = torch.stack([self.clip_transform(img) for img in support_imgs_pil])
        clip_support_masks = torch.stack([
            F.interpolate(m.unsqueeze(0).unsqueeze(0).float(), clip_support_imgs.size()[-2:], mode='nearest').squeeze() 
            for m in support_masks_tensors
        ])

        # =======================================================
        # 5. 构建增强属性提示词与 Batch 输出
        # =======================================================
        text_prompt = f"a photo of a {class_name}"
        class_attributes = PASCAL_ATTRIBUTES.get(class_name, [])
        attribute_prompts = [text_prompt] + [f"the {class_name} has {attr}" for attr in class_attributes]
        attribute_prompts_str = ", ".join(attribute_prompts)

        # 严格对齐 COCO 的输出字典 Keys 格式
        return {
            'query_img': sam_query_img,
            'clip_query_img': clip_query_img,
            'query_mask': sam_query_mask,
            'support_imgs': sam_support_imgs,
            'clip_support_imgs': clip_support_imgs,
            'support_masks': sam_support_masks,
            'clip_support_masks': clip_support_masks,
            'base_masks': [sam_base_supports, sam_base_query],
            'query_name': query_img_path,
            'support_names': support_names,
            'class_id': torch.tensor(class_chosen),
            'class_names': attribute_prompts_str
        }

def build(image_set, args):
    img_size = 518  # 可按需修改为与 COCO 的 640 一致，或改为 args.image_size
    
    # SAM2 Transform
    transform = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # CLIP Transform
    try:
        _, clip_preprocess = clip.load("ViT-B/16", device='cuda', jit=False)
    except Exception as e:
        print(f"[WARNING] CLIP load failed in pascal.py: {e}. Using fallback transform.")
        clip_preprocess = transform
    
    split_name = 'train' if image_set in ['train', 'trn'] else 'val'
    
    dataset = DatasetPASCAL(
        datapath=args.data_root, 
        fold=args.fold, 
        transform=transform,
        clip_transform=clip_preprocess,
        shot=args.shots, 
        split=split_name
    )
    return dataset