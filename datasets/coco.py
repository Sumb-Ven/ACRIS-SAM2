r""" COCO-20i few-shot semantic segmentation dataset with attribute-enhanced prompts """
import os
import pickle
from os.path import join
import numpy as np
import PIL.Image as Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from models.CLIP import clip


COCO_CLASSES = [ 
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", 
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", 
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", 
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", 
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", 
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", 
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", 
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", 
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


COCO_ATTRIBUTES = {
    "person": ["a human face", "two arms", "two legs", "a head with hair", "clothing"],
    "bicycle": ["two wheels", "handlebars", "pedals", "a metal frame", "a saddle seat"],
    "car": ["four wheels", "a metal body", "a windshield", "headlights", "doors"],
    "motorcycle": ["two wheels", "handlebars", "an engine", "a metal frame", "a rider seat"],
    "airplane": ["wings", "a fuselage", "a tail fin", "jet engines", "landing gear"],
    "bus": ["a large rectangular body", "multiple windows", "wheels", "doors for passengers", "seats inside"],
    "train": ["multiple connected cars", "a metal structure", "wheels on tracks", "a locomotive engine", "windows along sides"],
    "truck": ["a large cargo area", "a cab for driver", "wheels", "a metal frame", "headlights"],
    "boat": ["a hull shape", "a deck area", "floatation capability", "often a mast", "a rudder"],
    "traffic light": ["a vertical pole", "three colored lights", "red yellow green signals", "a metal housing", "electrical components"],
    "fire hydrant": ["red or yellow metal", "a vertical cylindrical shape", "water outlet valves", "connection to water mains", "protective caps"],
    "stop sign": ["an octagonal shape", "a red background", "white lettering", "a reflective surface", "a mounting pole"],
    "parking meter": ["a vertical pole", "a digital display", "a coin slot", "a metal housing", "a payment mechanism"],
    "bench": ["a flat seating surface", "support legs", "a backrest", "wood or metal material", "public furniture design"],
    "bird": ["feathers", "a beak", "wings", "two legs", "lightweight skeleton"],
    "cat": ["a furry body", "pointy ears", "whiskers", "a tail", "retractable claws"],
    "dog": ["a furry body", "floppy or pointy ears", "a tail", "a snout", "four legs"],
    "horse": ["a large muscular body", "a mane", "a tail", "four long legs", "hooves"],
    "sheep": ["a woolly coat", "four legs", "hooves", "curled horns", "a docile expression"],
    "cow": ["a large body", "four legs", "hooves", "an udder", "spots or solid coloring"],
    "elephant": ["a large gray body", "a long trunk", "large ears", "tusks", "thick wrinkled skin"],
    "bear": ["a large body", "thick fur", "powerful limbs", "sharp claws", "a prominent snout"],
    "zebra": ["black and white stripes", "a horse-like body", "a mane", "four legs", "hooves"],
    "giraffe": ["a long neck", "a tall stature", "a spotted pattern", "long legs", "small ossicones"],
    "backpack": ["shoulder straps", "a main compartment", "zipper closures", "padding for comfort", "multiple pockets"],
    "umbrella": ["a folding canopy", "a central shaft", "a handle grip", "metal ribs", "waterproof material"],
    "handbag": ["handles or straps", "a closure mechanism", "fabric or leather material", "interior pockets", "a fashionable design"],
    "tie": ["a long narrow fabric strip", "a knot at front", "silk or polyester material", "business attire function", "decorative patterns"],
    "suitcase": ["a rectangular shape", "wheels", "a telescopic handle", "zipper closures", "travel functionality"],
    "frisbee": ["a disc shape", "plastic material", "an aerodynamic design", "a smooth edge", "bright colors"],
    "skis": ["long narrow boards", "metal edges", "bindings for boots", "a curved tip", "a waxable base"],
    "snowboard": ["a wide rectangular board", "bindings for boots", "a curved shape", "flexible material", "a directional design"],
    "sports ball": ["a spherical shape", "bouncy material", "grip texture", "inflatable structure", "colorful patterns"],
    "kite": ["a lightweight frame", "fabric covering", "a stabilizing tail", "string attachment point", "an aerodynamic shape"],
    "baseball bat": ["a long cylindrical shape", "a thick hitting end", "a slender handle", "wood or metal material", "a grip area"],
    "baseball glove": ["leather material", "a padded palm", "finger slots", "webbing between fingers", "a wrist strap"],
    "skateboard": ["a flat deck", "four wheels", "metal trucks", "grip tape surface", "a kicktail"],
    "surfboard": ["a long flat board", "a curved bottom", "fins at back", "a waxed surface", "lightweight construction"],
    "tennis racket": ["an oval head", "string mesh", "a long handle", "a grip tape", "a lightweight frame"],
    "bottle": ["a narrow neck", "a wide body", "a cap or lid", "transparent or colored material", "liquid containment function"],
    "wine glass": ["a long stem", "a wide bowl", "a thin rim", "glass material", "an elegant design"],
    "cup": ["a cylindrical shape", "a handle", "an open top", "ceramic or plastic material", "a drinking function"],
    "fork": ["metal tines", "a long handle", "four prongs", "stainless steel material", "a dining function"],
    "knife": ["a sharp blade", "a handle grip", "a pointed tip", "metal material", "a cutting function"],
    "spoon": ["a shallow bowl", "a long handle", "metal or plastic material", "a scooping function", "dining tableware form"],
    "bowl": ["a curved interior", "a wide open top", "ceramic or plastic material", "a food holding function", "a round shape"],
    "banana": ["a curved yellow shape", "a peelable skin", "soft edible flesh", "a tropical origin", "a nutritious snack form"],
    "apple": ["a round shape", "red or green skin", "crisp flesh", "a core with seeds", "a common fruit form"],
    "sandwich": ["bread slices", "a filling layer", "a handheld form", "savory ingredients", "a portable meal design"],
    "orange": ["a round citrus shape", "an orange peel", "segmented juicy flesh", "a peelable skin", "a tangy flavor profile"],
    "broccoli": ["green florets", "a thick stem", "a vegetable form", "a crunchy texture", "a tree-like appearance"],
    "carrot": ["an orange root", "a long tapered shape", "a crunchy texture", "a sweet flavor", "an underground growth form"],
    "hot dog": ["a long sausage", "a bread bun", "condiment compatibility", "a grilled surface", "a handheld food form"],
    "pizza": ["a round flat base", "tomato sauce layer", "cheese topping", "a baked crust", "customizable toppings"],
    "donut": ["a circular shape", "a hole in center", "fried dough material", "a glazed surface", "a sweet pastry form"],
    "cake": ["layered sponge structure", "frosting decoration", "a sweet dessert form", "a celebration purpose", "a moist texture"],
    "chair": ["a seat surface", "a backrest", "four support legs", "indoor furniture design", "an ergonomic shape"],
    "couch": ["a long seating area", "a backrest", "armrests", "soft cushions", "upholstered surface material"],
    "potted plant": ["a living plant", "a decorative pot", "soil medium", "foliage coverage", "indoor decoration function"],
    "bed": ["a flat sleeping surface", "a mattress", "a headboard", "bedding layers", "a resting function"],
    "dining table": ["a flat surface", "support legs", "a seating capacity", "wood or metal material", "a dining function"],
    "toilet": ["a ceramic bowl", "a seat lid", "a flush mechanism", "plumbing connections", "a sanitation function"],
    "tv": ["a flat screen", "an electronic display", "audio speakers", "a remote control interface", "entertainment functionality"],
    "laptop": ["a foldable screen", "a keyboard base", "a portable form factor", "a touchpad", "battery power capability"],
    "mouse": ["a handheld shape", "click buttons", "a scroll wheel", "a cursor control function", "USB or wireless connectivity"],
    "remote": ["a handheld device", "button controls", "an infrared transmitter", "battery power", "electronic device control function"],
    "keyboard": ["key buttons", "an alphanumeric layout", "a typing interface", "wired or wireless connectivity", "an input device function"],
    "cell phone": ["a touch screen", "a camera lens", "a slim rectangular shape", "portable communication capability", "app interface functionality"],
    "microwave": ["a metal box structure", "a glass door", "a control panel", "a turntable interior", "electromagnetic heating capability"],
    "oven": ["an enclosed chamber", "heating elements", "a glass door", "temperature controls", "baking functionality"],
    "toaster": ["slot openings", "bread slice capacity", "heating elements", "an eject mechanism", "browning control functionality"],
    "sink": ["a basin shape", "a faucet fixture", "a drain opening", "water supply connections", "a washing function"],
    "refrigerator": ["a large metal box", "a cooling system", "interior shelves", "a temperature control system", "a food preservation function"],
    "book": ["paper pages", "a bound spine", "front and back covers", "printed text content", "a knowledge source function"],
    "clock": ["a circular face", "hour markers", "moving hands", "a timekeeping mechanism", "wall-mounted or standing design"],
    "vase": ["a narrow neck", "a wide base", "a flower holding function", "glass or ceramic material", "a decorative form"],
    "scissors": ["two sharp blades", "a pivot joint", "finger holes", "metal material", "a cutting function"],
    "teddy bear": ["a stuffed toy form", "soft plush material", "button eyes", "sewn facial features", "a huggable size"],
    "hair drier": ["a handheld device", "an air nozzle", "a heating element", "a plastic casing", "a hair drying function"],
    "toothbrush": ["a handle grip", "a bristle head", "a cleaning tool design", "oral hygiene functionality", "a compact size"]
}

class DatasetCOCO(Dataset):
    def __init__(self, datapath, fold, transform, clip_transform, split, shot, use_original_imgsize):
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.fold = fold
        self.nfolds = 4
        self.nclass = 80
        self.test_episodes = 1000
        self.benchmark = 'coco'
        self.shot = shot
        self.split_coco = split if split == 'val2014' else 'train2014'
        self.base_path = join(datapath, 'COCO2014')
        self.transform = transform
        self.clip_transform = clip_transform
        self.use_original_imgsize = use_original_imgsize
        self.class_ids = self.build_class_ids()
        self.img_metadata_classwise = self.build_img_metadata_classwise()
        self.img_metadata = self.build_img_metadata()

    def __len__(self):
        return len(self.img_metadata)  if self.split == 'trn' else self.test_episodes

    def __getitem__(self, idx):
        query_img, query_mask, support_imgs, support_masks, query_name, support_names, class_sample, org_qry_imsize, base_query, base_supports = self.load_frame()
        
        # --- SAM2 处理 ---
        sam_query_img = self.transform(query_img)
        query_mask = query_mask.float()
        if not self.use_original_imgsize:
            query_mask = F.interpolate(query_mask.unsqueeze(0).unsqueeze(0).float(), sam_query_img.size()[-2:], mode='nearest').squeeze()
            base_query = F.interpolate(base_query.unsqueeze(0).unsqueeze(0).float(), sam_query_img.size()[-2:], mode='nearest').squeeze()
        
        # --- CLIP 处理 ---
        clip_query_img = self.clip_transform(query_img)
        
        # --- Support 处理 ---
        sam_support_imgs = torch.stack([self.transform(img) for img in support_imgs])
        clip_support_imgs = torch.stack([self.clip_transform(img) for img in support_imgs])
        
        # --- Mask 对齐 ---
        support_masks = torch.stack([
            F.interpolate(m.unsqueeze(0).unsqueeze(0).float(), sam_support_imgs.size()[-2:], mode='nearest').squeeze() 
            for m in support_masks
        ])
        clip_support_masks = torch.stack([
            F.interpolate(m.unsqueeze(0).unsqueeze(0).float(), clip_support_imgs.size()[-2:], mode='nearest').squeeze() 
            for m in support_masks
        ])

        base_supports = torch.stack([
            F.interpolate(m.unsqueeze(0).unsqueeze(0).float(), sam_support_imgs.size()[-2:], mode='nearest').squeeze() 
            for m in base_supports
        ])
        
        # ✨ 核心修改：构建增强提示列表（主提示 + 属性提示）
        cat_name = COCO_CLASSES[int(class_sample)]
        text_prompt = f"a photo of a {cat_name}"  # 标准CLIP零样本提示
        class_attributes = COCO_ATTRIBUTES[cat_name]
        
        # 关键：将主提示作为第一个元素，后接所有属性提示
        attribute_prompts = [text_prompt] + [f"the {cat_name} has {attr}" for attr in class_attributes]
        # print("Attribute Prompts:", attribute_prompts)  # Debug 输出提示列表
        attribute_prompts_str = ", ".join(attribute_prompts)
        return {
            'query_img': sam_query_img,
            'clip_query_img': clip_query_img,
            'query_mask': query_mask,
            'support_imgs': sam_support_imgs,
            'clip_support_imgs': clip_support_imgs,
            'support_masks': support_masks,
            'clip_support_masks': clip_support_masks,
            'base_masks': [base_supports, base_query],
            'class_id': torch.tensor(class_sample),
            'class_names': attribute_prompts_str,          # 保留原始主提示（兼容旧代码）
        }

    def build_class_ids(self):
        nclass_trn = self.nclass // self.nfolds
        if self.fold != -1:
            class_ids_val = [self.fold + self.nfolds * v for v in range(nclass_trn)]
            class_ids_trn = [x for x in range(self.nclass) if x not in class_ids_val]
        else:
            class_ids_val = list(range(self.nclass))
            class_ids_trn = list(range(self.nclass))
        return class_ids_trn if self.split == 'trn' else class_ids_val

    @staticmethod
    def load_pickle(pickle_path):
        with open(pickle_path, 'rb') as fp:
            return pickle.load(fp)

    def build_img_metadata_classwise(self):
        if self.fold != -1:
            with open(f'{self.base_path}/splits/{self.split}/fold{self.fold}.pkl', 'rb') as f:
                return pickle.load(f)
        else:
            split_meta_path = [f'{self.base_path}/splits/{self.split}/fold{idx}.pkl' for idx in range(4)]
            if self.split == 'trn':
                meta0 = self.load_pickle(split_meta_path[0])
                meta1 = self.load_pickle(split_meta_path[1])
                for cid in meta0:
                    if not meta0[cid]:
                        meta0[cid] = meta1[cid]
                return meta0
            else:
                all_meta = [self.load_pickle(p) for p in split_meta_path]
                merged = {k: [] for k in all_meta[0]}
                for meta in all_meta:
                    for cid in meta:
                        if not merged[cid] and meta[cid]:
                            merged[cid] = meta[cid]
                return merged

    def build_img_metadata(self):
        metadata = []
        for imgs in self.img_metadata_classwise.values():
            metadata.extend(imgs)
        return sorted(set(metadata))

    def read_mask(self, name):
        mask_path = os.path.join(self.base_path, 'annotations', name.replace('.jpg', '.png'))
        return torch.tensor(np.array(Image.open(mask_path)))

    def load_frame(self):
        class_sample = np.random.choice(self.class_ids, 1, replace=False)[0]
        query_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
        
        query_img = Image.open(os.path.join(self.base_path, query_name)).convert('RGB')
        query_mask = self.read_mask(query_name)
        org_qry_imsize = query_img.size
        base_query_mask = query_mask.clone()
        
        # [修改] 向量化二值化与过滤 query_mask
        query_mask = (query_mask == class_sample + 1).float()
        
        valid_classes = torch.tensor([c + 1 for c in self.class_ids], device=base_query_mask.device)
        mask_to_keep = torch.isin(base_query_mask, valid_classes)
        base_query_mask[~mask_to_keep] = 0

        # 采样support样本
        support_names = []
        while len(support_names) < self.shot:
            name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
            if name != query_name and name not in support_names:
                support_names.append(name)
                
        support_imgs, support_masks, base_support_masks = [], [], []
        for name in support_names:
            support_imgs.append(Image.open(os.path.join(self.base_path, name)).convert('RGB'))
            mask = self.read_mask(name)
            base_mask = mask.clone()
            
            # [修改] 向量化二值化与过滤 support_mask
            mask = (mask == class_sample + 1).float()
            
            valid_support_mask = torch.isin(base_mask, valid_classes)
            base_mask[~valid_support_mask] = 0
                    
            support_masks.append(mask)
            base_support_masks.append(base_mask)
            
        return query_img, query_mask, support_imgs, support_masks, query_name, support_names, class_sample, org_qry_imsize, base_query_mask, base_support_masks

def build(image_set, args):
    img_size = 640
    
    # SAM2 Transform (ImageNet归一化)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # CLIP Transform (使用CLIP官方预处理)
    try:
        _, clip_preprocess = clip.load("ViT-B/16", device='cuda', jit=False)
    except Exception as e:
        print(f"[WARNING] CLIP load failed in coco.py: {e}. Using fallback transform.")
        clip_preprocess = transform
    
    return DatasetCOCO(
        datapath=args.data_root,
        fold=args.fold,
        transform=transform,
        clip_transform=clip_preprocess,
        shot=args.shots,
        use_original_imgsize=False,
        split=image_set
    )