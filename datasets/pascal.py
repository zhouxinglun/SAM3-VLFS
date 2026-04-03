import os
import random
import numpy as np
import PIL.Image as Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

PASCAL_CLASSES = [
    'background', 'airplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'dining table', 'dog', 'horse', 'motorcycle', 'person',
    'pot plant', 'sheep', 'sofa', 'train', 'monitor.n.04'
]

class DatasetPASCAL(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, use_original_imgsize):
        self.split = 'val' if split in ['val', 'test'] else 'train'
        self.fold = fold
        self.nfolds = 4
        self.nclass = 21
        self.test_episodes = 1000
        self.benchmark = 'pascal_voc'
        self.shot = shot
        self.datapath = datapath
        self.transform = transform
        self.use_original_imgsize = use_original_imgsize
        
        self.base_path = os.path.join(datapath, 'VOCdevkit2012/VOC2012')
        
        sub_class_file = os.path.join(self.base_path, f'fss_list/{self.split}/sub_class_file_list_{fold}.txt')
        with open(sub_class_file, 'r') as f:
            self.sub_class_file_list = eval(f.read())
            
        self.class_ids = list(self.sub_class_file_list.keys())
        self.img_metadata = self.build_img_metadata()
        self.class_names = PASCAL_CLASSES

    def build_img_metadata(self):
        img_metadata = []
        for k in self.sub_class_file_list.keys():
            for img_path, _ in self.sub_class_file_list[k]:
                img_metadata.append(img_path)
        return sorted(list(set(img_metadata)))

    def _clean_path(self, path_str):
        if path_str.startswith('../data/'):
            path_str = path_str[8:]
        return os.path.join(self.datapath, path_str)

    def __len__(self):
        return len(self.img_metadata) if self.split == 'train' else self.test_episodes

    def load_frame(self):
        class_sample = np.random.choice(self.class_ids, 1, replace=False)[0]
        
        query_pair = random.choice(self.sub_class_file_list[class_sample])
        query_img_path = self._clean_path(query_pair[0])
        query_mask_path = self._clean_path(query_pair[1])
        
        query_img = Image.open(query_img_path).convert('RGB')
        query_mask_np = np.array(Image.open(query_mask_path))

        org_qry_imsize = query_img.size
        
        base_query_mask = torch.tensor(query_mask_np).clone()
        query_mask = torch.tensor(query_mask_np).clone()
        
        query_mask[query_mask != class_sample] = 0
        query_mask[query_mask == class_sample] = 1

        all_classes = torch.unique(base_query_mask).tolist()
        for cl_i in all_classes:
            if cl_i not in self.class_ids and cl_i != 0 and cl_i != 255:
                base_query_mask[base_query_mask == cl_i] = 0

        support_pairs = []
        while True:
            support_pair = random.choice(self.sub_class_file_list[class_sample])
            if support_pair[0] != query_pair[0] and support_pair not in support_pairs:
                support_pairs.append(support_pair)
            if len(support_pairs) == self.shot: 
                break

        support_imgs, support_masks, base_support_masks, support_names = [], [], [], []
        for sp in support_pairs:
            sup_img_p = self._clean_path(sp[0])
            sup_mask_p = self._clean_path(sp[1])
            
            sup_img = Image.open(sup_img_p).convert('RGB')
            sup_mask_np = np.array(Image.open(sup_mask_p))
            
            base_support_mask = torch.tensor(sup_mask_np).clone()
            sup_mask = torch.tensor(sup_mask_np).clone()
            
            sup_mask[sup_mask != class_sample] = 0
            sup_mask[sup_mask == class_sample] = 1
            
            all_classes = torch.unique(base_support_mask).tolist()
            for cl_i in all_classes:
                if cl_i not in self.class_ids and cl_i != 0 and cl_i != 255:
                    base_support_mask[base_support_mask == cl_i] = 0
                    
            support_imgs.append(sup_img)
            support_masks.append(sup_mask)
            base_support_masks.append(base_support_mask)
            support_names.append(sup_img_p)

        return query_img, query_mask, support_imgs, support_masks, query_img_path, support_names, class_sample, org_qry_imsize, base_query_mask, base_support_masks

    def __getitem__(self, idx):
        query_img_pil, query_mask, support_imgs_pil, support_masks, query_name, support_names, class_sample, org_qry_imsize, base_query, base_supports = self.load_frame()

        query_img = self.transform(query_img_pil)
        query_mask = query_mask.float()
        
        if not self.use_original_imgsize:
            query_mask = F.interpolate(query_mask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()
            base_query = F.interpolate(base_query.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()

        support_imgs = torch.stack([self.transform(img) for img in support_imgs_pil])
        for midx, smask in enumerate(support_masks):
            support_masks[midx] = F.interpolate(smask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:], mode='nearest').squeeze()
            base_supports[midx] = F.interpolate(base_supports[midx].unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:], mode='nearest').squeeze()
            
        support_masks = torch.stack(support_masks)
        base_supports = torch.stack(base_supports)

        batch = {
            'query_img': query_img,
            'query_mask': query_mask,
            'support_imgs': support_imgs,
            'support_masks': support_masks,
            'base_masks': [base_supports, base_query], 
            'class_id': torch.tensor(class_sample),
            'class_name': self.class_names[class_sample]
        }
        return batch

def build(image_set, args):
    img_size = 1008 
    
    transform = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = DatasetPASCAL(
        datapath=args.data_root, 
        fold=args.fold, 
        transform=transform,
        shot=args.shots, 
        use_original_imgsize=False,
        split=image_set
    )
    return dataset
