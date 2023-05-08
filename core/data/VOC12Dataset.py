from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import math
import os
import torch
import random
import torchvision.transforms.functional as F
from torchvision import transforms
np.set_printoptions(threshold=np.inf)


IMG_FOLDER_NAME = "JPEGImages"
SAL_FOLDER_NAME = "SALImages"
SEG_FOLDER_NAME = "SegmentationClassAug"
SP_FOLDER_NAME = "voc_superpixels"
IGNORE = 255


cls_labels_dict = np.load('metadata/voc12/cls_labels.npy', allow_pickle=True).item()


def load_img_name_list(dataset_path):
    img_name_list = np.loadtxt(dataset_path, dtype=str)
    # img_name_list = np.loadtxt(dataset_path, dtype=str)
    # print(img_name_list[0])
    return img_name_list

def load_image_label_list_from_npy(img_name_list):
    return np.array([cls_labels_dict[img_name] for img_name in img_name_list])

def get_img_path(img_name, voc12_root):
    return os.path.join(voc12_root, IMG_FOLDER_NAME, img_name + '.jpg')

def get_sal_path(img_name, voc12_root):
    return os.path.join(voc12_root, SAL_FOLDER_NAME, img_name + '.png')

def get_seg_path(img_name, voc12_root):
    return os.path.join(voc12_root, SEG_FOLDER_NAME, img_name + '.png')

def get_sp_path(image_name, voc12_root):
    return os.path.join(voc12_root, SP_FOLDER_NAME, image_name + '.png')


class VOC12Dataset(Dataset):

    def __init__(self, img_name_list_path, voc12_root,
                 input_size=224):
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.label_list = load_image_label_list_from_npy(self.img_name_list)
        self.voc12_root = voc12_root

        self.cls_counts = torch.tensor([10231,680,583,808,540,810,467,1277,1124,1351,338,688,1316,524,
                                        568,4600,607,357,739,587,641.])

        self.patch_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor((0.485, 0.456, 0.406)),
                std=torch.tensor((0.229, 0.224, 0.225)))
        ])
        self.input_size = (input_size, input_size)
        self.small_input_size = (int(input_size / 8), int(input_size / 8))

    def __len__(self):
        return len(self.img_name_list)
    
    @staticmethod
    def get_params(img, scale=(0.75, 1.0), ratio=(3. / 4., 4. / 3.)):
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        area = img.size[0] * img.size[1]

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = img.size[0] / img.size[1]
        if in_ratio < min(ratio):
            w = img.size[0]
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = img.size[1]
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = img.size[0]
            h = img.size[1]
        i = (img.size[1] - h) // 2
        j = (img.size[0] - w) // 2
        return i, j, h, w

    def __getitem__(self, idx):
        name_str = self.img_name_list[idx]

        img = Image.open(get_img_path(name_str, self.voc12_root)).convert('RGB')
        sal = Image.open(get_sal_path(name_str, self.voc12_root)).convert('RGB')
        seg = Image.open(get_seg_path(name_str, self.voc12_root))
        sp = Image.open(get_sp_path(name_str, self.voc12_root)) #[h,w,3]
        # print(np.array(img).shape, np.array(sal).shape, np.array(seg).shape)
        label = torch.from_numpy(self.label_list[idx])

        img, sal, seg, sp = random_lr_flip((img,sal,seg,sp))
        
        i, j, h, w = self.get_params(img)
        img = F.resized_crop(img, i,j,h,w, self.input_size, F.InterpolationMode.BICUBIC)
        sal = F.resized_crop(sal, i,j,h,w, self.small_input_size, F.InterpolationMode.NEAREST)
        seg = F.resized_crop(seg, i,j,h,w, self.input_size, F.InterpolationMode.NEAREST)
        sp = F.resized_crop(sp, i,j,h,w, self.small_input_size, F.InterpolationMode.NEAREST)

        seg = np.array(seg) #[h,w]
        sal = np.asarray(sal).mean(axis=-1) / 255 #[h,w]
        sp = np.array(sp)[:,:,-1] #[h,w]
        img = self.patch_transform(img) #[3,h/8,w/8]

        return name_str, img, label, torch.tensor(sal).float(), torch.tensor(seg).float(), torch.tensor(sp).float()


def random_lr_flip(img, p=0.5):
    if random.random() > p:
        return [m.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for m in img]
    else:
        return img

def bulid_voc_datasets(args):
    return VOC12Dataset(args.name_list, args.voc12_root, args.input_size)
    
if __name__ == "__main__":
    ds = VOC12Dataset('metadata/voc12/train_aug.txt', voc12_root='/home/dogglas/mil/datasets/VOC2012')
    img_id, img, label, sal, seg_label, sp = ds[0]
