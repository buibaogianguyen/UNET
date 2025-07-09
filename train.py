import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import json
import os
from PIL import Image, ImageDraw
import numpy as np
from main_model import UNet
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class CocoDataset(Dataset):
    def __init__(self, root_path, annotation_file, transform=None, target_size = (160,160)):
        self.root_path = root_path
        self.transform = transform
        self.target_size = target_size
        
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)

        self.images = self.coco_data['images']
        self.annotations = self.coco_data['annotations']
        logger.info(f"Loaded {len(self.images)} images with{len(self.annotations)} annotations")

    # DataLoader required
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_metadata = self.images[idx]
        img_path = os.path.join(self.root_path, img_metadata['file name'])
        image = Image.open(img_path).convert('RGB')

        mask = np.zeros((img_metadata['height'], img_metadata['width']), dtype=np.uint8)

        for anno in self.annotations:
            if anno['image_id'] == img_metadata['id']:
                segmentation = anno['segmentation'][0]
                coords = np.array(segmentation).reshape(-1,2).astype(np.int32)
                mask_img = Image.fromarray(mask)
                draw = ImageDraw.Draw(mask_img)
                draw.polygon([tuple(coord) for coord in coords], 1)
                mask = np.array(mask_img)

        mask_img = Image.fromarray(mask)
        mask_img = mask_img.resize(self.target_size, Image.NEAREST)
        mask = np.array(mask_img)

        if self.transform:
            image = self.transform(image)
            mask = torch.from_numpy(mask).float().unsqueeze(0)

        return image, mask
    
