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
    def __init__(self, root_path, annotation_file, transform=None, target_size=(160, 160)):
        self.root_path = root_path
        self.transform = transform
        self.target_size = target_size
        
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)
        
        self.images = self.coco_data['images']
        self.annotations = self.coco_data['annotations']
        logger.info(f"Loaded dataset with {len(self.images)} images and {len(self.annotations)} annotations")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = os.path.join(self.root_path, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        
        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        for ann in self.annotations:
            if ann['image_id'] == img_info['id'] and ann['category_id'] == 1:
                segmentation = ann['segmentation'][0]
                points = np.array(segmentation).reshape(-1, 2).astype(np.int32)
                mask_img = Image.fromarray(mask)
                draw = ImageDraw.Draw(mask_img)
                draw.polygon([tuple(point) for point in points], fill=1)
                mask = np.array(mask_img)

        mask_img = Image.fromarray(mask)
        mask_img = mask_img.resize(self.target_size, Image.NEAREST)
        mask = np.array(mask_img, dtype=np.uint8)

        if self.transform:
            image = self.transform(image)
            mask = torch.from_numpy(mask).float().unsqueeze(0)

        return image, mask

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def train_model(model, train_loader, valid_loader, num_epochs=50, device='cuda' if torch.cuda.is_available() else 'cpu', checkpoint_dir='checkpoints'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in valid_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)
        
        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(valid_loader.dataset)
        logger.info(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        checkpoint_path = os.path.join(checkpoint_dir, f'unet_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss
        }, checkpoint_path)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_path = os.path.join(checkpoint_dir, 'best_unet_checkpoint.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss
            }, best_checkpoint_path)
    
    return best_checkpoint_path

if __name__ == '__main__':
    train_dataset = CocoDataset(
        root_path='C:/Users/buiba/UNET/data/train',
        annotation_file='C:/Users/buiba/UNET/data/train/_annotations.coco.json',
        transform=transform,
        target_size=(160, 160)
    )
    valid_dataset = CocoDataset(
        root_path='C:/Users/buiba/UNET/data/valid',
        annotation_file='C:/Users/buiba/UNET/data/valid/_annotations.coco.json',
        transform=transform,
        target_size=(160, 160)
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False, num_workers=0)

    model = UNet(in_channels=3, out_channels=1)
    checkpoint_path = train_model(model, train_loader, valid_loader)
    logger.info(f'Training completed. Best checkpoint saved at {checkpoint_path}')