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

epochs = 50 # adjust as needed
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        logger.info(f"Loaded {len(self.images)} images with {len(self.annotations)} annotations")

    # DataLoader required
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_metadata = self.images[idx]
        img_path = os.path.join(self.root_path, img_metadata['file_name'])
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
        else:
            mask = torch.from_numpy(mask).float().unsqueeze(0)

        return image, mask
    
def train_model(model, train_loader, val_loader, epochs, device, checkpoint_dir='checkpoints'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    lowest_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images,masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)

        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)
        logger.info(f'Epoch: {epoch+1}/{epochs}\nTrain Loss: {train_loss:.4f}\nValidation Loss: {val_loss:.4f}')

        checkpoint_path = os.path.join(checkpoint_dir, f'unet_training_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict' : model.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
            'val_loss' : val_loss
        }, checkpoint_path)

        if val_loss < lowest_val_loss:
            lowest_val_loss = val_loss

            best_checkpoint_path = os.path.join(checkpoint_dir, 'best_checkpoint.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss
            }, best_checkpoint_path)
            logger.info(f'Saved best/lowest validation loss checkpoint with loss: {val_loss:.4f}')

    return best_checkpoint_path

def mean_std(dataset, transform):
    channel_sum = np.zeros(3)
    channel_sum_sq = np.zeros(3)
    pixels = 0

    for idx in range(len(dataset)):
        image, _ = dataset[idx]
        image = transform(image).numpy()
        channel_sum += np.sum(image, axis=(1, 2))
        channel_sum_sq += np.sum(image**2, axis=(1, 2))
        pixels += image.shape[1] * image.shape[2]

    # image dataset stats formula
    mean = channel_sum / pixels
    std = np.sqrt((channel_sum_sq / pixels) - (mean ** 2))
    return mean, std

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor()
    ])

    train_dataset = CocoDataset(
        root_path = '', # input training folder path that includes training images and annotations
        annotation_file = '', # input the specific annotation file inside of the training folder path
        transform=None,
        target_size=(160,160)
    )
    valid_dataset = CocoDataset(
        root_path = '', # input validation folder path that includes validation images and annotations
        annotation_file = '', # input the specific annotation file inside of the validation folder path
        transform=None,
        target_size=(160,160)
    )

    train_mean, train_std = mean_std(train_dataset, transform)
    logger.info(f"Calculated mean: {train_mean.tolist()}")
    logger.info(f"Calculated standard deviation: {train_std.tolist()}")

    stats = {
        'mean' : train_mean.tolist(),
        'std' : train_std.tolist()
    }
    stats_path = os.path.join('checkpoints', 'dataset_stats.json')
    os.makedirs('checkpoints', exist_ok=True)
    with open(stats_path, 'w') as f:
        json.dump(stats, f)
    logger.info(f'Saved dataset mean and standard deviation to {stats_path}')

    # apply normalization
    transform = transforms.Compose([
        transforms.Resize((160,160)),
        transforms.ToTensor(),
        transforms.Normalize(mean = train_mean, std = train_std)
    ])
    train_dataset.transform = transform
    valid_dataset.transform = transform

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False, num_workers=0)

    model = UNet(in_channels=3, out_channels=1)
    checkpoint_path = train_model(model, train_loader, val_loader, epochs, device)
    logger.info(f'Training done. Best/Lowest validation loss checkpoint saved at {checkpoint_path}')