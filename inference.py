import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from main_model import UNet
import json

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def load_image(image_path, mean, std, device):
    if not os.path.exists(image_path):
        logger.info(f'Image path at {image_path} not found')

    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    return image

def inference(model, test_image_path, mean, std, checkpoint_path='checkpoints/best_checkpoint.pth', device='cuda' if torch.cuda.is_available() else 'cpu'):
    try:
        if not os.path.exists(checkpoint_path):
            logger.info(f'Checkpoint path at {checkpoint_path} not found')
        
        model = model.to(device)
        epoch, loss = model.load_checkpoint(checkpoint_path)
        logger.info(f'Loaded checkpoint from epoch {epoch} with loss {loss:.4f}')

        model.eval()

        input_image = load_image(test_image_path, mean, std, device)

        with torch.no_grad():
            output = model(input_image)
            output = output.squeeze().cpu().numpy()

        mask = (output > 0.5).astype(np.uint8)

        original_image = np.array(Image.open(test_image_path).resize((320, 320)))

        plt.figure(figsize=(15, 5)) 
        plt.subplot(1, 2, 1)
        plt.title('Input image')
        plt.imshow(original_image)
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.title('Predicted segmentation')
        plt.imshow(mask, cmap='gray')
        plt.axis('off')
        
        plt.show()

        return mask
    except Exception as e:
        logger.error(f'Inference failure: {str(e)}')
        raise

if __name__ == '__main__':
    try:
        stats_path = 'checkpoints/dataset_stats.json'
        if not os.path.exists(stats_path):
            logger.error(f'Dataset stats file at {stats_path} not found')
            raise FileNotFoundError(f'Dataset stats file at {stats_path} not found')
        
        with open(stats_path, 'r') as f:
            stats = json.load(f)

        train_mean = stats['mean']
        train_std = stats['std']
        logger.info(f"Loaded mean: {train_mean}")
        logger.info(f"Loaded standard deviation: {train_std}")
    
        model = UNet()
        # path should contain forward slash (/), not back slash (/)
        test_image_path = '' # input an image path
        inference(model, test_image_path=test_image_path, mean=train_mean, std=train_std)
        logger.info("Inference completed successfully")
    except Exception as e:
        logger.error(f"Main failure: {str(e)}")
        raise

            
