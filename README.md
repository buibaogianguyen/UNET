# U-Net Model for Binary Image Segmentation
This project implements a PyTorch based U-Net model that processes input images and outputs a single-channel segmentation mask



# Navigation
- [Research Paper - U-Net: Convolutional Networks for Biomedical Image Segmentation](#research-paper-u-net)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)


# Research Paper - U-Net: Convolutional Networks for Biomedical Image Segmentation <a id="research-paper-u-net"></a>
This model is based on the U-Net deep-learning architecture introduced in the 2015 research paper [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597), authored by Olaf Ronneberger, Philipp Fischer, and Thomas Brox. This model is not entirely based on the architecture in the paper and introduces modifications to create a more efficient variant of U-Net, tailored to binary image segmentation. Modifications include depthwise seperable convolutions, batch normalization, LeakyReLU, et cetera.

<p align="center">
<img src="https://i.postimg.cc/nhF6pvPn/Screenshot-2025-07-10-161302.png" width="800">
</p>

# Requirements

``` bash
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
matplotlib>=3.7.0
pillow>=9.5.0
```

- requirements.txt.

# Project Structure

``` bash
UNET/
├── main_model.py
├── train.py          # Training script
├── inference.py      # Inference script
├── checkpoints/ # Will be available after training
│   └── best_checkpoint.pth  # Lowest validation loss epoch
│   └── unet_training_epoch_1.pth  # Save for each epoch
│   └── unet_training_epoch_2.pth  # Save for each epoch
│   └── dataset_stats.json  # Holds the calculated dataset mean and std for any dataset + best validation loss
├── requirements.txt   # Dependencies
└── LICENSE

```

# Setup

Clone the repository:
``` cmd
git clone https://github.com/buibaogianguyen/UNET.git
cd UNET
```


Install dependencies:
``` cmd
pip install -r requirements.txt
```
If using a GPU, ensure the PyTorch version matches your CUDA toolkit (e.g., for CUDA 11.8, install torch>=2.0.0+cu118). Check PyTorch's official site for CUDA-specific installation.


Prepare the dataset:

Make sure datasets containing images and COCO annotations for each of them are available for training and validation:
``` bash
# Example
data/
├── train/ # Training dataset
│   └── annotations.coco.json  # Annotations
│   └── image1.png
│   └── image2.png
├── valid/ # Validation dataset
│   └── annotations.coco.json  # Annotations
│   └── image1.png
│   └── image2.png
```
Paths to the general folder housing the images for each the training dataset and validation dataset and paths to the annotations of each will be needed for input in ```train.py```
You may use this sample dataset: https://universe.roboflow.com/lesley-natrop-zgywz/road-detection-segmentation



Input paths for ```train.py``` in between the single quotations:
``` python
    # all paths should contain forward slash (/), not back slash (\)
    train_dataset = CocoDataset(
        root_path = '', # input training folder path that includes training images and annotations
        annotation_file = '', # input the specific annotation file inside of the training folder path
        transform=None,
        target_size=(320,320)
    )
    valid_dataset = CocoDataset(
        root_path = '', # input validation folder path that includes validation images and annotations
        annotation_file = '', # input the specific annotation file inside of the validation folder path
        transform=None,
        target_size=(320,320)
    )
```
Input an image path for ```inference.py``` to test run the model in between the single quotations:
``` python
# path should contain forward slash (/), not back slash (/)
        test_image_path = '' # input an image path
```


# Usage
## Training
To train the model, run:

``` cmd
python train.py
```

Configurations based on your device and dataset:
- Adjust epochs in ```train.py```
- Modify batch_size in the DataLoader (default is 8)
- This model supports configurable image resolutions for training and inference, the default is 320x320. You can manually change the resolution by changing all .resize((320,320)), transforms.Resize((320,320)), and target_size=(320,320) functions across ```train.py``` and ```inference.py```. You can change it to other square values like 160x160 or 640x640. 

## Inference

Prepare a test image:
Place a test image in the project directory or a subdirectory. (Optional, image can be anywhere as long as the path is accessible)
Example: your image is in C:\Users\me\UNET\data\test

Update the inference code:
``` python
test_image_path = 'C:/Users/me/UNET/data/test/image1.png' # use forward slashes (/)
```

Run inference
``` cmd
python inference.py
```

License
This project is licensed under the MIT License. See the LICENSE file for details.
Contributing
Contributions are welcome! Please open an issue or submit a pull request with improvements or bug fixes.
