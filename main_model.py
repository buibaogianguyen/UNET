import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DepthWiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(DepthWiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x  

class DoubleConvBlock(nn.Module):
    def __init__ (self, in_channels, out_channels, dropout_rate=0.1):
        super(DoubleConvBlock, self).__init__()
        self.conv = nn.Sequential(
            DepthWiseSeparableConv(in_channels, out_channels, 3, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            DepthWiseSeparableConv(out_channels, out_channels, 3, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(dropout_rate)
        )

        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual = None

    def forward(self, x):
        residual = x
        out = self.conv(x)
        if self.residual is not None:
            residual = self.residual(x)
        return out + residual
    
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64,128,256,512,1024]):
        super(UNet, self).__init__()
        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        # Down/Encoder
        for feature in features:
            self.encoder_blocks.append(DoubleConvBlock(in_channels, feature))
            in_channels = feature

        # Bridge
        self.bottleneck = DoubleConvBlock(features[-1], features[-1]*2)

        # Up/Decoder
        for feature in reversed(features):
            self.decoder_blocks.append(nn.Sequential(
                nn.ConvTranspose2d(feature*2, feature,2, 2, 0, 0),
                nn.BatchNorm2d(feature)
                ))
            self.decoder_blocks.append(DoubleConvBlock(feature*2, feature))

        self.final_conv = nn.Sequential(nn.Conv2d(features[0], out_channels, kernel_size=1),
                                        nn.Sigmoid() if out_channels == 1 else nn.Softmax(dim=1)                    
        )

    def forward(self,x):
        skip_connections = []

        # Encoder path
        for decoder_block in self.encoder_blocks:
            x = decoder_block(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Decoder path
        for index in range(0, len(self.decoder_blocks), 2):
            x = self.decoder_blocks[index](x)
            skip_connection = skip_connections[index//2]

            # Size adjustment in case of mismatch
            if x.shape[2:] != skip_connection.shape[2:]:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.decoder_blocks[index+1](concat_skip)

        return self.final_conv(x)

def test():
    x = torch.randn((3,3,160,160))
    model = UNet(in_channels=3,out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape[0] == x.shape[0] and preds.shape[2:] == x.shape[2:]

if __name__ == "__main__":
    test()