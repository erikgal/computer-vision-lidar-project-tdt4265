import torch 
#from torch import nn
from typing import Tuple, List


class BasicModel(torch.nn.Module):
    """
    This is a basic backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[3], 3, 3),
     shape(-1, output_channels[4], 1, 1)]
    """

    # Each convolutional layer has a filter size of 3 × 3 and padding of 1. 
    # Each MaxPool2D layer has a stride of 2 and a kernel size of 2 × 2. 
    # Note that output channels[i] is a variable that you can set in the config file (By default this is: [128, 256, 128, 128, 64, 64]). 
    # The last convolution should have a stride of 1, padding of 0, and a kernel size of 3 × 3

    def __init__(self,
            output_channels: List[int],
            image_channels: int,
            output_feature_sizes: List[Tuple[int]]):
        super().__init__()
        self.out_channels = output_channels
        self.output_feature_shape = output_feature_sizes

        #Task 4a,b
        self.conv_layer1= torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=image_channels, out_channels=32, kernel_size=3, stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2, stride = 2),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=64, out_channels=output_channels[0], kernel_size=3, stride=2,padding=1),
            torch.nn.ReLU()
        )
        self.conv_layer2 = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=output_channels[0], out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=128, out_channels=output_channels[1], kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU()
        )
        self.conv_layer3 = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=output_channels[1], out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=256, out_channels=output_channels[2], kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU()
        )
        self.conv_layer4 = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=output_channels[2], out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=128, out_channels=output_channels[3], kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU()
        )
        self.conv_layer5 = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=output_channels[3], out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=128, out_channels=output_channels[4], kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU()
        )
        self.conv_layer6 = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=output_channels[4], out_channels=128, kernel_size=2, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=128, out_channels=output_channels[5], kernel_size=2, stride=1, padding=0),
            torch.nn.ReLU()
        )

        self.feature_extractor = [self.conv_layer1, self.conv_layer2, self.conv_layer3, self.conv_layer4, self.conv_layer5, self.conv_layer6]
        self.resetParameters()

    def resetParameters(self):
        for layer in self.feature_extractor:
            if isinstance(layer, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(layer.weight, a=0, mode="fan_in", nonlinearity="leaky_relu")
                layer.bias.data.fill(0.0)
            elif isinstance(layer, torch.nn.BatchNorm2d):
                torch.nn.constant_(layer.weight, 1)
                torch.nn.constant_(layer.bias, 0)

            

    def forward(self, x):
        """
        The forward functiom should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[3], 3, 3),
            shape(-1, output_channels[4], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """
        out_features = []
        for conv_layer in self.feature_extractor:
            out = conv_layer(x)
            out_features.append(out)
            x = out

        for idx, feature in enumerate(out_features):
            out_channel = self.out_channels[idx]
            h, w = self.output_feature_shape[idx]
            expected_shape = (out_channel, h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        assert len(out_features) == len(self.output_feature_shape),\
            f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"
        return tuple(out_features)