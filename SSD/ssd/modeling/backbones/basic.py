import torch
from torch import nn
from typing import Tuple, List
from torchvision import transforms


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
    def __init__(self,
            output_channels: List[int],
            image_channels: int,
            output_feature_sizes: List[Tuple[int]]):

        super().__init__()
        self.out_channels = output_channels
        self.output_feature_shape = output_feature_sizes

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        self.feature_extractor = [None] * 6

        ######################################################################
        self.feature_extractor_a = nn.Sequential(

        # 1st output => Resolution: 38 × 38
        transforms.ColorJitter(),

        nn.Conv2d(image_channels, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(num_features=64),
        nn.ReLU(),
        nn.MaxPool2d(2, stride=2),
            
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), 
        nn.BatchNorm2d(num_features=128),
        nn.ReLU(),
        nn.MaxPool2d(2, stride=2), 
        
        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(num_features=256), 
        nn.ReLU(),
            
        nn.Conv2d(256, output_channels[0], kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(num_features=output_channels[0]), 
        nn.ReLU()).to(device)
        ######################################################################
        
        # 2nd output => Resolution: 19 × 19
        self.feature_extractor_b = nn.Sequential(
        nn.ReLU(),
        nn.Conv2d(output_channels[0], 512, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(num_features=512), 
        nn.ReLU(),
        nn.Conv2d(512, output_channels[1], kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(num_features= output_channels[1]),  
        nn.ReLU()).to(device)
        ######################################################################

        # 3rd output => Resolution: 10 × 10
        self.feature_extractor_c = nn.Sequential(
        nn.ReLU(),
        nn.Conv2d(output_channels[1], 1024, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(num_features=1024),  
        nn.ReLU(),
        nn.Conv2d(1024, output_channels[2], kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(num_features = output_channels[2]),   
        nn.ReLU()).to(device)
        ######################################################################

        # 4th output => Resolution: 5 × 5
        self.feature_extractor_d = nn.Sequential(
        nn.ReLU(),
        nn.Conv2d(output_channels[2], 1024, kernel_size=3, stride=1, padding=1), 
        nn.BatchNorm2d(num_features=1024),  
        nn.ReLU(),
        nn.Conv2d(1024, output_channels[3], kernel_size=3, stride=2, padding=1), 
        nn.BatchNorm2d(num_features = output_channels[3]),   
        nn.ReLU()).to(device)
        ######################################################################

        # 5th output => Resolution: 3 × 3
        self.feature_extractor_e = nn.Sequential(
        nn.ReLU(),
        nn.Conv2d(output_channels[3], 512, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(num_features=512),   
        nn.ReLU(),
        nn.Conv2d(512, output_channels[4], kernel_size=3, stride=2, padding=1), 
        nn.BatchNorm2d(num_features= output_channels[4]),   
        nn.ReLU()).to(device)
        ######################################################################

        # 6th output => Resolution: 1 × 1
        self.feature_extractor_f = nn.Sequential(
        nn.ReLU(),
        nn.Conv2d(output_channels[4], 256, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(num_features=256),    
        nn.ReLU(),
        nn.Conv2d(256, output_channels[5], kernel_size=3, stride=1, padding=0), 
        #nn.BatchNorm2d(num_features= output_channels[5]),   
        nn.ReLU()).to(device)
        ######################################################################


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

        # for i, nn in enumerate(self.feature_extractor):
        #     x = nn(x)
        #     out_features.append(x)
        x = self.feature_extractor_a(x)
        out_features.append(x)

        x = self.feature_extractor_b(x)
        out_features.append(x)

        x = self.feature_extractor_c(x)
        out_features.append(x)

        x = self.feature_extractor_d(x)
        out_features.append(x)

        x = self.feature_extractor_e(x)
        out_features.append(x)

        x = self.feature_extractor_f(x)
        out_features.append(x)


        for idx, feature in enumerate(out_features):
            out_channel = self.out_channels[idx]
            h, w = self.output_feature_shape[idx]
            expected_shape = (out_channel, h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        assert len(out_features) == len(self.output_feature_shape),\
            f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"
        return tuple(out_features)

