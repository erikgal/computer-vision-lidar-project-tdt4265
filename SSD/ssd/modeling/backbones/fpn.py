import torch
from torch import nn
from typing import Tuple, List
import torchvision.models as models
from collections import OrderedDict
import torchvision

class FPN(torch.nn.Module):
        def __init__(self,
            model_type: str, 
            pretrained: bool,
            output_feature_sizes: List[Tuple[int]],
            out_channels: int,
            ):
            
            super().__init__()
            
            self.output_feature_shape = output_feature_sizes
            self.out_channels = [out_channels, out_channels, out_channels, out_channels, out_channels, out_channels]
            
            model = getattr(models, model_type)(pretrained=pretrained)
            print(model)      
            
            self.layer1 = nn.Sequential(
                model.conv1,
                model.bn1,
                model.relu,
                model.maxpool,
                model.layer1
            )
            
            self.layer2 = model.layer2
            
            self.layer3 = model.layer3
            
            self.layer4 = model.layer4
    
            self.layer5 = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1),
                nn.ReLU()
            )

            self.layer6 = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=2, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=0),
                nn.ReLU(),
            )
            
            self.layers = [self.layer1, self.layer2, self.layer3, self.layer4, self.layer5, self.layer6]

            img = torch.zeros((1, 3, 128, 1024))
            
            # output shape of layer 1
            print("\n", "LAYER 1", self.layer1)
            layer1_output = self.layer1(img)
            print(layer1_output.shape, "\n")
            
            # output shape of layer 2
            print("LAYER 2", self.layer2)
            layer2_output = self.layer2(layer1_output)
            print(layer2_output.shape, "\n")
            
            # output shape of layer 3
            print("LAYER 3", self.layer3)
            layer3_output = self.layer3(layer2_output)
            print(layer3_output.shape, "\n")
            
            # output shape of layer 4
            print("LAYER 4", self.layer4)
            layer4_output = self.layer4(layer3_output)
            print(layer4_output.shape, "\n")
            
            # output shape of layer 5
            print("LAYER 5", self.layer5)
            layer5_output = self.layer5(layer4_output)
            print(layer5_output.shape, "\n")
            
            # output shape of layer 6
            print("LAYER 6", self.layer6)
            layer6_output = self.layer6(layer5_output)
            print(layer6_output.shape, "\n")
            
            self.m = torchvision.ops.FeaturePyramidNetwork([64, 128, 256, 512, 512, 512], self.out_channels[0])
            
        
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
            for i, layer in enumerate(self.layers):
                if i == 0:
                    out_features.append(layer(x))
                else:
                    out_features.append(layer(out_features[-1]))
                        
            # Building Feature Pyramide Network
            x = OrderedDict()
            x['0'] = out_features[0]
            x['1'] = out_features[1]
            x['2'] = out_features[2]
            x['3'] = out_features[3]
            x['4'] = out_features[4]
            x['5'] = out_features[5]
            #self.x = x
            
            # compute the FPN on top of x
            self.output = self.m(x)
            i = 0
            for k, v in self.output.items():
                out_features[i] = self.output[f"{i}"]
                i += 1
                

            for idx, feature in enumerate(out_features):
                out_channel = self.out_channels[idx]
                h, w = self.output_feature_shape[idx]
                expected_shape = (out_channel, h, w)

                print(feature.shape[1:], expected_shape)
                assert feature.shape[1:] == expected_shape, \
                    f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
            assert len(out_features) == len(self.output_feature_shape),\
                f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was:{len(out_features)}"
            
            return tuple(out_features)
