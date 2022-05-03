import torch
from torch import nn
from typing import Tuple, List
import torchvision.models as models
from collections import OrderedDict
import torchvision

class BiFPN(nn.Module):
    def __init__(self,  fpn_sizes):
        super(BiFPN, self).__init__()
        
        P1_channels, P2_channels, P3_channels, P4_channels, P5_channels, P6_channels = fpn_sizes
        self.W_bifpn = 256

<<<<<<< HEAD
        self.p6_upsample = nn.Upsample(scale_factor=2, mode='nearest')

=======
>>>>>>> 5d102b41efe871d91314614aea965a02ac5928fd
        self.p5_td_conv  = nn.Conv2d(P5_channels, self.W_bifpn, kernel_size=3, stride=1, bias=True, padding=1)
        self.p5_td_conv_2  = nn.Conv2d(self.W_bifpn, self.W_bifpn, kernel_size=3, stride=1, groups=self.W_bifpn, bias=True, padding=1)
        self.p5_td_act   = nn.ReLU()
        self.p5_td_conv_bn = nn.BatchNorm2d(self.W_bifpn)
        self.p5_td_w1    = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p5_td_w2    = torch.tensor(1, dtype=torch.float, requires_grad=True)
<<<<<<< HEAD
        self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')
=======
>>>>>>> 5d102b41efe871d91314614aea965a02ac5928fd

        self.p4_td_conv  = nn.Conv2d(P4_channels,self.W_bifpn, kernel_size=3, stride=1, bias=True, padding=1)
        self.p4_td_conv_2  = nn.Conv2d(self.W_bifpn,self.W_bifpn, kernel_size=3, stride=1, groups=self.W_bifpn, bias=True, padding=1)
        self.p4_td_act   = nn.ReLU()
        self.p4_td_conv_bn = nn.BatchNorm2d(self.W_bifpn)
        self.p4_td_w1    = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p4_td_w2    = torch.tensor(1, dtype=torch.float, requires_grad=True)

        self.p3_td_conv  = nn.Conv2d(P3_channels, self.W_bifpn, kernel_size=3, stride=1, bias=True, padding=1)
        self.p3_td_conv_2  = nn.Conv2d(self.W_bifpn, self.W_bifpn, kernel_size=3, stride=1, groups=self.W_bifpn, bias=True, padding=1)
        self.p3_td_act   = nn.ReLU()
        self.p3_td_conv_bn = nn.BatchNorm2d(self.W_bifpn)
        self.p3_td_w1    = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p3_td_w2    = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p4_upsample   = nn.Upsample(scale_factor=2, mode='nearest')

        self.p2_td_conv = nn.Conv2d(P2_channels, self.W_bifpn, kernel_size=3, stride=1, bias=True, padding=1)
        self.p2_td_conv_2 = nn.Conv2d(self.W_bifpn, self.W_bifpn, kernel_size=3, stride=1, groups=self.W_bifpn, bias=True, padding=1)
        self.p2_td_act   = nn.ReLU()
        self.p2_td_conv_bn = nn.BatchNorm2d(self.W_bifpn)
        self.p2_td_w1   = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p2_td_w2   = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p3_upsample  = nn.Upsample(scale_factor=2, mode='nearest')

        self.p1_out_conv = nn.Conv2d(P1_channels, self.W_bifpn, kernel_size=3, stride=1, bias=True, padding=1)
        self.p1_out_conv_2 = nn.Conv2d(self.W_bifpn, self.W_bifpn, kernel_size=3, stride=1, groups=self.W_bifpn, bias=True, padding=1)
        self.p1_out_act   = nn.ReLU()
        self.p1_out_conv_bn = nn.BatchNorm2d(self.W_bifpn)
        self.p1_out_w1   = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p1_out_w2   = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p2_upsample  = nn.Upsample(scale_factor=2, mode='nearest')

        self.p2_out_conv = nn.Conv2d(self.W_bifpn, self.W_bifpn, kernel_size=3, stride=1, groups=self.W_bifpn, bias=True, padding=1)
        self.p2_out_act   = nn.ReLU()
        self.p2_out_conv_bn = nn.BatchNorm2d(self.W_bifpn)
        self.p2_out_w1   = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p2_out_w2   = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p2_out_w3   = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p1_downsample= nn.MaxPool2d(kernel_size=2)

        self.p3_out_conv = nn.Conv2d(self.W_bifpn, self.W_bifpn, kernel_size=3, stride=1, groups=self.W_bifpn, bias=True, padding=1)
        self.p3_out_act   = nn.ReLU()
        self.p3_out_conv_bn = nn.BatchNorm2d(self.W_bifpn)
        self.p3_out_w1   = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p3_out_w2   = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p3_out_w3   = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p2_downsample= nn.MaxPool2d(kernel_size=2)

        self.p4_out_conv = nn.Conv2d(self.W_bifpn, self.W_bifpn, kernel_size=3, stride=1, groups=self.W_bifpn, bias=True, padding=1)
        self.p4_out_act   = nn.ReLU()
        self.p4_out_conv_bn = nn.BatchNorm2d(self.W_bifpn)
        self.p4_out_w1   = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p4_out_w2   = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p4_out_w3   = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p3_downsample= nn.MaxPool2d(kernel_size=2)

        self.p5_out_conv = nn.Conv2d(self.W_bifpn,self.W_bifpn, kernel_size=3, stride=1, groups=self.W_bifpn, bias=True, padding=1)
        self.p5_out_act   = nn.ReLU()
        self.p5_out_conv_bn = nn.BatchNorm2d(self.W_bifpn)
        self.p5_out_w1   = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p5_out_w2   = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p5_out_w3   = torch.tensor(1, dtype=torch.float, requires_grad=True)
<<<<<<< HEAD
        self.p4_downsample = nn.MaxPool2d(kernel_size=2)
=======
>>>>>>> 5d102b41efe871d91314614aea965a02ac5928fd

        self.p6_out_conv = nn.Conv2d(P6_channels, self.W_bifpn, kernel_size=3, stride=1, bias=True, padding=1)
        self.p6_out_conv_2 = nn.Conv2d(self.W_bifpn,self.W_bifpn, kernel_size=3, stride=1, groups=self.W_bifpn, bias=True, padding=1)
        self.p6_out_act  = nn.ReLU()
        self.p6_out_conv_bn = nn.BatchNorm2d(self.W_bifpn)
        self.p6_out_w1   = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p6_out_w2   = torch.tensor(1, dtype=torch.float, requires_grad=True)
<<<<<<< HEAD
        self.p5_downsample = nn.MaxPool2d(kernel_size=2)

    def forward(self, inputs):
        epsilon = 0.0001
        P1, P2, P3, P4, P5, P6 = inputs['0'], inputs['1'], inputs['2'], inputs['3'], inputs['4'], inputs['5']
        print(P1.shape, P2.shape, P3.shape, P4.shape, P5.shape, P6.shape) 

        P6_td  = self.p6_out_conv(P6)
        P5_td_inp = self.p5_td_conv(P5)

        P5_td = self.p5_td_conv_2((self.p5_td_w1 * P5_td_inp + self.p5_td_w2 * self.p6_upsample(P6_td)) /
                                 (self.p5_td_w1 + self.p5_td_w2 + epsilon))

        P5_td = self.p5_td_act(P5_td)
        P5_td = self.p5_td_conv_bn(P5_td)
         
        P4_td_inp = self.p4_td_conv(P4)
        P4_td = self.p4_td_conv_2((self.p4_td_w1 * P4_td_inp + self.p4_td_w2 * self.p5_upsample(P5_td)) /
=======

    def forward(self, inputs):
        epsilon = 0.0001
        P1, P2, P3, P4, P5, P6 = inputs

        P6_td  = self.p6_out_conv(P6)

        P5_td_inp = self.p5_td_conv(P5)
        P5_td = self.p5_td_conv_2((self.p5_td_w1 * P5_td_inp + self.p5_td_w2 * P6_td) /
                                 (self.p5_td_w1 + self.p5_td_w2 + epsilon))

                        
        P5_td = self.p5_td_act(P5_td)
        P5_td = self.p5_td_conv_bn(P5_td)

         
        P4_td_inp = self.p4_td_conv(P4)
        P4_td = self.p4_td_conv_2((self.p4_td_w1 * P4_td_inp + self.p4_td_w2 * P5_td) /
>>>>>>> 5d102b41efe871d91314614aea965a02ac5928fd
                                 (self.p4_td_w1 + self.p4_td_w2 + epsilon))
        P4_td = self.p4_td_act(P4_td)
        P4_td = self.p4_td_conv_bn(P4_td)

        P3_td_inp = self.p3_td_conv(P3)
        P3_td = self.p3_td_conv_2((self.p3_td_w1 * P3_td_inp + self.p3_td_w2 * self.p4_upsample(P4_td)) /
                                 (self.p3_td_w1 + self.p3_td_w2 + epsilon))
        P3_td = self.p3_td_act(P3_td)
        P3_td = self.p3_td_conv_bn(P3_td)

        P2_td_inp = self.p2_td_conv(P2)
        P2_td = self.p2_td_conv_2((self.p2_td_w1 * P2_td_inp + self.p2_td_w2 * self.p3_upsample(P3_td)) /
                                 (self.p2_td_w1 + self.p2_td_w2 + epsilon))
        P2_td = self.p2_td_act(P2_td)
        P2_td = self.p2_td_conv_bn(P2_td)

        P1_td  = self.p1_out_conv(P1)
        P1_out = self.p1_out_conv_2((self.p1_out_w1 * P1_td + self.p1_out_w2 * self.p2_upsample(P2_td)) /
                                 (self.p1_out_w1 + self.p1_out_w2 + epsilon))
        P1_out = self.p1_out_act(P1_out)
        P1_out = self.p1_out_conv_bn(P1_out)

        P2_out = self.p2_out_conv((self.p2_out_w1 * P2_td_inp  + self.p2_out_w2 * P2_td + self.p2_out_w3 * self.p1_downsample(P1_out) )
                                    / (self.p2_out_w1 + self.p2_out_w2 + self.p2_out_w3 + epsilon))
        P2_out = self.p2_out_act(P2_out)
        P2_out = self.p2_out_conv_bn(P2_out)

        P3_out = self.p3_out_conv((self.p3_out_w1 * P3_td_inp  + self.p3_out_w2 * P3_td + self.p3_out_w3 * self.p2_downsample(P2_out) )
                                    / (self.p3_out_w1 + self.p3_out_w2 + self.p3_out_w3 + epsilon))
        P3_out = self.p3_out_act(P3_out)
        P3_out = self.p3_out_conv_bn(P3_out)

        P4_out = self.p4_out_conv((self.p4_out_w1 * P4_td_inp  + self.p4_out_w2 * P4_td + self.p4_out_w3 * self.p3_downsample(P3_out) )
                                    / (self.p4_out_w1 + self.p4_out_w2 + self.p4_out_w3 + epsilon))
        P4_out = self.p4_out_act(P4_out)
        P4_out = self.p4_out_conv_bn(P4_out)
        
<<<<<<< HEAD
        P5_out = self.p5_out_conv((self.p5_out_w1 * P5_td_inp + self.p5_out_w2 * P5_td + self.p5_out_w3 * self.p4_downsample(P4_out) )
                                    / (self.p5_out_w1 + self.p5_out_w2 + self.p5_out_w3 + epsilon))
        P5_out = self.p5_out_act(P5_out)
        P5_out = self.p5_out_conv_bn(P5_out)
        
        P6_out = self.p6_out_conv_2((self.p6_out_w1 * P6_td + self.p6_out_w2 * self.p5_downsample(P5_out)) /
=======
        P5_out = self.p5_out_conv((self.p5_out_w1 * P5_td_inp + self.p5_out_w2 * P5_td + self.p5_out_w3 * (P4_out) )
                                    / (self.p5_out_w1 + self.p5_out_w2 + self.p5_out_w3 + epsilon))
        P5_out = self.p5_out_act(P5_out)
        P5_out = self.p5_out_conv_bn(P5_out)


        P6_out = self.p6_out_conv_2((self.p6_out_w1 * P6_td + self.p6_out_w2 * P5_out) /
>>>>>>> 5d102b41efe871d91314614aea965a02ac5928fd
                                 (self.p6_out_w1 + self.p6_out_w2 + epsilon))
        P6_out = self.p6_out_act(P6_out)
        P6_out = self.p6_out_conv_bn(P6_out)
        

        return [P1_out, P2_out, P3_out, P4_out, P5_out, P6_out]

<<<<<<< HEAD
=======
        fpn = BiFPN([40, 112, 192, 192, 1280])

        c1 = torch.randn([1, 40, 64, 64])
        c2 = torch.randn([1, 112, 32, 32])
        c3 = torch.randn([1, 192, 16, 16])
        c4 = torch.randn([1, 192, 16, 16])
        c5 = torch.randn([1, 1280, 16, 16])

        feats = [c1, c2, c3, c4, c5]
        output = fpn.forward(feats)

>>>>>>> 5d102b41efe871d91314614aea965a02ac5928fd
class biFPN(torch.nn.Module):
        def __init__(self,
            model_type: str, 
            pretrained: bool,
            output_feature_sizes: List[Tuple[int]]):
            
            super().__init__()
            
            self.output_feature_shape = output_feature_sizes
            self.out_channels = [256, 256, 256, 256, 256, 256]
            
            model = getattr(models, model_type)(pretrained=pretrained)
<<<<<<< HEAD
=======
            print(model)      
>>>>>>> 5d102b41efe871d91314614aea965a02ac5928fd
            
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
<<<<<<< HEAD
            
            self.m = BiFPN([64, 128, 256, 512, 512, 512])

=======

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
            
            self.m = torchvision.ops.FeaturePyramidNetwork([64, 128, 256, 512, 512, 512], 256)
>>>>>>> 5d102b41efe871d91314614aea965a02ac5928fd
            
        
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
            
            # compute the FPN on top of x
<<<<<<< HEAD
            out_features = self.m.forward(x)
=======
            output = self.m(x)
            i = 0
            for k, v in output.items():
                out_features[i] = output[f"{i}"]
                i += 1
                
>>>>>>> 5d102b41efe871d91314614aea965a02ac5928fd

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
