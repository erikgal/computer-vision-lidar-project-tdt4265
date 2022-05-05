# Inherit configs from the default ssd300
import torchvision
from ssd.data import TDT4265Dataset
from tops.config import LazyCall as L
from ssd.data.transforms import (
    ToTensor, Normalize, Resize,
    GroundTruthBoxesToAnchors,  RandomHorizontalFlip, Resize, RandomSampleCrop, ColorJitter)
from .ssd300 import train, anchors, optimizer, schedulers, data_train, data_val, model
from .utils import get_dataset_dir
from ssd.modeling.backbones import FPN
from ssd.modeling.focal_loss import FocalLoss
from ssd.modeling.deeper_reg_heads import DeeperRegHeads



# Keep the model, except change the backbone and number of classes
train.imshape = (128, 1024)
train.image_channels = 3
train.epochs = 50
model.num_classes = 8 + 1  # Add 1 for background class

anchors.feature_sizes = [[32, 256], [16, 128],
                         [8, 64], [4, 32], [2, 16], [1, 8]]
anchors.strides = [[4, 4], [8, 8], [16, 16], [32, 32], [64, 64], [128, 128]]
anchors.min_sizes = [[16, 8], [32, 16], [38, 24],
                     [48, 42], [72, 64], [98, 136], [128, 400]]
anchors.aspect_ratios = [[2, 4], [2, 4], [2, 4], [2, 4], [2, 4], [2, 3]]

train_cpu_transform = L(torchvision.transforms.Compose)(transforms=[
    L(RandomSampleCrop)(),
    L(ToTensor)(),
    L(RandomHorizontalFlip)(),
    L(Resize)(imshape="${train.imshape}"),
    L(GroundTruthBoxesToAnchors)(anchors="${anchors}", iou_threshold=0.5),
])

val_cpu_transform = L(torchvision.transforms.Compose)(transforms=[
    L(ToTensor)(),
    L(Resize)(imshape="${train.imshape}"),
])
gpu_transform = L(torchvision.transforms.Compose)(transforms=[
    L(Normalize)(mean=[0.4765, 0.4774, 0.2259], std=[0.2951, 0.2864, 0.2878])
])
data_train.dataset = L(TDT4265Dataset)(
    img_folder=get_dataset_dir("tdt4265_2022_updated"),
    transform="${train_cpu_transform}",
    annotation_file=get_dataset_dir("tdt4265_2022_updated/train_annotations.json"))
data_val.dataset = L(TDT4265Dataset)(
    img_folder=get_dataset_dir("tdt4265_2022_updated"),
    transform="${val_cpu_transform}",
    annotation_file=get_dataset_dir("tdt4265_2022_updated/val_annotations.json"))
data_val.gpu_transform = gpu_transform
data_train.gpu_transform = gpu_transform

label_map = {idx: cls_name for idx,
             cls_name in enumerate(TDT4265Dataset.class_names)}

fpn_out_channels = 256

backbone = L(FPN)(
    model_type="resnet34", 
    pretrained=True, 
    output_feature_sizes="${anchors.feature_sizes}",
    out_channels=fpn_out_channels
)

loss_objective = L(FocalLoss)(anchors="${anchors}", alpha = [0.01, *[1 for i in range(model.num_classes-1)]])


model = L(DeeperRegHeads)(
    feature_extractor="${backbone}",
    anchors="${anchors}",
    loss_objective="${loss_objective}",
    num_classes=8 + 1,
    anchor_prob_init=True,
    p = 0.99,
    in_ch = fpn_out_channels,
    out_ch = 256,
)

