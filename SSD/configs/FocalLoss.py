from .FPN import (
    train, 
    optimizer, 
    anchors,
    schedulers, 
    model, 
    backbone, 
    data_train, 
    data_val, 
    train_cpu_transform,
    val_cpu_transform,
    gpu_transform,
    label_map
)
from tops.config import LazyCall as L
from ssd.modeling.focal_loss import FocalLoss
from ssd.modeling.backbones import FPN

loss_objective = L(FocalLoss)(anchors="${anchors}", alpha = [0.01, *[1 for i in range(model.num_classes-1)]])

fpn_out_channels = 256

backbone = L(FPN)(
    model_type="resnet34", 
    pretrained=True, 
    output_feature_sizes="${anchors.feature_sizes}",
    out_channels=fpn_out_channels,
)


