from .FPN import (
    train,
    optimizer,
    schedulers,
    loss_objective,
    data_train,
    data_val,
    val_cpu_transform,
    train_cpu_transform,
    gpu_transform,
    label_map,
    anchors,
    model
)
from tops.config import LazyCall as L
from ssd.modeling.backbones import biFPN


backbone = L(biFPN)(
    model_type="resnet34", 
    pretrained=True, 
    output_feature_sizes="${anchors.feature_sizes}")
