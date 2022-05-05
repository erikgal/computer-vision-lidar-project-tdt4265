from ssd.modeling import AnchorBoxes
from tops.config import LazyCall as L
from ssd.modeling.backbones import FPN
from .tdt4265 import (
    train, 
    optimizer, 
    anchors,
    schedulers, 
    loss_objective,
    model, 
    data_train, 
    data_val, 
    train_cpu_transform,
    val_cpu_transform,
    gpu_transform,
    label_map
)

backbone = L(FPN)(
    model_type="resnet34", 
    pretrained=True, 
    output_feature_sizes="${anchors.feature_sizes}",
    out_channels=256,
    )
