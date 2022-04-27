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

loss_objective = L(FocalLoss)(anchors="${anchors}", alpha = [0.1, *[1 for i in range(model.num_classes-1)]])



