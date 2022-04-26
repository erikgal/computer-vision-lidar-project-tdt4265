from ssd.modeling import AnchorBoxes
from tops.config import LazyCall as L
from ssd.modeling.backbones import FPN_KRISTIAN
from ssd.modeling.focal_loss import FocalLoss
from .FPN import (
    train, 
    optimizer, 
    anchors,
    schedulers, 
    #loss_objective,
    model, 
    backbone, 
    data_train, 
    data_val, 
    train_cpu_transform,
    val_cpu_transform,
    gpu_transform,
    label_map
)

loss_objective = L(FocalLoss)(anchors="${anchors}", alpha = [0.1,*[1 for i in range(model.num_classes-1)]])



