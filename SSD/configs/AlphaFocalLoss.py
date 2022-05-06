from .WeightInit import (
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
    backbone,
    model
)
from ssd.modeling.focal_loss import FocalLoss
from tops.config import LazyCall as L
from ssd.modeling.deeper_reg_heads import DeeperRegHeads

loss_objective = L(FocalLoss)(anchors="${anchors}", alpha = [0.01, 1/0.50008, 7.0, 4.0, 0, 1/0.60289, 3, 1/0.29090, 4])

model = L(DeeperRegHeads)(
    feature_extractor="${backbone}",
    anchors="${anchors}",
    loss_objective="${loss_objective}",
    num_classes=8 + 1,
    anchor_prob_init=True,
    p = 0.99,
    in_ch = 256,
    out_ch = 256,
)
