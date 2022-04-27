from .FocalLoss import (
    train,
    optimizer,
    schedulers,
    loss_objective,
    # model
    data_train,
    data_val,
    val_cpu_transform,
    train_cpu_transform,
    gpu_transform,
    label_map,
    anchors,
    backbone
)
from tops.config import LazyCall as L
from ssd.modeling.deeper_reg_heads import DeeperRegHeads

model = L(DeeperRegHeads)(
    feature_extractor="${backbone}",
    anchors="${anchors}",
    loss_objective="${loss_objective}",
    num_classes=8 + 1,
    anchor_prob_init=False
)
