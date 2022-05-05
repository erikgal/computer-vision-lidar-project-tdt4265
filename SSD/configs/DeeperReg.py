from .FocalLoss import (
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
    backbone
)
from tops.config import LazyCall as L
from ssd.modeling.deeper_reg_heads import DeeperRegHeads
from ssd.modeling.backbones import FPN


anchors.aspect_ratios = [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]]

fpn_out_channels = 256

backbone = L(FPN)(
    model_type="resnet34", 
    pretrained=True, 
    output_feature_sizes="${anchors.feature_sizes}",
    out_channels=fpn_out_channels,
)

model = L(DeeperRegHeads)(
    feature_extractor="${backbone}",
    anchors="${anchors}",
    loss_objective="${loss_objective}",
    num_classes=8 + 1,
    anchor_prob_init=False,
    p = 0.99,
    in_ch = fpn_out_channels,
    out_ch = 256,
)
