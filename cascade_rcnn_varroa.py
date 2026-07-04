_base_ = [
    "../cascade_rcnn/cascade-rcnn_r50_fpn_1x_coco.py",
]

data_root = "/marimo/datasets/varroa_coco/"

metainfo = dict(
    classes=("varroa",),
)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="Resize", scale=(640, 640), keep_ratio=True),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PackDetInputs"),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(640, 640), keep_ratio=True),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]

model = dict(
    roi_head=dict(
        bbox_head=[
            dict(
                type="Shared2FCBBoxHead",
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type="DeltaXYWHBBoxCoder",
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2],
                ),
                reg_class_agnostic=True,
                loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type="SmoothL1Loss", beta=1.0, loss_weight=1.0),
            ),
            dict(
                type="Shared2FCBBoxHead",
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type="DeltaXYWHBBoxCoder",
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.05, 0.05, 0.1, 0.1],
                ),
                reg_class_agnostic=True,
                loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type="SmoothL1Loss", beta=1.0, loss_weight=1.0),
            ),
            dict(
                type="Shared2FCBBoxHead",
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type="DeltaXYWHBBoxCoder",
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.033, 0.033, 0.067, 0.067],
                ),
                reg_class_agnostic=True,
                loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type="SmoothL1Loss", beta=1.0, loss_weight=1.0),
            ),
        ]
    )
)

train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file="annotations/instances_train.json",
        data_prefix=dict(img="images/train/"),
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file="annotations/instances_val.json",
        data_prefix=dict(img="images/val/"),
        pipeline=test_pipeline,
    ),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file="annotations/instances_test.json",
        data_prefix=dict(img="images/test/"),
        pipeline=test_pipeline,
    ),
)

val_evaluator = dict(
    ann_file=data_root + "annotations/instances_val.json",
)

test_evaluator = dict(
    ann_file=data_root + "annotations/instances_test.json",
)

train_cfg = dict(
    type="EpochBasedTrainLoop",
    max_epochs=100,
    val_interval=10,
)

optim_wrapper = dict(
    type="AmpOptimWrapper",
    optimizer=dict(
        lr=0.0025,
    )
)

default_hooks = dict(
    checkpoint=dict(
        interval=10,
        max_keep_ckpts=3,
        save_best="coco/bbox_mAP",
        rule="greater",
    )
)

load_from = "/marimo/mmdetection/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth"

work_dir = "./work_dirs/cascade_rcnn_varroa"
