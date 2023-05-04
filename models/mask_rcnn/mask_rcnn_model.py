import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

RESNET_MEAN = (0.485, 0.456, 0.406)
RESNET_STD = (0.229, 0.224, 0.225)


def get_model(normalize=False, box_detections_per_img=500):
    # クラス数：半導体（検出したいもの）+背景
    NUM_CLASSES = 2

    if normalize:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            pretrained=True,
            box_detections_per_img=box_detections_per_img,
            image_mean=RESNET_MEAN,
            image_std=RESNET_STD,
        )
    else:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            pretrained=True, box_detections_per_img=box_detections_per_img
        )

    # 分類用のinput feature数
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # headの付け替え
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

    # 物体検出用のinput feature数
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # headの付け替え
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, NUM_CLASSES
    )
    return model
