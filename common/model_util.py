import os
import random

import numpy as np
import torch


def fix_all_seeds(seed):
    """乱数シード固定"""
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


def trim_zeros(x: np.ndarray):
    assert len(x.shape) == 2
    return x[~np.all(x == 0, axis=1)]


# === https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/utils.py === より引用


def compute_overlaps_masks(masks1: np.ndarray, masks2: np.ndarray):
    """masks1とmasks2のiou行列を作成する
    Args:
        masks1: (masks1のインスタンス数, Height, Width)
        masks2: (masks2のインスタンス数, Height, Width)
    Returns:
        masks1とmasks2のインスタンス間のIOU行列
        (masks1のインスタンス数, masks2のインスタンス数)
    """
    masks1_instances = masks1.shape[0]
    masks2_instances = masks2.shape[0]
    if masks1_instances == 0 or masks2_instances == 0:
        return np.zeros((masks1_instances, masks2_instances))
    # flat化: (インスタンス数, Height, Width) => (インスタンス数, Height * Width)
    masks1 = np.reshape(masks1, (masks1_instances, -1)).astype(np.float32)
    masks2 = np.reshape(masks2, (masks2_instances, -1)).astype(np.float32)
    area1 = np.sum(masks1, axis=1)
    area2 = np.sum(masks2, axis=1)
    intersections = np.dot(masks1, masks2.T)
    union = area1[:, None] + area2[None, :] - intersections
    return intersections / union


def compute_matches(
    gt_boxes,
    gt_class_ids,
    gt_masks,
    pred_boxes,
    pred_class_ids,
    pred_scores,
    pred_masks,
    iou_th=0.5,
    score_th=0.0,
):
    """正解マスクと対応する予測マスクのインデックスを返す
    Args:
        gt_boxes: 正解バウンディングボックス (インスタンス数, (x1, y1, x2, y2))
            0 <= x1 < x2 <= 画像幅, 0 <= y1 < y2 <= 画像高さ
        gt_class_ids: 各予測結果の所属クラス
        gt_masks: 正解マスク (インスタンス数, HEIGHT, WIDTH)
        pred_boxes: 予測結果バウンディングボックス (インスタンス数, (x1, y1, x2, y2))
        pred_class_ids: 予測結果の所属クラス (インスタンス数, )
        pred_scores: 予測スコア (インスタンス数, )
        pred_masks: 2値化済みの予測マスク (インスタンス数, HEIGHT, WIDTH)
        iou_th: マッチしていると判定するiouのしきい値
        score_th: iouを切り捨てる際のしきい値
    Returns:
        gt_mactch: 正解とマッチした予測結果インスタンスのインデックス
        pred_match: 予測結果とマッチした正解インスタンスのインデックス
        overlaps: iou行列
            (正解インスタンス数, 予測インスタンス数)
            overlaps[0, 10] => 正解インスタンス0と予測インスタンス10のiou
            -1の場合は、マッチング不成立を示す
    """
    gt_boxes = trim_zeros(gt_boxes)
    gt_masks = gt_masks[: gt_boxes.shape[0]]
    pred_boxes = trim_zeros(pred_boxes)
    pred_scores = pred_scores[: pred_boxes.shape[0]]
    # スコア降順にソート
    indices = np.argsort(pred_scores)[::-1]
    pred_boxes = pred_boxes[indices]
    pred_class_ids = pred_class_ids[indices]
    pred_scores = pred_scores[indices]
    pred_masks = pred_masks[indices]
    overlaps = compute_overlaps_masks(pred_masks, gt_masks)

    match_count = 0
    pred_match = -1 * np.ones([pred_boxes.shape[0]])
    gt_match = -1 * np.ones([gt_boxes.shape[0]])
    for i in range(len(pred_boxes)):
        # iou降順にソートする
        sorted_idx = np.argsort(overlaps[i])[::-1]
        low_score_idx = np.where(overlaps[i, sorted_idx] < score_th)[0]
        if low_score_idx.size > 0:
            sorted_idx = sorted_idx[: low_score_idx[0]]
        for j in sorted_idx:
            if gt_match[j] > -1:
                continue
            iou = overlaps[i, j]
            # iouは、予測インスタンスに対して降順のためiou < iout_thになった時点でその先は計算する必要がない
            if iou < iou_th:
                break
            if pred_class_ids[i] == gt_class_ids[j]:
                match_count += 1
                gt_match[j] = i
                pred_match[i] = j
                break
    return gt_match, pred_match, overlaps


def compute_ap(
    gt_boxes,
    gt_class_ids,
    gt_masks,
    pred_boxes,
    pred_class_ids,
    pred_scores,
    pred_masks,
    iou_th=0.5,
):
    """AP計算用関数
    Args:
        gt_boxes: 正解バウンディングボックス (インスタンス数, (x1, y1, x2, y2))
            0 <= x1 < x2 <= 画像幅, 0 <= y1 < y2 <= 画像高さ
        gt_class_ids: 各予測結果の所属クラス
        gt_masks: 正解マスク (インスタンス数, HEIGHT, WIDTH)
        pred_boxes: 予測結果バウンディングボックス (インスタンス数, (x1, y1, x2, y2))
        pred_class_ids: 予測結果の所属クラス (インスタンス数, )
        pred_scores: 予測スコア (インスタンス数, )
        pred_masks: 2値化済みの予測マスク (インスタンス数, HEIGHT, WIDTH)
        iou_th: マッチング時のiouしきい値
    Note:
        処理フローは下記の通り
        1. 正解インスタンスと予測インスタンス間でiouを用いたマッチングを実施
        2. マッチング結果からprecision, recallを計算
        3. https://pystyle.info/how-to-calculate-object-detection-metrics-map/#outline__8 に記載された方法でprecisionを補正
        4. 曲線下の面積を求める
    Returns:
        mAP: mean average precision
        precisions: map計算に使用したprecision
        recalls: map計算に使用したrecall
        overlaps: マッチング結果
    """
    gt_match, pred_match, overlaps = compute_matches(
        gt_boxes,
        gt_class_ids,
        gt_masks,
        pred_boxes,
        pred_class_ids,
        pred_scores,
        pred_masks,
        iou_th,
    )
    precisions = np.cumsum(pred_match > -1) / (np.arange(len(pred_match)) + 1)
    recalls = np.cumsum(pred_match > -1).astype(np.float32) / len(gt_match)

    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])

    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    mAP = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])
    return mAP, precisions, recalls, overlaps


def compute_ap_range(
    gt_boxes,
    gt_class_ids,
    gt_masks,
    pred_boxes,
    pred_class_ids,
    pred_scores,
    pred_masks,
    iou_thresholds=None,
    verbose=0,
):
    """指定iouしきい値レンジにおけるmapの平均を計算して返す
    Args:
        gt_boxes: 正解バウンディングボックス (インスタンス数, (x1, y1, x2, y2))
            0 <= x1 < x2 <= 画像幅, 0 <= y1 < y2 <= 画像高さ
        gt_class_ids: 各予測結果の所属クラス
        gt_masks: 正解マスク (インスタンス数, HEIGHT, WIDTH)
        pred_boxes: 予測結果バウンディングボックス (インスタンス数, (x1, y1, x2, y2))
        pred_class_ids: 予測結果の所属クラス (インスタンス数, )
        pred_scores: 予測スコア (インスタンス数, )
        pred_masks: 2値化済みの予測マスク (インスタンス数, HEIGHT, WIDTH)
        iou_thresholds: iouしきい値配列(default: 0.5~1.0を0.05刻み)
        verbose: 計算結果をprintするか(default: 0)
    Returns:
        AP: 各iouしきい値におけるapの平均値
    Notes:
        使用例
        ```
        DEVICE = "cuda"
        # モデルのロード(pytorchの mask-rcnnを使用している想定)
        model = get_model()
        model.to(DEVICE)

        # 学習済みモデルのロード
        model.load_state_dict(torch.load(<path to pretrained model>))
        model.eval()

        # 画像の推論
        with torch.no_grad():
            preds = model([img.to(DEVICE)])[0]
            preds = {k: v.cpu().numpy() for k, v in preds.items()}
        # 推論結果取得
        pred_boxes = preds['boxes']
        pred_labels = preds['labels']
        pred_scores = preds['scores']
        pred_masks = preds['masks'][:, 0, ...]

        use_masks_filter = pred_scores > 0
        pred_boxes = pred_boxes[use_masks_filter]
        pred_labels = pred_labels[use_masks_filter]
        pred_scores = pred_scores[use_masks_filter]
        pred_masks = pred_masks[use_masks_filter]
        # ピクセル内の確率が指定しきい値以下のピクセルを0に
        pred_masks = (pred_masks > 0.5).astype(int)

        ap = compute_ap_range(
            target['boxes'], target['labels'], target['masks'],
            pred_boxes, pred_labels, pred_scores, pred_masks,
        )
        ```
    """
    iou_thresholds = iou_thresholds or np.arange(0.5, 1.0, 0.05)
    AP = []
    for iou_thoreshold in iou_thresholds:
        ap, _, _, _ = compute_ap(
            gt_boxes,
            gt_class_ids,
            gt_masks,
            pred_boxes,
            pred_class_ids,
            pred_scores,
            pred_masks,
            iou_th=iou_thoreshold,
        )
        if verbose:
            print(f"AP @{iou_thoreshold:.3f}:\t {ap:.3f}")
        AP.append(ap)
    AP = np.mean(AP)
    if verbose:
        print(f"AP @{iou_thresholds[0]:.3f}-{iou_thresholds[-1]:.3f}:\t {AP:.3f}")
    return AP


class EarlyStopping:
    def __init__(self, dst, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score = np.Inf
        self.force_cancel = False
        self.dst = dst

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)
        elif score < self.best_score:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        if self.verbose:
            print(
                f"Validation score ({self.val_score:.6f} --> {score:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.dst)
        self.val_score = score
