import collections
import os

import pandas as pd
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F


from common import rle


class SemiDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        df: pd.DataFrame,
        transforms=None,
        resize=False,
        width=500,
        height=300,
    ):
        self.transforms = transforms
        self.image_dir = image_dir
        self.df = df

        # リサイズ
        self.should_resize = resize is not False
        if self.should_resize:
            self.height = int(height * resize)
            self.width = int(width * resize)
        else:
            self.height = height
            self.width = width

        self.image_info = collections.defaultdict(dict)
        temp_df = self.df.groupby("image")["rle"].agg(lambda x: list(x)).reset_index()
        temp_df = temp_df.merge(
            self.df[["image", "height", "width"]].drop_duplicates(),
            on="image",
            how="inner",
        ).reset_index(drop=True)
        for index, row in temp_df.iterrows():
            self.image_info[index] = {
                "image_id": row["image"],
                "path": self.image_dir / row["image"],
                "annotations": row["rle"],
                "height": row["height"],
                "width": row["width"],
            }

    def get_box(self, a_mask):
        """与えられたマスクのバウンディングボックスを取得"""
        pos = np.where(a_mask)
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        return [xmin, ymin, xmax, ymax]

    def __getitem__(self, idx):
        """画像とターゲットを取得する"""

        # 画像の読み込み
        img_path = self.image_info[idx]["path"]
        img = Image.open(img_path).convert("RGB")

        # 読み込んだ画像をリサイズ
        if self.should_resize:
            img = img.resize((self.width, self.height), resample=Image.BILINEAR)

        # アノテーションや高さ・幅の情報を取得
        info = self.image_info[idx]

        # アノテーションの数
        n_objects = len(info["annotations"])
        # mask作る
        masks = np.zeros(
            (len(info["annotations"]), self.height, self.width), dtype=np.uint8
        )
        boxes = []

        for i, annotation in enumerate(info["annotations"]):
            # rleをmaskに変更
            a_mask = rle.rle_decode(annotation, (info["height"], info["width"]))
            a_mask = Image.fromarray(a_mask)

            # maskもリサイズ
            if self.should_resize:
                a_mask = a_mask.resize(
                    (self.width, self.height), resample=Image.BILINEAR
                )

            a_mask = np.array(a_mask) > 0
            masks[i, :, :] = a_mask

            boxes.append(self.get_box(a_mask))

        # ラベルは1種類
        labels = np.array([1 for _ in range(n_objects)])

        if self.transforms:
            aug = self.transforms(
                image=np.array(img),
                masks=list(masks),
                bboxes=np.array(boxes),
                category_id=labels,
            )
            img = aug["image"]
            masks = np.array(aug["masks"])
            boxes = np.array(aug["bboxes"])
            labels = np.array(aug["category_id"])

        # 今回使用するMask R-CNNで求められるデータをまとめる
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((n_objects,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd,
        }

        return F.to_tensor(img), target

    def __len__(self):
        return len(self.image_info)


class SemiTestDataset(Dataset):
    def __init__(
        self, image_dir, df, transforms=None, resize=False, height=300, width=500
    ):
        self.transforms = transforms
        self.image_dir = image_dir
        self.df = df
        self.image_ids = self.df["image"].values
        self.origin_height = self.df["height"].values
        self.origin_width = self.df["width"].values

        # リサイズ
        self.should_resize = resize is not False
        if self.should_resize:
            self.height = int(height * resize)
            self.width = int(width * resize)
        else:
            self.height = height
            self.width = width

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.image_dir, image_id)
        image = Image.open(image_path).convert("RGB")

        # 読み込んだ画像をリサイズ
        if self.should_resize:
            image = image.resize((self.width, self.height), resample=Image.BILINEAR)

        if self.transforms is not None:
            image, _ = self.transforms(image=image, target=None)

        # 大きさを戻すために元画像の大きさを取得
        origin_height = self.origin_height[idx]
        origin_width = self.origin_width[idx]

        return {
            "image": image,
            "image_id": image_id,
            "height": origin_height,
            "width": origin_width,
            "resize": self.should_resize,
        }

    def __len__(self):
        return len(self.image_ids)
