from pathlib import Path

# データセット
DATASET_PATH = "/work/dataset"
TRAIN_CSV_PATH = Path(DATASET_PATH, "train.csv")
TRAIN_IMG_ROOT = Path(DATASET_PATH, "train_imgs")
TEST_CSV_PATH = Path(DATASET_PATH, "test.csv")
TEST_IMG_ROOT = Path(DATASET_PATH, "test_imgs")


# 保存先
DST_ROOT_PATH = "/work/dst"
