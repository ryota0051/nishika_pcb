import numpy as np


def rle_decode(mask_rle: str, shape, color=1):
    """rle(run length encoding)をdecodeする
    Args:
        mask_rle: エンコードされたマスク
        shape: (height, width)
        color: 背景以外のカラーインデックス
    Returns:
        マスク画像
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    # startsをインデックスに変換
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.float32)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = color
    return img.reshape(shape)


def rle_encoding(img: np.ndarray):
    dots = np.where(img.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if b > prev + 1:
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return " ".join(map(str, run_lengths))
