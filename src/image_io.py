from pathlib import Path

import cv2
import numpy as np


def load_image(path):
    """兼容 Windows 中文路径；返回 BGR 彩色图和灰度图。"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {path}")
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not decode image: {path}")
    return image, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
