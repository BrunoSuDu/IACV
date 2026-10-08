"""数据集只负责提供文件路径，不包含特征算法。"""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ImagePair:
    dataset: str
    sequence: str
    category: str
    target_index: int
    image1: Path
    image2: Path
    homography: Path

    @property
    def name(self):
        return f"{self.sequence}_1_to_{self.target_index}"
