"""Frozen source manifest; generated images stay in memory during manual experiments."""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from transformations import transformation_grid


DEFAULT_MANIFEST = Path(__file__).resolve().parents[2] / "configs/synthetic_sources.json"


def select_sources(root, manifest=DEFAULT_MANIFEST):
    root = Path(root).resolve()
    spec = json.loads(Path(manifest).read_text(encoding="utf-8"))
    paths = spec["sources"]
    if len(paths) != 8 or len(set(paths)) != 8:
        raise ValueError("Expected eight unique synthetic sources")
    if sum(p.startswith("i_") for p in paths) != 4 or sum(p.startswith("v_") for p in paths) != 4:
        raise ValueError("Expected four illumination and four viewpoint references")
    sources = []
    for relative in paths:
        if len(Path(relative).parts) != 2:
            raise ValueError(f"Synthetic source must be <sequence>/1.ppm: {relative}")
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or path.name != "1.ppm":
            raise ValueError(f"Invalid synthetic source: {relative}")
        if not path.is_file():
            raise FileNotFoundError(path)
        sources.append(path)
    return sources, spec


@dataclass(frozen=True)
class SyntheticPair:
    image1: Path
    setting: object
    homography: np.ndarray
    target_index: int
    dataset: str = "synthetic"

    @property
    def sequence(self):
        return self.image1.parent.name

    @property
    def category(self):
        return self.setting.kind

    @property
    def name(self):
        return f"{self.sequence}__{self.setting.name}"

    @property
    def image2(self):
        return f"in_memory:{self.name}"


def expected_synthetic_records(source_count=8, method_count=10, setting_count=None):
    return source_count * (len(transformation_grid()) if setting_count is None else setting_count) * method_count
