from pathlib import Path

from datasets import ImagePair


def iter_hpatches_pairs(root, sequences=None):
    root = Path(root)
    directories = sorted(path for path in root.iterdir()
                         if path.is_dir() and path.name.startswith(("i_", "v_")))
    if sequences:
        missing = set(sequences) - {path.name for path in directories}
        if missing:
            raise ValueError(f"Unknown HPatches sequences: {sorted(missing)}")
        directories = [path for path in directories if path.name in sequences]
    if not directories:
        raise ValueError(f"No HPatches sequences found in {root}")
    for sequence in directories:
        category = "illumination" if sequence.name.startswith("i_") else "viewpoint"
        for target in range(2, 7):
            pair = ImagePair("hpatches", sequence.name, category, target,
                             sequence / "1.ppm", sequence / f"{target}.ppm",
                             sequence / f"H_1_{target}")
            for path in (pair.image1, pair.image2, pair.homography):
                if not path.is_file():
                    raise FileNotFoundError(path)
            yield pair
