from pathlib import Path

from datasets import ImagePair


def iter_graf_pairs(root):
    root = Path(root)
    for target in (2, 4):
        pair = ImagePair("graf", "graf", "viewpoint", target, root / "img1.ppm",
                         root / f"img{target}.ppm", root / f"H1to{target}p.txt")
        for path in (pair.image1, pair.image2, pair.homography):
            if not path.is_file():
                raise FileNotFoundError(path)
        yield pair
