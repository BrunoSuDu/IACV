"""Independent transformations; float64 GT maps source pixels to a fixed-size canvas."""
from dataclasses import dataclass

import cv2
import numpy as np

from homography import transform_points


@dataclass(frozen=True)
class Transformation:
    kind: str
    parameter: float

    @property
    def name(self):
        return f"{self.kind}_{self.parameter:g}"


CURVE_VALUES = {
    "rotation": [-90, -60, -30, 0, 30, 60, 90],
    "scale": [0.5, 0.75, 1, 1.25, 1.5, 2],
    "translation_x": [-0.25, -0.1, 0, 0.1, 0.25],
    "translation_y": [-0.25, -0.1, 0.1, 0.25],
    "shear": [-0.25, -0.125, 0, 0.125, 0.25],
    "perspective": [-0.12, -0.06, 0, 0.06, 0.12],
}


def is_identity(kind, parameter):
    return parameter == (1 if kind == "scale" else 0)


def transformation_grid():
    settings = [Transformation("identity", 0.0)]
    for kind, values in CURVE_VALUES.items():
        settings.extend(Transformation(kind, float(value)) for value in values
                        if not is_identity(kind, value))
    if len(settings) != 28 or len({t.name for t in settings}) != 28:
        raise ValueError("Synthetic grid must contain exactly 28 unique settings")
    return settings


def corners(shape):
    h, w = shape[:2]
    if h < 2 or w < 2:
        raise ValueError("Images must be at least 2x2")
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float64)


def perspective_destination(shape, strength):
    """Top corners shift inward by s*(w-1); bottom corners shift outward.

    y is unchanged. Positive strength makes a trapezoid wider at its bottom.
    This is a dimensionless projective operation, with no added rotation/translation.
    """
    if not np.isfinite(strength) or abs(strength) >= 0.5:
        raise ValueError("Perspective strength must be finite with abs(s) < 0.5")
    target = corners(shape)
    d = strength * (shape[1] - 1)
    target[:, 0] += [d, -d, d, -d]
    return target


def homography_for(setting, shape):
    h, w = shape[:2]
    source_corners = corners(shape)
    cx, cy = (w - 1) / 2, (h - 1) / 2
    kind, p = setting.kind, setting.parameter
    if not np.isfinite(p):
        raise ValueError("Transformation parameter must be finite")
    H = np.eye(3, dtype=np.float64)
    if kind == "identity":
        if p != 0:
            raise ValueError("Identity parameter must be 0")
    elif kind == "rotation":
        # OpenCV positive angles rotate counterclockwise in screen coordinates.
        H[:2] = cv2.getRotationMatrix2D((cx, cy), p, 1.0)
    elif kind == "scale":
        if p <= 0:
            raise ValueError("Scale must be positive")
        H = np.array([[p, 0, cx * (1 - p)], [0, p, cy * (1 - p)], [0, 0, 1]], np.float64)
    elif kind == "translation_x":
        H[0, 2] = p * w
    elif kind == "translation_y":
        H[1, 2] = p * h
    elif kind == "shear":
        H[0, 1], H[0, 2] = p, -p * cy
    elif kind == "perspective":
        target = perspective_destination(shape, p)
        # Solve eight equations in float64 instead of casting corners to float32.
        A, b = [], []
        for (x, y), (u, v) in zip(source_corners, target):
            A.extend([[x, y, 1, 0, 0, 0, -u*x, -u*y],
                      [0, 0, 0, x, y, 1, -v*x, -v*y]])
            b.extend([u, v])
        H = np.append(np.linalg.solve(np.asarray(A, np.float64), b), 1).reshape(3, 3)
    else:
        raise ValueError(f"Unknown transformation: {kind}")
    polygon = transform_points(source_corners, H)
    denominators = np.column_stack((source_corners, np.ones(4))) @ H[2]
    if (not np.isfinite(H).all() or abs(np.linalg.det(H)) < 1e-12
            or not np.isfinite(polygon).all() or np.any(denominators <= 0)
            or not cv2.isContourConvex(polygon.astype(np.float32))
            or abs(cv2.contourArea(polygon.astype(np.float32))) < 1):
        raise ValueError("Invalid projective homography or transformed polygon")
    return H


def erode_valid_mask(mask, margin):
    if margin < 0:
        raise ValueError("Mask margin must be nonnegative")
    return cv2.erode(mask.astype(np.uint8), np.ones((2*margin + 1, 2*margin + 1), np.uint8),
                     borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)


def warp_with_mask(image, H):
    """Same H, same canvas, same bilinear interpolation for image and mask.

    Reflect padding reduces sharp black edges; constant-zero mask marks ALL padding
    invalid. Mask additionally removes one interpolation-boundary pixel.
    """
    h, w = image.shape[:2]
    transformed = cv2.warpPerspective(image, H, (w, h), flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_REFLECT_101)
    coverage = cv2.warpPerspective(np.ones((h, w), np.float32), H, (w, h),
                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=0)
    geometric_valid = coverage >= 1 - 1e-6
    valid = erode_valid_mask(geometric_valid, 1)
    source_overlap = cv2.warpPerspective(geometric_valid.astype(np.uint8), np.linalg.inv(H), (w, h),
                                        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=0).astype(bool)
    return transformed, valid, {
        "overlap_ratio": float(source_overlap.mean()),
        "target_valid_ratio": float(valid.mean()),
        "canvas": "fixed source width/height; origin (0,0); no canvas translation",
        "interpolation": "INTER_LINEAR", "border_mode": "BORDER_REFLECT_101",
        "mask_interpolation": "INTER_LINEAR; coverage>=1-1e-6; erosion=1px",
    }
