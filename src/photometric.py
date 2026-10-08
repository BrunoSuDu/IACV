"""Independent grayscale intensity changes; never resample spatial coordinates."""
import numpy as np

from transformations import Transformation, transformation_grid

PHOTOMETRIC_VALUES = {
    "brightness": [-60, -30, 30, 60],
    "contrast": [0.6, 0.8, 1.2, 1.4],
    "gamma": [0.6, 0.8, 1.25, 1.6],
}
PHOTOMETRIC_PROTOCOL = {
    "input": "existing uint8 grayscale image, no resampling",
    "brightness": "x + offset",
    "contrast": "127.5 + factor * (x - 127.5)",
    "gamma": "255 * (x / 255) ** gamma; gamma is the exponent, not its inverse",
    "quantization": "float64; clip to [0,255]; np.rint ties-to-even; cast uint8",
    "clipped_fraction": "fraction of pre-quantization values strictly outside [0,255]",
    "saturated_fraction": "fraction of final uint8 pixels equal to 0 or 255",
    "geometry": "H_gt=I; same dimensions, origin and coordinates; original support mask",
    "identity": "reference geometry identity_0; no extra photometric baseline rows",
}


def synthetic_grid():
    return transformation_grid() + [Transformation(kind, float(value))
                                    for kind, values in PHOTOMETRIC_VALUES.items() for value in values]


def apply_photometric(gray, setting):
    if gray.dtype != np.uint8 or gray.ndim != 2 or not gray.size:
        raise ValueError("Photometric input must be a nonempty uint8 grayscale array")
    x = gray.astype(np.float64)
    p = setting.parameter
    if not np.isfinite(p):
        raise ValueError("Photometric parameter must be finite")
    if setting.kind == "brightness":
        values = x + p
    elif setting.kind == "contrast" and p > 0:
        values = 127.5 + p * (x - 127.5)
    elif setting.kind == "gamma" and p > 0:
        values = 255.0 * np.power(x / 255.0, p)
    else:
        raise ValueError("Unknown photometric operation or nonpositive factor")
    result = np.rint(np.clip(values, 0, 255)).astype(np.uint8)
    if result.shape != gray.shape or result.dtype != gray.dtype:
        raise ValueError("Photometric operation changed dimensions/dtype")
    return result, {
        "photometric_clipped_fraction": float(((values < 0) | (values > 255)).mean()),
        "photometric_saturated_fraction": float(((result == 0) | (result == 255)).mean()),
        "photometric_source_saturated_fraction": float(((gray == 0) | (gray == 255)).mean()),
    }
