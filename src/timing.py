"""计时不包含对象创建、文件读写和画图。"""
from time import perf_counter

import numpy as np


def measure(operation, warmup=2, repetitions=5, synchronize=None):
    if warmup < 0 or repetitions < 1:
        raise ValueError("warmup must be >= 0; repetitions must be >= 1")
    for _ in range(warmup):
        operation()
    elapsed_ms = []
    for _ in range(repetitions):
        if synchronize is not None:
            synchronize()
        start = perf_counter()
        result = operation()
        if synchronize is not None:
            synchronize()
        elapsed_ms.append((perf_counter() - start) * 1000)
    return result, float(np.mean(elapsed_ms)), float(np.std(elapsed_ms))
