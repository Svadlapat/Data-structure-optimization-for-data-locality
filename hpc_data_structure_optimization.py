"""
hpc_data_structure_optimization.py

Demonstration of data structure optimization for data locality
in the style of the Azad et al. (2023) HPC performance bugs study.

Baseline: Python list-of-lists with explicit loops.
Optimized: NumPy 2D array with vectorized operations (contiguous layout).
"""

import time
import statistics
from typing import List, Tuple

import numpy as np


def create_grid_list(n: int) -> List[List[float]]:
    """Create an n x n grid represented as a Python list of lists."""
    return [[float(i * n + j) for j in range(n)] for i in range(n)]


def create_grid_numpy(n: int) -> np.ndarray:
    """Create an n x n grid represented as a NumPy array (contiguous)."""
    data = np.arange(n * n, dtype=np.float64)
    return data.reshape((n, n))


def stencil_baseline_list(grid: List[List[float]]) -> List[List[float]]:
    """
    Baseline stencil: list-of-lists and explicit loops.
    For each interior cell, compute the average of the cell and its 4 neighbors.
    Edges are left unchanged for simplicity.
    """
    n = len(grid)
    # Deep copy to avoid modifying the original grid
    result = [[grid[i][j] for j in range(n)] for i in range(n)]

    for i in range(1, n - 1):
        row = grid[i]
        row_above = grid[i - 1]
        row_below = grid[i + 1]
        for j in range(1, n - 1):
            center = row[j]
            up = row_above[j]
            down = row_below[j]
            left = row[j - 1]
            right = row[j + 1]
            result[i][j] = (center + up + down + left + right) / 5.0

    return result


def stencil_optimized_numpy(grid: np.ndarray) -> np.ndarray:
    """
    Optimized stencil: NumPy 2D array and vectorized operations.
    Uses contiguous layout and avoids Python-level loops.
    """
    # Copy to keep behavior consistent with baseline
    result = grid.copy()

    # Interior slice indices
    center = grid[1:-1, 1:-1]
    up = grid[:-2, 1:-1]
    down = grid[2:, 1:-1]
    left = grid[1:-1, :-2]
    right = grid[1:-1, 2:]

    result[1:-1, 1:-1] = (center + up + down + left + right) / 5.0
    return result


def benchmark(fn, *args, repeats: int = 5) -> Tuple[float, float]:
    """
    Run a simple benchmark on function `fn`, returning (mean_time, stdev_time)
    over `repeats` runs.
    """
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn(*args)
        end = time.perf_counter()
        times.append(end - start)
    return statistics.mean(times), statistics.stdev(times)


def main():
    # Grid size: adjust as needed; larger sizes better show the difference.
    n = 1500  # e.g., 1500 x 1500 ~ 2.25M cells

    print(f"Creating {n}x{n} grids...")
    grid_list = create_grid_list(n)
    grid_np = create_grid_numpy(n)

    print("Warm-up run (to reduce first-time overhead)...")
    stencil_baseline_list(grid_list)
    stencil_optimized_numpy(grid_np)

    print("\nBenchmarking baseline (list-of-lists)...")
    base_mean, base_std = benchmark(stencil_baseline_list, grid_list, repeats=5)
    print(f"Baseline mean time: {base_mean:.4f}s (std={base_std:.4f}s)")

    print("\nBenchmarking optimized (NumPy array)...")
    opt_mean, opt_std = benchmark(stencil_optimized_numpy, grid_np, repeats=5)
    print(f"Optimized mean time: {opt_mean:.4f}s (std={opt_std:.4f}s)")

    speedup = base_mean / opt_mean if opt_mean > 0 else float("inf")
    print(f"\nEstimated speedup: {speedup:.2f}x")


if __name__ == "__main__":
    main()
