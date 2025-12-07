# Data-structure-optimization-for-data-locality

The optimization technique selected is:
Data Structure Optimization for Memory Locality
(Also referred to as “Data Locality Optimization” or “Improving Spatial Locality” in HPC.)

This project demonstrates how data layout affects performance in High-Performance Computing (HPC) applications. The experiment compares two implementations of a 2D stencil computation:

Baseline: Python list-of-lists with nested loops

Optimized: NumPy array with contiguous memory and vectorized operations

Both versions perform the same computation, but the optimized implementation uses a cache-friendly data structure.

Key Findings

Baseline runtime: 0.4747 seconds

Optimized runtime: 0.0117 seconds

Overall speedup: 40.43×

The performance improvement is achieved through better memory locality, not algorithmic change. This reflects patterns observed in empirical HPC performance bug studies, where developers replace scattered data structures with contiguous representations.

Files Included:
hpc_data_structure_optimization.py
README.md

Execution:
``` bash
pip install numpy
python hpc_data_structure_optimization.py
```

Explanation

The optimized version uses NumPy arrays, which store data contiguously in memory and allow efficient vectorized operations implemented in C. This eliminates interpreter overhead and improves cache utilization, resulting in drastically faster execution.