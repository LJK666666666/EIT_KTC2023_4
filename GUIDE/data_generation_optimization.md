# Data Generation Performance Optimization Summary

## Overview

EIT (Electrical Impedance Tomography) simulation data generation involves two compute-intensive steps per sample:

1. **SolveForward**: FEM matrix assembly + sparse linear system solve
2. **reconstruct_list**: Linearised reconstruction with regularisation (5 alpha sets)

Optimizations target both steps, using vectorization, COO sparse assembly, CuPy GPU acceleration, and algorithmic improvements.

---

## Completed Optimizations

### 1. A0 Stiffness Matrix Assembly: Vectorized + COO

**File**: `src/ktc_methods/KTCFwd.py` — `SolveForward`

**Before**: Python for-loop iterating over all HN elements, calling `grinprod_gauss_quad_node` per element. Each call computes a 6x6 local stiffness matrix via 3-point Gauss quadrature with `np.linalg.inv` and `np.linalg.det` on 2x2 Jacobians.

**After**: Batch all HN elements into arrays, compute all Jacobians/inverses/determinants simultaneously:

| Operation | Before | After |
|-----------|--------|-------|
| Element data gathering | `g[ind, :]` per element | `g[H]` all at once → (HN, 6, 2) |
| Jacobian | `L @ g` per element | `np.einsum('ik,ekj->eij', L, all_gg)` batched |
| 2x2 Inverse | `np.linalg.inv(Jt)` per element | Analytical formula, vectorized over HN |
| Gradient outer product | `G.T @ G` per element | `np.einsum('eik,eij->ekj', G, G)` batched |
| Matrix construction | Pre-allocated arrays + CSR | COO `(data, (row, col))` → `tocsr()` |

**Expected speedup**: A0 assembly from ~900ms to ~50-100ms (~10-15x)

---

### 2. S0 Electrode Boundary Assembly: COO Replacing CSR

**File**: `src/ktc_methods/KTCFwd.py` — `SolveForward`

**Before**: Directly modifying CSR sparse matrices `M[i,j] += value` and `K[i,j] += value`. CSR stores data in compressed row format — every insertion forces a full matrix memory copy.

```
SparseEfficiencyWarning: Changing the sparsity structure of a csr_matrix is expensive.
```

**After**: Collect `(row, col, val)` triplets into Python lists during the loop, build COO matrix at the end. COO automatically sums duplicate indices on `tocsr()` conversion.

Additional improvement: `sp.sparse.diags(s.flatten())` replaces `sp.sparse.csr_matrix(np.diag(s.flatten()))`, avoiding a dense (Nel x Nel) intermediate matrix.

**Expected speedup**: Eliminates SparseEfficiencyWarning, S0 assembly ~5x faster.

---

### 3. reconstruct_list: Diagonal Matrix Optimization

**File**: `src/reconstruction/linearised_reco.py` — `reconstruct_list`

**Before**: Constructing full M x M dense diagonal matrix `GammaInv = np.diag(gamma_vec)`, then computing `BJ.T @ GammaInv @ BJ` — O(M^2) memory, O(M^2 * N) computation.

**After**: Vectorized row-scaling `BJ_w = BJ * gamma_vec[:, None]`, then `JGJ = BJ_w.T @ BJ` — O(M*N) memory, O(M * N^2) computation.

```python
# Before (M=2356, wasteful):
GammaInv = np.diag(gamma_vec)           # (2356, 2356) dense diagonal
JGJ = BJ.T @ GammaInv @ BJ             # O(M^2 * N)

# After (vectorized):
BJ_w = BJ * gamma_vec[:, None]          # (M, N) row scaling
JGJ = BJ_w.T @ BJ                      # O(M * N^2)
```

**Speedup**: Benefits both CPU and GPU paths. ~2-3x on CPU.

---

### 4. reconstruct_list: CuPy GPU Acceleration

**File**: `src/reconstruction/linearised_reco.py` — `_reconstruct_list_gpu`

BJ, Rtv, Rsm matrices are preloaded to GPU in `__init__`. Per-sample computation (diagonal scaling + JGJ + 5x `cp.linalg.solve`) runs entirely on GPU.

**Benchmark result**: reco step 22789ms (CPU) → 3353ms (GPU) = **6.8x speedup**

---

### 5. interpolate_to_image: Cached Delaunay Triangulation

**File**: `src/reconstruction/linearised_reco.py` — `interpolate_to_image`

**Before**: `LinearNDInterpolator(self.pos, sigma)` rebuilds Delaunay triangulation on every call. Mesh centroids `self.pos` never change.

**After**: Precompute `self._tri = Delaunay(self.pos)` and pixel grid `self._pixcenters` in `__init__`. Reuse cached triangulation via `LinearNDInterpolator(self._tri, sigma)`.

**Speedup**: ~1.5-2x per call (saves ~20ms per call, called 5x per sample).

---

### 6. SolveForward: Reverted GPU Dense Solve

**File**: `src/ktc_methods/KTCFwd.py`

Initially attempted `cp.linalg.solve(A_gpu, b_gpu)` with dense matrix on GPU. This was **6.7x slower** than CPU because:

- `A.toarray()` destroys sparsity → O(N^3) dense LU instead of O(N^1.5) sparse solve
- 288MB CPU→GPU data transfer per sample
- FEM assembly (Python loop) is the real bottleneck, not the solve step

**Decision**: Reverted to CPU `scipy.sparse.linalg.spsolve`. With the vectorized assembly optimization above, the total forward time should drop from ~1000ms to ~150-200ms (assembly ~100ms + solve ~50-100ms).

---

## Files Modified

| File | Changes |
|------|---------|
| `src/ktc_methods/KTCFwd.py` | Vectorized A0 assembly, COO S0 assembly, `use_gpu` parameter |
| `src/reconstruction/linearised_reco.py` | Diagonal optimization, CuPy GPU path, cached Delaunay |
| `scripts/generate_data.py` | `--gpu` flag, per-step timing, timing JSON output |
| `scripts/benchmark_data_gen.py` | New: CPU vs GPU benchmark script |

---

## Potential Future Optimizations

### High Impact

**A. Numba JIT for S0 Electrode Assembly**

The S0 electrode boundary loop still uses Python for-loops with `np.where` lookups. Applying `@numba.njit` could speed this up 10-50x. However, the loop only iterates over electrode-adjacent elements (a small subset of HN), so the absolute time saving may be modest (~10-30ms).

```python
from numba import njit

@njit
def _assemble_electrode_boundary(elements, nodes, z, ...):
    # Compiled to machine code, eliminates Python overhead
    ...
```

**B. Batch Forward Solve with Prefactored Matrix**

When generating many samples, the matrix structure `A = A0(sigma) + S0(z)` changes per sample, but S0 only depends on `z` which is constant across samples. Precompute and cache S0 once:

```python
# In __init__ or first call:
self._S0_cached = self._assemble_S0(z)

# In SolveForward:
self.A = A0 + self._S0_cached  # Skip S0 assembly entirely
```

**Expected**: Save ~30-50ms per sample.

**C. CPU Multiprocess Pipeline**

Data generation is embarrassingly parallel — each sample is independent. Use `concurrent.futures.ProcessPoolExecutor` to overlap:
- Process 1-N: Generate phantoms + assemble FEM
- Main process: Solve + save

```
Worker 1: [phantom+assembly] [phantom+assembly] ...
Worker 2: [phantom+assembly] [phantom+assembly] ...
Main:     [solve+save]       [solve+save]       ...
```

**Expected**: 2-3x throughput improvement with 4+ CPU cores.

**D. HDF5 Batch Storage**

Replace per-sample `.npy` files with a single HDF5 file. Benefits:
- Eliminates filesystem overhead from millions of small files
- Enables chunked sequential writes (faster I/O)
- PyTorch DataLoader can use HDF5 slicing (faster training data loading)

```python
import h5py
with h5py.File('dataset/level_1.h5', 'w') as f:
    f.create_dataset('gt', shape=(N, 256, 256), dtype='float32')
    f.create_dataset('measurements', shape=(N, 2356), dtype='float64')
    # Append samples during generation
```

### Medium Impact

**E. Sparse Cholesky Factorization**

EIT stiffness matrix A is symmetric positive definite (SPD). Using Cholesky instead of general LU halves the computation:

```python
# Requires scikit-sparse (Linux/Mac only)
from sksparse.cholmod import cholesky
factor = cholesky(self.A)
UU = factor(self.b)
```

**Expected**: Solve step from ~50-100ms to ~25-50ms.

**F. CuPy Sparse Solve for Forward**

If CuPy adds matrix-RHS support for `cupyx.scipy.sparse.linalg.spsolve`, the forward solve could move to GPU. Current limitation: CuPy spsolve only accepts 1D RHS (our `b` has 76 columns for 76 current patterns).

Workaround: loop over 76 columns individually on GPU. May not be faster than CPU sparse solve due to kernel launch overhead.

**G. Precompute Quadrature Constants**

The 3 Gauss quadrature points produce fixed `S_q` and `L_q` arrays. Precompute them once in `__init__` instead of recalculating per `SolveForward` call:

```python
# In __init__:
self._quad_S = [...]  # 3 shape function arrays
self._quad_L = [...]  # 3 gradient arrays
```

Marginal time saving (~1ms) but cleaner code.

### Low Impact (For Very Large Scale)

**H. Full NumPy Vectorization of S0 Assembly**

Vectorize the electrode boundary loop entirely — precompute which elements touch each electrode, batch `bound_quad1` and `bound_quad2` calls. Complex to implement due to variable number of electrode elements per electrode.

**I. Mixed Precision (float32)**

Use float32 instead of float64 for FEM assembly and solve. Halves memory bandwidth and doubles SIMD throughput. Requires careful validation that reconstruction quality is maintained.

**J. CUDA Kernel for FEM Assembly**

Write a custom CUDA kernel (via CuPy `RawKernel` or Numba CUDA) for the full FEM assembly. This is the "nuclear option" — eliminates all Python overhead and runs the entire assembly on GPU. High implementation cost, but could reduce assembly to ~1-5ms.
