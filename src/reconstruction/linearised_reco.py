# Regularized Gauss-Newton
# similar to https://arxiv.org/pdf/2103.15138.pdf

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import time
import os
from scipy.linalg import block_diag
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator
from scipy.spatial import Delaunay

"""
Here, we use the Jacobian calculated using Fenics.

"""

from ..ktc_methods import create_tv_matrix, SMPrior

class LinearisedRecoFenics():
    def __init__(self, Uref, B, vincl, mesh_name="dense",
                    base_path = "data",
                    version2=True, noise_level=(0.05, 0.01),
                    use_gpu=False, alphas=None):
        """
        Uref: reference measurements from empty water-tank
        B: observation operator 31x32 Mpat.T
        solver: instance of EITFEM solver
        Ltv: TV matrix
        mesh_name: mesh for sigma
        use_gpu: if True, use CuPy for matrix operations on GPU
        alphas: if provided, precompute projection matrices for fast reco
        """

        self.Uref = Uref
        self.mesh_name = mesh_name
        self.use_gpu = use_gpu
        self._base_path = base_path

        if use_gpu:
            import cupy as cp
            self._cp = cp

        self.noise_std1 = noise_level[0]
        self.noise_std2 = noise_level[1]
        self.sigma_background = 0.745

        self.vincl = vincl
        self.vincl_flatten = vincl.T.flatten()

        # Load mesh info (always needed for interpolation)
        self.coordinates = np.load(os.path.join(base_path, f"mesh_coordinates_{mesh_name}.npy"))
        self.cells = np.load(os.path.join(base_path, f"cells_{mesh_name}.npy"))
        pos = [[(self.coordinates[self.cells[i,0],0] + self.coordinates[self.cells[i,1],0] + self.coordinates[self.cells[i,2],0])/3.
                ,(self.coordinates[self.cells[i,0],1] + self.coordinates[self.cells[i,1],1] + self.coordinates[self.cells[i,2],1])/3.] for i in range(self.cells.shape[0])]
        self.pos = np.array(pos)

        # Precompute interpolation
        pixwidth = 0.23 / 256
        pixcenter_x = np.linspace(-0.115 + pixwidth / 2,
                                  0.115 - pixwidth / 2 + pixwidth, 256)
        X, Y = np.meshgrid(pixcenter_x, pixcenter_x)
        self._pixcenters = np.column_stack((X.ravel(), Y.ravel()))
        self._tri = Delaunay(self.pos)
        self._interp_W = self._build_interp_matrix()

        # --- Check R matrix cache BEFORE loading heavy matrices ---
        self._R_precomputed = None
        self._R_precomputed_gpu = None

        if alphas is not None:
            cache_path = self._get_cache_path(alphas)
            if os.path.exists(cache_path):
                # CACHE HIT: mmap R matrices, skip Rtv/Rsm/BJ entirely
                # This is the fast path — workers share physical memory pages
                R_stack = np.load(cache_path, mmap_mode='r')
                self._R_precomputed = [R_stack[i]
                                       for i in range(R_stack.shape[0])]
                if use_gpu:
                    self._R_precomputed_gpu = [
                        cp.asarray(np.array(R))
                        for R in self._R_precomputed]
                return  # Skip loading Rtv/Rsm/BJ — saves ~900 MB

        # --- Load heavy matrices (cache miss or no precomputation) ---
        self._load_heavy_matrices(base_path, mesh_name, B, version2)

        if alphas is not None:
            # CACHE MISS: compute R, save to disk, reload as mmap
            self._compute_save_and_load_R(alphas, cache_path)

            if use_gpu:
                self._R_precomputed_gpu = [
                    cp.asarray(np.array(R))
                    for R in self._R_precomputed]

            # Free heavy matrices — no longer needed with precomputed R
            for attr in ('Rtv', 'Rsm', 'BJ'):
                if hasattr(self, attr):
                    delattr(self, attr)
        elif use_gpu:
            # Non-precomputed GPU path: preload to GPU
            self._BJ_gpu = cp.asarray(self.BJ)
            self._Rtv_gpu = cp.asarray(self.Rtv)
            self._Rsm_gpu = cp.asarray(self.Rsm)

    def _load_heavy_matrices(self, base_path, mesh_name, B, version2):
        """Load Rtv, Rsm, BJ — only called on cache miss."""
        # TV matrix
        Rtv_raw = np.load(os.path.join(base_path,
                          f"mesh_neighbour_matrix_{self.mesh_name}.npy"))
        Rtv_dense = Rtv_raw.astype(np.float64)
        np.fill_diagonal(Rtv_dense, -Rtv_dense.sum(axis=0))
        self.Rtv = -Rtv_dense
        del Rtv_raw, Rtv_dense

        # Smoothness prior
        self.Rsm = np.load(os.path.join(base_path,
                           f"smoothnessR_{self.mesh_name}.npy"))

        # Jacobian
        J = np.load(os.path.join(base_path, f"jac_{mesh_name}.npy"))
        J = J.reshape(76*32, J.shape[-1])

        self.B = block_diag(*[np.array(B) for i in range(76)])

        if version2:
            self.BJ = self.B[self.vincl_flatten,:] @ J
        else:
            self.BJ = (self.B @ J)[self.vincl_flatten,:]

        del J  # J no longer needed

    def _get_cache_path(self, alphas):
        """Deterministic cache path from alphas + noise params + mesh."""
        import hashlib
        alpha_bytes = np.array(alphas, dtype=np.float64).tobytes()
        key_data = (alpha_bytes
                    + f"{self.noise_std1}_{self.noise_std2}_{self.mesh_name}"
                      .encode())
        cache_hash = hashlib.md5(key_data).hexdigest()[:12]
        return os.path.join(self._base_path,
                            f"R_cache_{self.mesh_name}_{cache_hash}.npy")

    def _compute_save_and_load_R(self, alphas, cache_path):
        """Compute R matrices, save to disk, reload as mmap."""
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        # File lock to prevent concurrent writes
        lock_path = cache_path + '.lock'
        got_lock = False
        try:
            lock_fd = os.open(lock_path,
                              os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(lock_fd)
            got_lock = True
        except OSError:
            pass

        if got_lock:
            try:
                self._compute_R_to_file(alphas, cache_path)
            finally:
                try:
                    os.remove(lock_path)
                except OSError:
                    pass
        else:
            # Another process is computing — wait for complete cache file
            import time as _time
            for _ in range(600):
                _time.sleep(0.5)
                if os.path.exists(cache_path):
                    # Ensure file is fully written (size stable across 2 checks)
                    try:
                        s1 = os.path.getsize(cache_path)
                        _time.sleep(0.2)
                        s2 = os.path.getsize(cache_path)
                        if s1 == s2 and s1 > 0:
                            break
                    except OSError:
                        pass
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"R cache was not created successfully: {cache_path}"
            )

        R_stack = np.load(cache_path, mmap_mode='r')
        self._R_precomputed = [R_stack[i] for i in range(R_stack.shape[0])]

    def _compute_R_to_file(self, alphas, cache_path):
        """Compute projection matrices and save as (5, N, M) array."""
        N = self.BJ.shape[1]
        diag_idx = np.arange(N)

        uref_vincl = np.array(self.Uref)[self.vincl_flatten].flatten()
        var_meas = ((self.noise_std1 / 100) * np.abs(uref_vincl))**2
        var_meas += ((self.noise_std2 / 100)
                     * np.max(np.abs(uref_vincl)))**2
        gamma_fixed = 1.0 / var_meas

        BJ_w = self.BJ * gamma_fixed[:, None]
        JGJ = BJ_w.T @ self.BJ
        jgj_diag = JGJ[diag_idx, diag_idx].copy()
        b_weight = BJ_w.T

        R_list = []
        _CHUNK = 256
        for alpha in alphas:
            A = JGJ.copy()
            if alpha[0] != 0:
                for j in range(0, N, _CHUNK):
                    e = min(j + _CHUNK, N)
                    A[j:e] += alpha[0] * self.Rtv[j:e]
            if alpha[1] != 0:
                for j in range(0, N, _CHUNK):
                    e = min(j + _CHUNK, N)
                    A[j:e] += alpha[1] * self.Rsm[j:e]
            if alpha[2] != 0:
                A[diag_idx, diag_idx] += alpha[2] * jgj_diag
            R = np.linalg.solve(A, b_weight)
            R_list.append(R)
            del A

        R_stack = np.stack(R_list)
        del R_list
        # Write to temp file then atomic rename to avoid partial-read races
        tmp_path = cache_path + '.tmp'
        with open(tmp_path, 'wb') as f:
            np.save(f, R_stack)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, cache_path)

    def reconstruct(self, Uel, alpha_tv, alpha_sm, alpha_lm):
        Uel = np.array(Uel)
        deltaU = Uel - np.array(self.Uref)
        deltaU = deltaU[self.vincl_flatten]

        var_meas = np.power(((self.noise_std1 / 100) * (np.abs(deltaU))),2)
        var_meas = var_meas + np.power((self.noise_std2 / 100) * np.max(np.abs(deltaU)),2)
        GammaInv = 1./var_meas
        GammaInv = np.diag(GammaInv[:,0])

        JGJ = self.BJ.T @ GammaInv @ self.BJ
        b = self.BJ.T  @ GammaInv @ deltaU

        A = JGJ + alpha_tv*self.Rtv + alpha_sm*self.Rsm + alpha_lm * np.diag(np.diag(JGJ))
        delta_sigma = np.linalg.solve(A,b)
        return delta_sigma

    def reconstruct_list(self, Uel, alpha_list=None):
        """Reconstruct with multiple alpha sets. Uses precomputed R if available."""
        Uel = np.array(Uel)
        deltaU = Uel - np.array(self.Uref)
        deltaU = deltaU[self.vincl_flatten]

        # Fast path: precomputed projection matrices
        if self._R_precomputed is not None:
            if self.use_gpu and self._R_precomputed_gpu is not None:
                cp = self._cp
                deltaU_gpu = cp.asarray(deltaU)
                return [cp.asnumpy(R @ deltaU_gpu)
                        for R in self._R_precomputed_gpu]
            return [R @ deltaU for R in self._R_precomputed]

        if self.use_gpu:
            return self._reconstruct_list_gpu(deltaU, alpha_list)

        # CPU fallback: per-sample noise model
        var_meas = np.power(((self.noise_std1 / 100) * (np.abs(deltaU))),2)
        var_meas = var_meas + np.power((self.noise_std2 / 100) * np.max(np.abs(deltaU)),2)
        gamma_vec = (1.0 / var_meas).flatten()

        BJ_w = self.BJ * gamma_vec[:, None]
        JGJ = BJ_w.T @ self.BJ
        b = BJ_w.T @ deltaU

        delta_sigma_list = []
        N = JGJ.shape[0]
        diag_idx = np.arange(N)
        jgj_diag = JGJ[diag_idx, diag_idx].copy()
        for alphas in alpha_list:
            A = JGJ.copy()
            if alphas[0] != 0:
                A += alphas[0] * self.Rtv
            if alphas[1] != 0:
                A += alphas[1] * self.Rsm
            if alphas[2] != 0:
                A[diag_idx, diag_idx] += alphas[2] * jgj_diag
            delta_sigma = np.linalg.solve(A, b)
            delta_sigma_list.append(delta_sigma)

        return delta_sigma_list

    def _reconstruct_list_gpu(self, deltaU, alpha_list):
        """GPU-accelerated reconstruct_list using CuPy."""
        cp = self._cp
        var_meas = np.power(((self.noise_std1 / 100) * (np.abs(deltaU))),2)
        var_meas = var_meas + np.power((self.noise_std2 / 100) * np.max(np.abs(deltaU)),2)
        gamma_vec = (1.0 / var_meas).flatten()

        gamma_gpu = cp.asarray(gamma_vec)
        deltaU_gpu = cp.asarray(deltaU)
        BJ = self._BJ_gpu
        BJ_w = BJ * gamma_gpu[:, None]
        JGJ = BJ_w.T @ BJ
        b = BJ_w.T @ deltaU_gpu

        jgj_diag = cp.diag(JGJ)
        Rtv = self._Rtv_gpu
        Rsm = self._Rsm_gpu

        delta_sigma_list = []
        for alphas in alpha_list:
            A = JGJ + alphas[0]*Rtv + alphas[1]*Rsm + alphas[2] * cp.diag(jgj_diag)
            delta_sigma = cp.linalg.solve(A, b)
            delta_sigma_list.append(cp.asnumpy(delta_sigma))
        return delta_sigma_list

    def reconstruct_from_deltaU(self, deltaU, alphas):
        deltaU = deltaU[self.vincl_flatten]
        var_meas = np.power(((self.noise_std1 / 100) * (np.abs(deltaU))),2)
        var_meas = var_meas + np.power((self.noise_std2 / 100) * np.max(np.abs(deltaU)),2)
        GammaInv = 1./var_meas
        GammaInv = np.diag(GammaInv[:,0])
        JGJ = self.BJ.T @ GammaInv @ self.BJ
        b = self.BJ.T  @ GammaInv @ deltaU
        A = JGJ + alphas[0]*self.Rtv + alphas[1]*self.Rsm + alphas[2] * np.diag(np.diag(JGJ))
        delta_sigma = np.linalg.solve(A,b)
        return delta_sigma

    def _build_interp_matrix(self):
        """Build sparse interpolation matrix from mesh elements to pixel grid."""
        from scipy import sparse as sp
        simplex_idx = self._tri.find_simplex(self._pixcenters)
        valid = simplex_idx >= 0
        n_pixels = len(self._pixcenters)
        n_elements = len(self.pos)
        si = simplex_idx[valid]
        pts = self._pixcenters[valid]
        Tinv = self._tri.transform[si, :2, :]
        r = self._tri.transform[si, 2, :]
        bary2 = np.einsum('ijk,ik->ij', Tinv, pts - r)
        bary3 = np.column_stack([bary2, 1.0 - bary2.sum(axis=1)])
        vertices = self._tri.simplices[si]
        valid_idx = np.where(valid)[0]
        rows = np.repeat(valid_idx, 3)
        cols = vertices.ravel()
        data = bary3.ravel()
        W = sp.csr_matrix((data, (rows, cols)), shape=(n_pixels, n_elements))
        return W

    def interpolate_to_image(self, sigma):
        sigma_flat = np.asarray(sigma).flatten()
        grid = self._interp_W @ sigma_flat
        return np.flipud(grid.reshape(256, 256))
