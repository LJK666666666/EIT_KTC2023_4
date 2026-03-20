● 在 __init__ 中对 self.b 做 SVD 降秩：

● Update(src\ktc_methods\KTCFwd.py)
  ⎿  Added 9 lines
       97          self.b = np.concatenate(
       98              (np.zeros((self.ng2, Inj.shape[1])), self.C.T * Inj), axis=0)
       99
      100 +        # Reduced-RHS decomposition: rank(Injref)=15 for current ref.mat,                                                                                        
      101 +        # so self.b has rank 15. Solve 15 basis RHS instead of 76, then                                                                                          
      102 +        # reconstruct via coefficient matrix. Exact (machine-precision).                                                                                         
      103 +        U, s, Vt = np.linalg.svd(self.b, full_matrices=False)                                                                                                    
      104 +        rank = int(np.sum(s > s[0] * 1e-12))                                                                                                                     
      105 +        self._b_basis = U[:, :rank] * s[:rank]   # (N, rank)                                                                                                     
      106 +        self._b_coeff = Vt[:rank, :]              # (rank, 76)                                                                                                   
      107 +        self._b_rank = rank                                                                                                                                      
      108 +                                                                                                                                                                 
      109          # QC and MpatC (constant: depends only on Mpat, C)
      110          if Mpat is not None:
      111              self.QC = np.block([

● 在 SolveForward 中用降秩求解替换完整求解：

● Update(src\ktc_methods\KTCFwd.py)
  ⎿  Added 4 lines, removed 3 lines
      196
      197          self.A = A0 + self._S0_cached
      198
      199 -        # === Solve ===                                                                                                                                          
      199 +        # === Solve (reduced-RHS: 15 basis columns instead of 76) ===                                                                                            
      200          if _HAS_PARDISO:
      201 -            UU = pardiso_spsolve(self.A, self.b)                                                                                                                 
      201 +            UU_basis = pardiso_spsolve(self.A, self._b_basis)                                                                                                    
      202          else:
      203 -            UU = sp.sparse.linalg.spsolve(self.A, self.b)                                                                                                        
      203 +            UU_basis = sp.sparse.linalg.spsolve(self.A, self._b_basis)                                                                                           
      204 +        UU = UU_basis @ self._b_coeff                                                                                                                            
      205          self.theta = UU
      206          self.Pot = UU[0:self.ng2, :]
      207          self.Imeas = self.Inj
