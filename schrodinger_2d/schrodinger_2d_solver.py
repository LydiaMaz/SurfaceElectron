"""
schrodinger_2d_solver.py
2D Schrödinger equation solver using DLL-FDM method (https://arxiv.org/pdf/1512.05826) in cylindrical coordinates.
Python translation of Ziheng Zhang's Schrodinger_2dsolver.m (WASHU)
"""
#-----------------------------------------------------------------------
import numpy as np
from scipy.sparse import diags, kron, identity, spdiags, csc_matrix
from scipy.sparse.linalg import eigsh
#-----------------------------------------------------------------------


def schrodinger_2d_solver(r, z, V, m, n_states=5, tol=1e-8, maxit=3000):

    # Physical constants
    hbar = 1.0545718e-34
    me = 9.10938356e-31
    e = 1.602e-19

    r = np.asarray(r, dtype=float)
    z = np.asarray(z, dtype=float)
    V = np.asarray(V, dtype=float)

    Nr = r.size
    Nz = z.size

    if V.shape != (Nr, Nz):
        raise ValueError(f"V must have shape (Nr, Nz) = ({Nr}, {Nz}), "
                         f"but got {V.shape}")

    # grid spacing
    dr = r[1] - r[0]
    dz = z[1] - z[0]

    #-----------------------------------------------------------------------
    # z part: central difference for 2nd deriv.
    #-----------------------------------------------------------------------
    factor = hbar**2 / (2.0 * me)
    Zdiag = factor * 2.0 / dz**2
    Zoff = -factor * 1.0 / dz**2

    main_z = np.full(Nz, Zdiag, dtype=float)
    off_z = np.full(Nz - 1, Zoff, dtype=float)
    Hz = diags(
        diagonals=[off_z, main_z, off_z],
        offsets=[-1, 0, 1],
        format="csc"
    )

    #-----------------------------------------------------------------------
    # r part: DLL-FDM radial op
    #-----------------------------------------------------------------------
    coeff = -hbar**2 / (2.0 * me * dr**2)

    rdiag = np.zeros(Nr, dtype=float)
    rupper = np.zeros(Nr - 1, dtype=float)

    for i in range(1, Nr + 1):
        idx = i - 1
        rdiag[idx] = -2.0 - m**2 / (i - 0.5)**2
        if i < Nr:
            rupper[idx] = i / np.sqrt((i - 0.5) * (i + 0.5))

    Hr = coeff * diags(
        diagonals=[rupper, rdiag, rupper],
        offsets=[-1, 0, 1],
        format="csc"
    )

    #-----------------------------------------------------------------------
    # Total Hamiltonian
    #-----------------------------------------------------------------------
    Vvec = V.reshape(Nr * Nz, order="F") 
    V_full = spdiags(Vvec, 0, Nr * Nz, Nr * Nz, format="csc")

    Ir = identity(Nr, format="csc")
    Iz = identity(Nz, format="csc")

    Hr_full = kron(Iz, Hr, format="csc")
    Hz_full = kron(Hz, Ir, format="csc")

    H = (Hr_full + Hz_full + V_full) / e        # Hamiltonian in eV

    #-----------------------------------------------------------------------
    # Shift-invert eigensolver
    #-----------------------------------------------------------------------
    sigma = float(Vvec.min() / e - 0.1)

    E_vals, vecs = eigsh(
        H,
        k=n_states,
        sigma=sigma,
        which="LM",
        tol=tol,
        maxiter=maxit
    )

    # Sort ascending
    idx = np.argsort(E_vals)
    E_vals = E_vals[idx]
    vecs = vecs[:, idx] 

    psi = vecs

    return psi, E_vals
