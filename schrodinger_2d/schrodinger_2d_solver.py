import numpy as np
from scipy.sparse import diags, kron, identity, spdiags, csc_matrix
from scipy.sparse.linalg import eigsh


def schrodinger_2d_solver(r, z, V, m, n_states=5, tol=1e-8, maxit=3000):
    """
    Python translation of Ziheng's Schrodinger_2dsolver.m

    DLL-FDL scheme (https://arxiv.org/pdf/1512.05826)

    Solve for eigenstates of the cylindrical-symmetric 3D Schrödinger
    equation reduced to 2D in (r, z), with separation in φ and the
    transformation u = sqrt(r) * ψ.

    Inputs
    ------
    r : 1D array, shape (Nr,)
        Radial grid (meters). Must be uniformly spaced.
        In Ne.m, r = (0.5:Nr-0.5) * dr.
    z : 1D array, shape (Nz,)
        z grid (meters). Must be uniformly spaced.
    V : 2D array, shape (Nr, Nz)
        Potential energy in Joules at each (r_i, z_j).
        NOTE: In MATLAB you passed V' because you built V as (Nz x Nr).
    m : int
        Azimuthal quantum number.
    n_states : int, optional
        Number of eigenstates to compute (default 5).
    tol : float, optional
        Tolerance for eigensolver.
    maxit : int, optional
        Max iterations for eigensolver.

    Returns
    -------
    psi : 2D array, shape (Nr*Nz, n_states)
        Eigenvectors of H (in the u-basis) as columns.
        To reshape into (Nr, Nz) like MATLAB:
            psi_n = psi[:, n].reshape((Nr, Nz), order="F")
    E : 1D array, shape (n_states,)
        Eigenvalues (energies) in eV, sorted ascending.
    """
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

    # Grid spacings (assumed uniform)
    dr = r[1] - r[0]
    dz = z[1] - z[0]

    # ==========================
    # z part: standard 2nd-deriv
    # ==========================
    # Zdiag = hbar^2/(2*me) * 2/dz^2 * ones(...)
    # Zoff  = -hbar^2/(2*me) * 1/dz^2 * ones(...)
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

    # ==========================
    # r part: DLL-FDM radial op
    # ==========================

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

    # ==========================
    # Total Hamiltonian
    # ==========================
    Vvec = V.reshape(Nr * Nz, order="F")  # column-major
    V_full = spdiags(Vvec, 0, Nr * Nz, Nr * Nz, format="csc")

    Ir = identity(Nr, format="csc")
    Iz = identity(Nz, format="csc")

    Hr_full = kron(Iz, Hr, format="csc")
    Hz_full = kron(Hz, Ir, format="csc")

    # Hamiltonian in eV
    H = (Hr_full + Hz_full + V_full) / e

    # ==========================
    # Shift-invert eigensolver
    # ==========================
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
    vecs = vecs[:, idx]  # columns corresponding to sorted eigenvalues

    psi = vecs  # shape (Nr*Nz, n_states)

    return psi, E_vals
