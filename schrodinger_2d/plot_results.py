"""
plot_results.py
Plot the wavefunction results from the 2D Schrödinger solver.
"""
#-----------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
#-----------------------------------------------------------------------


def plot_wavefunction_results(r, z, psi, E, Ngrid=900, title_suffix=""):

    Nr = r.size
    Nz = z.size
    
    def reconstruct_rho(col_index):
        psi_col = psi[:, col_index]  # shape (Nr*Nz,)
        psi_grid = psi_col.reshape((Nr, Nz), order="F")  # (Nr, Nz)
        # Divide by sqrt(r) along the radial axis
        psi_true = psi_grid / np.sqrt(r)[:, None]
        rho = np.abs(psi_true)**2       # (Nr, Nz)
        rho_zr = rho.T                  # transpose -> (Nz, Nr), index 0 ~ z, 1 ~ r
        return rho_zr

    # Reconstruct probability densities
    rho1 = reconstruct_rho(0)
    rho2 = reconstruct_rho(1) 
    rho3 = reconstruct_rho(2)
    rho4 = reconstruct_rho(3)
    rho5 = reconstruct_rho(4)

    # Truncate grids
    Ngrid = min(Ngrid, Nr, Nz)
    rhom1 = rho1[:Ngrid, :Ngrid].copy()
    rhom2 = rho2[:Ngrid, :Ngrid].copy()
    rhom3 = rho3[:Ngrid, :Ngrid].copy()
    rhom4 = rho4[:Ngrid, :Ngrid].copy()
    rhom5 = rho5[:Ngrid, :Ngrid].copy()

    rm = r[:Ngrid]
    zm = z[:Ngrid]

    # Print energy level separations
    dE01 = (E[1] - E[0]) * 1000.0
    dE12 = (E[2] - E[1]) * 1000.0
    dE23 = (E[3] - E[2]) * 1000.0
    dE34 = (E[4] - E[3]) * 1000.0
    print(f"ΔE01 = {dE01:.3f} meV, ΔE12 = {dE12:.3f} meV, "
          f"ΔE23 = {dE23:.3f} meV, ΔE34 = {dE34:.3f} meV")

    # Plot 1: |ψ|² vs r at fixed z (z = zm[0])
    plt.figure(1, figsize=(10, 6))
    plt.plot(rm[:400], rhom1[0, :400], linewidth=2,
             label=f"E0 = {E[0]*1000:.1f} meV")
    plt.plot(rm[:400], rhom2[0, :400], linewidth=2,
             label=f"E1 = {E[1]*1000:.1f} meV")
    plt.plot(rm[:400], rhom3[0, :400], linewidth=2,
             label=f"E2 = {E[2]*1000:.1f} meV")
    plt.xlabel("r (m)", fontsize=16)
    plt.ylabel(r"$|\psi|^2$", fontsize=16)
    plt.title(f"Radial probability density at surface {title_suffix}".strip(), fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Plot 2: |ψ|² vs z at fixed r (r = rm[0])
    plt.figure(2, figsize=(10, 6))
    plt.plot(zm, rhom1[:, 0], linewidth=2,
             label=f"E0 = {E[0]*1000:.1f} meV")
    plt.plot(zm, rhom2[:, 0], linewidth=2,
             label=f"E1 = {E[1]*1000:.1f} meV")
    plt.plot(zm, rhom3[:, 0], linewidth=2,
             label=f"E2 = {E[2]*1000:.1f} meV")
    plt.xlabel("z (m)", fontsize=16)
    plt.ylabel(r"$|\psi|^2$", fontsize=16)
    plt.title(f"Vertical probability density on axis {title_suffix}".strip(), fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Plot 3: 2D contour plot of ground state
    plt.figure(3, figsize=(12, 8))
    R_plot, Z_plot = np.meshgrid(rm, zm)
    levels = np.linspace(0, np.max(rhom1), 20)
    contour = plt.contourf(R_plot*1e9, Z_plot*1e9, rhom1, levels=levels, cmap='viridis')
    plt.colorbar(contour, label=r'$|\psi_0|^2$')
    plt.xlabel("r (nm)", fontsize=16)
    plt.ylabel("z (nm)", fontsize=16)
    plt.title(f"Ground state probability density (E0 = {E[0]*1000:.1f} meV) {title_suffix}".strip(), fontsize=14)
    plt.tight_layout()

    plt.show()

    return {
        'rho': [rhom1, rhom2, rhom3, rhom4, rhom5],
        'r_truncated': rm,
        'z_truncated': zm,
        'energy_separations': [dE01, dE12, dE23, dE34]
    }


def plot_energy_levels(states_list):

    fig, ax = plt.subplots(figsize=(8, 10))
    
    # Group states by m
    m0_states = [(En, n) for En, m, n in states_list if m == 0]
    m1_states = [(En, n) for En, m, n in states_list if m == 1]
    
    # Plot energy levels
    for En, n in m0_states:
        ax.hlines(En*1000, 0, 0.4, colors='blue', linewidth=3)
        ax.text(-0.1, En*1000, f'n={n}', ha='right', va='center', fontsize=10)
        
    for En, n in m1_states:
        ax.hlines(En*1000, 0.6, 1.0, colors='red', linewidth=3)
        ax.text(1.1, En*1000, f'n={n}', ha='left', va='center', fontsize=10)
    
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylabel('Energy (meV)', fontsize=14)
    ax.set_title('Energy Level Diagram', fontsize=16)
    ax.text(0.2, ax.get_ylim()[0] - (ax.get_ylim()[1]-ax.get_ylim()[0])*0.05, 
            'm = 0', ha='center', va='top', fontsize=12, color='blue', fontweight='bold')
    ax.text(0.8, ax.get_ylim()[0] - (ax.get_ylim()[1]-ax.get_ylim()[0])*0.05, 
            'm = 1', ha='center', va='top', fontsize=12, color='red', fontweight='bold')
    ax.set_xticks([])
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("This is a plotting utility. Import and use plot_wavefunction_results() or plot_energy_levels().")