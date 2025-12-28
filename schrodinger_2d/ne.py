import numpy as np

from schrodinger_2d_solver import schrodinger_2d_solver
from plot_results import plot_wavefunction_results, plot_energy_levels


def main():

    eps0 = 8.8541878128e-12
    eps = 1.244 * eps0                  # dielectric constant of solid neon
    hbar = 1.0545718e-34
    me = 9.10938356e-31
    e = 1.602e-19

    #-----------------------------------------------------------------------
    # Depth scan setup
    #-----------------------------------------------------------------------

    da = np.linspace(100e-10, 1000e-10, 10)  # 100 Å to 1000 Å
    Egs = np.zeros(10)
    E1s = np.zeros(10)
    E2n = np.zeros(10)

    
    k = 0  
    d = da[k]

    # grids
    Nr = 1200
    r_max = 75e-9
    dr = r_max / Nr
    r = (np.arange(0.5, Nr + 0.5) * dr) 

    Nz = 1200
    z = np.linspace(-2e-9, 50e-9, Nz) # [-2nm, 50nm]

    Nr = r.size
    Nz = z.size

    R, Z = np.meshgrid(r, z, indexing="ij") 

    #-----------------------------------------------------------------------
    # Point charge potential Vq
    #-----------------------------------------------------------------------

    two_over = 2.0 / (1.0 + eps / eps0)
    denom = 4.0 * np.pi * eps0

    Vq = np.empty_like(R)

    mask_inside = (Z <= 0.0)
    mask_outside = ~mask_inside

    Vq[mask_inside] = 0.7 * e
    Vq[mask_outside] = -e**2 * two_over / (
        denom * np.sqrt((Z[mask_outside] + d)**2 + R[mask_outside]**2)
    )

    #-----------------------------------------------------------------------
    # Dipole potential Vd 
    #-----------------------------------------------------------------------

    dd = 3e-10
    p = e * dd
    Vd = np.empty_like(R)

    mask_inside_d = (Z <= 0.0)
    mask_outside_d = ~mask_inside_d

    Vd[mask_inside_d] = 0.7 * e
    Vd[mask_outside_d] = -e * p * (Z[mask_outside_d] + d) / (
        4.0 * np.pi * eps0 * ((Z[mask_outside_d] + d)**2 + R[mask_outside_d]**2)**(1.5)
    )

    #-----------------------------------------------------------------------
    # Image charge potential Vim
    #-----------------------------------------------------------------------

    Vim = np.zeros_like(R)
    alpha = -((eps - eps0) / (eps + eps0)) * e**2 / (16.0 * np.pi * eps0)

    mask_z_lt_0 = (Z < 0.0)                      # inside neon
    mask_z_mid = (Z > 0.0) & (Z < 2.3e-10)       # close to surface
    mask_z_gt = (Z >= 2.3e-10)                   # far from surface

    Vim[mask_z_lt_0] = 0.0
    Vim[mask_z_mid] = alpha / 2.3e-10
    Vim[mask_z_gt] = alpha / Z[mask_z_gt]

    #-----------------------------------------------------------------------
    # Total potential V (J)
    #-----------------------------------------------------------------------

    V = Vq + Vim  

    #-----------------------------------------------------------------------
    # Solve Schrödinger equation for m=0 and m=1
    #-----------------------------------------------------------------------

    all_states = []
    all_psi = {}
    
    for m in [0, 1]:
        print(f"\nSolving for m = {m}...")
        psi, E = schrodinger_2d_solver(r, z, V, m=m, n_states=5)
        all_psi[m] = psi
        
        for n, En in enumerate(E):
            all_states.append((En, m, n))
        
        print(f"m={m}: E0={E[0]*1000:.1f} meV, E1={E[1]*1000:.1f} meV, E2={E[2]*1000:.1f} meV")
    
    # Sort all states by energy 
    all_states_sorted = sorted(all_states)
    
    print(f"\nGlobal energy spectrum (first 6 states):")
    for i, (En, m, n) in enumerate(all_states_sorted[:6]):
        print(f"State {i}: E = {En*1000:.1f} meV (m={m}, n={n})")
    
    Egs[k] = all_states_sorted[0][0]  # Ground state energy
    E1s[k] = all_states_sorted[1][0]  # First excited state
    E2n[k] = all_states_sorted[2][0]  # Second excited state
    
    plot_energy_levels(all_states_sorted[:6])
    
    ground_m = all_states_sorted[0][1]  
    first_excited_m = all_states_sorted[1][1] 
    
    #-----------------------------------------------------------------------
    # Plotting
    #-----------------------------------------------------------------------
    
    print(f"\nPlotting wavefunctions...")
    print(f"Ground state: m={ground_m}")
    results_ground = plot_wavefunction_results(r, z, all_psi[ground_m], 
                                             [s[0] for s in all_states if s[1] == ground_m], 
                                             Ngrid=900, title_suffix=f"m={ground_m}")
    
    if first_excited_m != ground_m:
        print(f"First excited state: m={first_excited_m}")  
        results_excited = plot_wavefunction_results(r, z, all_psi[first_excited_m],
                                                   [s[0] for s in all_states if s[1] == first_excited_m],
                                                   Ngrid=900, title_suffix=f"m={first_excited_m}")


if __name__ == "__main__":
    main()
