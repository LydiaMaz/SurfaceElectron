"""
depth_scan.py
Study how burial depth d affects energy separation between ground (m=0,n=0) and first excited (m=1,n=0) states.
"""
#-----------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

from schrodinger_2d_solver import schrodinger_2d_solver
#-----------------------------------------------------------------------


def depth_scan_analysis():

    eps0 = 8.8541878128e-12
    eps = 1.244 * eps0 
    e = 1.602e-19

    # Depth scan setup
    da = np.linspace(100e-10, 1000e-10, 20)  # 100 Å to 1000 Å
    n_depths = len(da)
    
    # Arrays to store results
    E_ground = np.zeros(n_depths)   
    E_first_excited = np.zeros(n_depths) 
    delta_E = np.zeros(n_depths)  
    
    Nr = 1200
    r_max = 75e-9
    dr = r_max / Nr
    r = (np.arange(0.5, Nr + 0.5) * dr)
    
    Nz = 1200
    z = np.linspace(-2e-9, 50e-9, Nz)
    
    R, Z = np.meshgrid(r, z, indexing="ij")
    
    print(f"Starting depth scan: {da[0]*1e10:.1f} Å to {da[-1]*1e10:.1f} Å")
    print(f"Number of depths: {n_depths}")
    print("="*60)
    
    for i, d in enumerate(da):
        print(f"Depth {i+1}/{n_depths}: d = {d*1e10:.1f} Å", end=" ... ")
        
        # Point charge potential
        two_over = 2.0 / (1.0 + eps / eps0)
        denom = 4.0 * np.pi * eps0
        
        Vq = np.empty_like(R)
        mask_inside = (Z <= 0.0)
        mask_outside = ~mask_inside
        
        Vq[mask_inside] = 0.7 * e
        Vq[mask_outside] = -e**2 * two_over / (
            denom * np.sqrt((Z[mask_outside] + d)**2 + R[mask_outside]**2)
        )
        
        # Image charge potential 
        Vim = np.zeros_like(R)
        alpha = -((eps - eps0) / (eps + eps0)) * e**2 / (16.0 * np.pi * eps0)
        
        mask_z_lt_0 = (Z < 0.0)
        mask_z_mid = (Z > 0.0) & (Z < 2.3e-10)
        mask_z_gt = (Z >= 2.3e-10)
        
        Vim[mask_z_lt_0] = 0.0
        Vim[mask_z_mid] = alpha / 2.3e-10
        Vim[mask_z_gt] = alpha / Z[mask_z_gt]
        
        # Total potential
        V = Vq + Vim
        
        # Solve for m=0 (ground state)
        psi_m0, E_m0 = schrodinger_2d_solver(r, z, V, m=0, n_states=3)
        E_ground[i] = E_m0[0]  # Ground state
        
        # Solve for m=1 (first excited)
        psi_m1, E_m1 = schrodinger_2d_solver(r, z, V, m=1, n_states=3)
        E_first_excited[i] = E_m1[0]  # Lowest m=1 state
        
        # Energy separation
        delta_E[i] = (E_first_excited[i] - E_ground[i]) * 1000.0  # in meV
        
        print(f"ΔE = {delta_E[i]:.2f} meV")
    
    print("="*60)
    print("Depth scan completed!")
    
    # Plot results
    plot_depth_scan_results(da, E_ground, E_first_excited, delta_E)
    
    return da, E_ground, E_first_excited, delta_E


def plot_depth_scan_results(da, E_ground, E_first_excited, delta_E):

    da_angstrom = da * 1e10
    
    # Plot 1: Energy levels vs depth
    plt.figure(1, figsize=(12, 8))
    plt.plot(da_angstrom, E_ground * 1000, 'bo-', linewidth=2, markersize=6,
             label='Ground state (m=0, n=0)')
    plt.plot(da_angstrom, E_first_excited * 1000, 'ro-', linewidth=2, markersize=6,
             label='First excited (m=1, n=0)')
    plt.xlabel('Point charge depth d (Å)', fontsize=14)
    plt.ylabel('Energy (meV)', fontsize=14)
    plt.title('Energy levels vs point charge depth', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Plot 2: Energy separation vs depth
    plt.figure(2, figsize=(12, 6))
    plt.plot(da_angstrom, delta_E, 'go-', linewidth=2, markersize=6)
    plt.xlabel('Point charge depth d (Å)', fontsize=14)
    plt.ylabel('Energy separation ΔE (meV)', fontsize=14)
    plt.title('Energy separation (m=1,n=0) - (m=0,n=0) vs depth', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    
    plt.show()


if __name__ == "__main__":
    da, E_ground, E_first_excited, delta_E = depth_scan_analysis()