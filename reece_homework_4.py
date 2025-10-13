import numpy as np
import iapws
from iapws import IAPWS97
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.sparse import lil_matrix, csr_matrix, diags
from matplotlib.animation import FuncAnimation

# --- Physical parameters and setup ---
P_out = 0.20265  # MPa, outlet pressure
P_in = P_out*2   # MPa, inlet pressure
T_out = T_bot = 273.15  # K, bottom temperature
T_top = 363.15  # K, top temperature

# --- Mesh and geometry parameters ---
N_water = 120     # number of radial nodes in water region
N_wall = 12       # number of radial nodes in wall region
N_vert = 200      # number of axial nodes in water region

z_interface_up = 0.12 - 0.0002  # m, top of water region
z_interface_down = 0.0002       # m, bottom of water region

z_wall_bottom = 0    # m, bottom of wall
z_wall_top    = 0.12 # m, top of wall

r_interface = 0.03 - 0.0002  # m, radius separating water/wall
delta_r = (r_interface) / N_water
delta_w = (0.03 - r_interface) / N_wall
delta_z_water = (z_interface_up - z_interface_down) / N_vert

# --- Axial wall mesh nodes (bottom and top) ---
delta_z_wall_bottom = z_interface_down / N_wall
z_wall_bottom_nodes = np.linspace(delta_z_wall_bottom/2,
                                  z_interface_down - delta_z_wall_bottom/2,
                                  N_wall)

delta_z_wall_top = z_interface_down / N_wall
z_wall_top_nodes = np.linspace(z_interface_up+delta_z_wall_top/2,
                               z_wall_top - delta_z_wall_top/2,
                               N_wall)

# --- Radial mesh for water and wall ---
r_water = np.linspace(delta_r/2, r_interface - delta_r/2, N_water)  # m
r_wall = np.linspace(r_interface + delta_w/2, 0.03 - delta_w/2, N_wall)  # m

# --- Axial nodes for water region ---
z_water_nodes = np.linspace(z_wall_bottom + delta_z_water/2, z_wall_top - delta_z_water/2, N_vert)

# --- Combine into total meshes ---
total_nodes = np.concatenate((r_water, r_wall))  # radial coordinates [m]
z_nodes = np.concatenate([z_wall_bottom_nodes, z_water_nodes, z_wall_top_nodes])  # axial [m]

N_r = len(total_nodes)
N_z = len(z_nodes)

# faces and cell thicknesses
r_faces = np.zeros(N_r + 1)
r_faces[1:-1] = 0.5 * (total_nodes[:-1] + total_nodes[1:])
r_faces[0] = 0.0
r_faces[-1] = total_nodes[-1] + 0.5 * (total_nodes[-1] - total_nodes[-2])
dr_cells = np.diff(r_faces)               # Δr for each radial cell (length N_r)

z_faces = np.zeros(N_z + 1)
z_faces[1:-1] = 0.5 * (z_nodes[:-1] + z_nodes[1:])
z_faces[0] = z_nodes[0] - 0.5 * (z_nodes[1] - z_nodes[0])
z_faces[-1] = z_nodes[-1] + 0.5 * (z_nodes[-1] - z_nodes[-2])
dz_cells = np.diff(z_faces)               # Δz for each axial cell (length N_z)

# --- Initial conditions and constants ---
orig_T = [35+273.15]*(N_r*N_z)  # K, initial uniform temperature
target_T = 10+273.15  # K, target average water temperature
h_out = h_bot = 956.25614  # W/m²·K, convection at outer/bottom surfaces
h_top = 5.13               # W/m²·K, convection at top
r_wall = 0.03-0.000175     # m, inner wall radius
r_outer = 0.03             # m, outer wall radius
delt = 10                  # s, time step (not used directly)
delT = 0.5                 # K, small ΔT for property derivative calculations
del_space = total_nodes[1:] - total_nodes[:-1]  # radial spacing array [m]

# --- Material property models ---
def aluminum_properties(T):
   """
   Compute aluminum properties as functions of temperature T [K].
   Returns:
       rho [kg/m³], cp [J/kg·K], k [W/m·K]
   """
   rho = (-2e-4*T**2) + (0.0282*T) +  2697.6
   if T >= 298:
      c_p = (0.3162*T) + 788.95
   else:
      c_p = (0.3162*298) + 788.95
   k = (2e-7*T**3) - (4e-4*T**2) + (0.2456*T) + 196.91
   return rho, c_p, k


def water_props(T):
    """
    Compute water properties using IAPWS97 at inlet pressure P_in [MPa].
    Inputs:
        T : temperature [K] (scalar or array)
    Returns:
        rho [kg/m³], cp [J/kg·K], k [W/m·K]
    """
    T = np.atleast_1d(T)
    rho_vals, cp_vals, k_vals = [], [], []
    for Ti in T:
        w = IAPWS97(P=P_in, T=Ti)
        rho_vals.append(w.rho)
        cp_vals.append(w.cp * 1000)  # convert kJ/kg·K → J/kg·K
        k_vals.append(w.k)
    rho_vals, cp_vals, k_vals = map(np.array, (rho_vals, cp_vals, k_vals))
    if rho_vals.size == 1:
        return float(rho_vals[0]), float(cp_vals[0]), float(k_vals[0])
    return rho_vals, cp_vals, k_vals

def build_A_matrix_volume_safe_nonuniform(r, z, k_flat, denom_tol=1e-12):
    """
    Build finite-volume matrix A and source term b for transient 2D (r,z) conduction.
    Axisymmetric form with non-uniform mesh. Expects radial fastest flattening
    (i + j*N_r)

    Inputs:
        r : radial node array [m]
        z : axial node array [m]
        k_flat : flattened conductivity array [W/m·K]
        denom_tol : small tolerance for division by near-zero geometry terms

    Returns:
        A : sparse FV coefficient matrix (CSR format)
        b : source vector (W/m³·K term for convective boundaries)

    Geometry: cylindrical (r,z)
    Equation form: (ρ·c_p)·dT/dt = A·T + b
    """
    # (Full implementation follows...)
    # [All geometric and harmonic averaging handled inside]
    # BCs:
    #   - outer radius: convective to ambient (h_out, T_out)
    #   - top/bottom: convective to T_top/T_bot
    #   - symmetry on r=0 handled by FV formulation

    A = lil_matrix((N_r * N_z, N_r * N_z), dtype=float)
    b = np.zeros(N_r * N_z, dtype=float)

    k = np.asarray(k_flat, dtype=float).reshape((N_r, N_z), order='F')

    def idx(i, j):
        return i + N_r * j

    def harm(a, b):
        return (2.0 * a * b / (a + b)) if (a > 0.0 and b > 0.0) else 0.0

    for j in range(N_z):
        dz_cell = dz_cells[j]
        for i in range(N_r):
            p = idx(i, j)

            # face radii
            r_im = r_faces[i]       # r_{i-1/2}
            r_ip = r_faces[i + 1]   # r_{i+1/2}
            denom_r = (r_ip**2 - r_im**2)
            if abs(denom_r) < denom_tol:
                # fallback small-radius regularization
                # try to use a small physical scale rather than zero
                if i == 0:
                    denom_r = max(denom_r, 0.5 * r_ip * (r[1] - r[0]))
                else:
                    denom_r = max(denom_r, 0.5 * r[i] * ( (r[i] - r[i-1]) + (r[i+1] - r[i]) 
                                if i < N_r-1 else (r[i] - r[i-1]) ))

            # harmonic conductivity at faces
            k_ip = harm(k[i, j], k[i + 1, j]) if (i < N_r - 1) else k[i, j]
            k_im = harm(k[i, j], k[i - 1, j]) if (i > 0) else k[i, j]
            k_jp = harm(k[i, j], k[i, j + 1]) if (j < N_z - 1) else k[i, j]
            k_jm = harm(k[i, j], k[i, j - 1]) if (j > 0) else k[i, j]

            # center-to-center distances (for gradient at faces)
            dr_center_e = (r[i + 1] - r[i]) if (i < N_r - 1) else (r[i] - r[i - 1])
            dr_center_w = (r[i] - r[i - 1]) if (i > 0) else (r[1] - r[0])

            dz_cc_n = (z[j + 1] - z[j]) if (j < N_z - 1) else (z[j] - z[j - 1])
            dz_cc_s = (z[j] - z[j - 1]) if (j > 0) else (z[1] - z[0])

            # Radial conductances
            G_e = 0.0
            if i < N_r - 1:
                G_e = 2.0 * r_ip * k_ip / (denom_r * dr_center_e)
                A[p, idx(i + 1, j)] += G_e

            G_w = 0.0
            if i > 0:
                G_w = 2.0 * r_im * k_im / (denom_r * dr_center_w)
                A[p, idx(i - 1, j)] += G_w

            # Axial conductances
            G_n = 0.0
            if j < N_z - 1:
                # gradient at face uses center-to-center dz_cc_n, and divide by local cell height dz_cell
                G_n = k_jp / (dz_cc_n * dz_cell)
                A[p, idx(i, j + 1)] += G_n

            G_s = 0.0
            if j > 0:
                G_s = k_jm / (dz_cc_s * dz_cell)
                A[p, idx(i, j - 1)] += G_s

            # Diagonal (negative sum; matches your 1-D sign convention)
            A[p, p] = - (G_e + G_w + G_n + G_s)

            # Convective BCs — properly scaled to per-volume coefficients
            # Outer radial convection (i == N_r-1)
            if i == N_r - 1:
                # face radius used for outer face (r_{i+1/2})
                r_face = r_faces[i + 1]
                denom_face = r_face**2 - r_faces[i]**2
                if abs(denom_face) < denom_tol:
                    denom_face = max(denom_face, 0.5 * r_face * dr_cells[i])
                G_conv_r = 2.0 * h_out * r_face / denom_face
                A[p, p] -= G_conv_r
                b[p] += G_conv_r * T_out

            # Top axial BC (j == N_z-1): Robin / convective on the top face -> contribution ~ h / dz_cell
            if j == N_z - 1:
                G_conv_top = h_top / dz_cell
                A[p, p] -= G_conv_top
                b[p] += G_conv_top * T_top

            # Bottom axial BC (j == 0)
            if j == 0:
                G_conv_bot = h_bot / dz_cell
                A[p, p] -= G_conv_bot
                b[p] += G_conv_bot * T_bot

    return A.tocsr(), b

def composite_properties(T_flat, r_nodes, z_nodes, tol=1e-12):
    """
    Combine material properties for each control volume depending on region.
    Water in central region, aluminum at walls/top/bottom.

    Inputs:
        T_flat : flattened temperature array [K]
        r_nodes : radial coordinates [m]
        z_nodes : axial coordinates [m]
    Returns:
        rho, cp, k : flattened arrays [kg/m³], [J/kg·K], [W/m·K]
    """
    rho = np.zeros_like(T_flat)
    cp  = np.zeros_like(T_flat)
    k   = np.zeros_like(T_flat)

    for i_z, z in enumerate(z_nodes):
        for i_r, r in enumerate(r_nodes):
            idx = i_r + N_r * i_z  # radial varies fastest
            Ti = T_flat[idx]

            # Check axial position first for top/bottom wall
            if z <= z_interface_down + tol or z >= z_interface_up - tol:
                rho_i, cp_i, k_i = aluminum_properties(Ti)
            else:
                # Check radial position
                if r < r_interface - tol:
                    rho_i, cp_i, k_i = water_props(Ti)
                elif r > r_interface + tol:
                    rho_i, cp_i, k_i = aluminum_properties(Ti)
                else:
                    # Node near radial interface → weighted average
                    rho_w, cp_w, k_w = water_props(Ti)
                    rho_al, cp_al, k_al = aluminum_properties(Ti)
                    frac = (r_interface - (r - tol)) / (2*tol)
                    rho_i = frac*rho_w + (1-frac)*rho_al
                    cp_i  = frac*cp_w  + (1-frac)*cp_al
                    k_i   = frac*k_w   + (1-frac)*k_al

            rho[idx] = rho_i
            cp[idx]  = cp_i
            k[idx]   = k_i

    return rho, cp, k

# --- Time derivative function for ODE solver ---
def dTdt_vol(t, T):
    """
    Returns dT/dt for all control volumes.
    Called by solve_ivp during transient integration.
    """
    print(t)
    if t == 0:
        # Use initial properties
        A, b = build_A_matrix_volume_safe_nonuniform(total_nodes, z_nodes, k_0)
        return (A.dot(T) + b) / (rho_0 * cp_0)
    else:
        # Update properties as temperature evolves
        rho, cp, k = composite_properties(T, total_nodes, z_nodes)
        A, b = build_A_matrix_volume_safe_nonuniform(total_nodes, z_nodes, k)
        return (A.dot(T) + b) / (rho * cp)  # elementwise division


def jac_A_div_rhocp(t, T):
    """
    Jacobian matrix for implicit solver.
    Computes J = (1/(ρ·c_p))·A as sparse matrix.
    """
    rho, cp, k = composite_properties(T, total_nodes, z_nodes)
    A, b = build_A_matrix_volume_safe_nonuniform(total_nodes, z_nodes, k)
    inv_rhocp = 1.0 / (rho * cp)
    J = diags(inv_rhocp) @ A
    return J

def main():
    rho_0, cp_0, k_0 = composite_properties(orig_T, total_nodes, z_nodes)

    # Solve transient heat equation
    sol_t = np.linspace(0, 3000, 3001)
    sol = solve_ivp(
        fun=dTdt_vol,
        t_span=(0, 3000.0),
        y0=orig_T,
        method='BDF',
        jac=jac_A_div_rhocp,
        rtol=1e-3,
        atol=1e-4,
        first_step=0.1,
        t_eval=sol_t
    )

    # Extract water region temperatures
    index_r = 0
    for idx, i in enumerate(total_nodes):
        if i < r_interface:
            index_r = idx

    index_j_t = index_j_b = 0
    for jdx, j in enumerate(z_nodes):
        if j < z_interface_down:
            index_j_b = jdx
        if j < z_interface_up:
            index_j_t = jdx

    T_water = np.zeros((N_water, N_vert, sol.t.size))
    for i in range(0, index_r + 1):
        for j in range(index_j_b + 1, index_j_t + 1):
            T_water[i, j - (index_j_b + 1)] = sol.y.reshape(N_r, N_z, -1, order='F')[i, j, :]

    # Mean temperature
    T_mean = np.mean(T_water, axis=(0, 1))

    # Find time when target temperature reached
    target_T = 10 + 273.15
    below = np.where(T_mean < target_T)[0]
    if len(below):
        t_target = sol.t[below[0]]
        print(f"Target reached at t = {t_target:.2f} s, T_mean = {T_mean[below[0]]:.2f} K")

    # Visualization
    i_center = np.argmin(np.abs(total_nodes))
    j_center = len(z_nodes) // 2

    T_all = sol.y.reshape(N_r, N_z, -1, order='F')
    vmin, vmax = T_all.min(), T_all.max()

    R, Z = np.meshgrid(z_nodes, total_nodes)
    fig, ax = plt.subplots(figsize=(4, 8))
    pcm = ax.pcolormesh(Z, R, T_all[:, :, 0], shading='auto', cmap='inferno', vmin=vmin, vmax=vmax)
    ax.set_xlabel('r [m]')
    ax.set_ylabel('z [m]')
    cbar = fig.colorbar(pcm)
    cbar.set_label('Temperature [K]')

    text = ax.text(0.02, 0.95, '', color='white', fontsize=10, transform=ax.transAxes)
    text_1 = ax.text(0.02, 0.05, '', color='white', fontsize=10, transform=ax.transAxes)

    def update(frame):
        pcm.set_array(T_all[:, :, frame].ravel())  # flatten for pcolormesh
        ax.set_title(f'Time = {sol.t[frame]:.1f} s')
        text.set_text(f"T_center = {T_all[i_center, j_center, frame]:.2f} K")
        text_1.set_text(f"T_mean = {T_mean[frame]:.1f} K")
        return [pcm, text, text_1]

    skip = 5
    ani = FuncAnimation(fig, update, frames=range(0, T_all.shape[2], skip), interval=50, blit=True)
    ani.save('temperature_evolution.gif', writer='pillow', fps=60)
    plt.close(fig)

if __name__ == "__main__":
    main()