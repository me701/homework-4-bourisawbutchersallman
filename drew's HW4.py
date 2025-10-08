import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

R, L = 0.05, 0.12 #radius and can length
Nr, Nz = 20, 20 #number of nodes in r and z
dr, dz = R / Nr, L / Nz #size of nodes
r = np.linspace(dr/2, R-dr/2, Nr)
z = np.linspace(dz/2, L-dz/2, Nz)

#material properties
rho = 1000
cp = 1000
k0 = 0.6
alpha = 0.00118    
T_ref = 300.0

#time steps and temperatures
dt = 0.5
nt = 2000
T_ice = 100
T0 = 300.0
T_avg_history=[]

#initial temperatures in can for all nodes
T = np.ones((Nr, Nz)) * T0
T_new = np.zeros_like(T)

#define the center node of the can
i_center = 0
j_center = Nz // 2

# --- setup figure for animation ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))
im = ax1.imshow(T.T, origin='lower', extent=[0,L,0,R], aspect='auto', vmin=T_ice, vmax=T0)
cbar = fig.colorbar(im)
cbar.set_label('Temperature [K]')
ax1.set_xlabel('z [m]')
ax1.set_ylabel('r [m]')
ax1.set_title('Temperature Distribution')

avg_line, = ax2.plot([], [], 'r-')
ax2.set_xlim(0, nt*dt)
ax2.set_ylim(T_ice, T0)
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Average Interior Temp [K]')
ax2.set_title('Average Temperature vs Time')

time_counter = 0.0

# --- add text for counter ---
time_text = ax1.text(0.02*L, 0.95*R, '', color='white', fontsize=12, alpha=0.5)

#each frame, rune this function for a new graphical distribution
def update(frame):
    global T, T_new, time_counter
    
    k = k0 * (1 + alpha * (T - T_ref))  # update thermal conductivity

    # --- finite differences for interior nodes ---
    for i in range(1, Nr-1):
        for j in range(1, Nz-1):
            k_e = 2.0 / (1.0/k[i+1,j] + 1.0/k[i,j])
            k_w = 2.0 / (1.0/k[i-1,j] + 1.0/k[i,j])
            k_n = 2.0 / (1.0/k[i,j+1] + 1.0/k[i,j])
            k_s = 2.0 / (1.0/k[i,j-1] + 1.0/k[i,j])

            d2T_dr2 = (k_e*(T[i+1,j]-T[i,j]) - k_w*(T[i,j]-T[i-1,j])) / dr**2
            d2T_dz2 = (k_n*(T[i,j+1]-T[i,j]) - k_s*(T[i,j]-T[i,j-1])) / dz**2
            dT_dr_over_r = (k[i,j]/r[i]) * (T[i+1,j]-T[i-1,j])/(2*dr)
            
            T_new[i,j] = T[i,j] + dt/(rho*cp) * (d2T_dr2 + d2T_dz2 + dT_dr_over_r)
    
    # --- boundary conditions ---
    T_new[0,:] = T_new[1,:]      # symmetry at center
    T_new[-1,:] = T_ice          # outer radius
    T_new[:,0]  = T_ice          # bottom
    T_new[:,-1] = T_ice          # top
    
    T = T_new.copy()
    
    # --- update counter and plot ---
    time_counter += dt
    time_text.set_text(f'Time = {time_counter:.1f} s')

    # --- compute average interior temperature ---
    T_avg = np.mean(T[1:-1, 1:-1])
    T_avg_history.append(T_avg)

    im.set_array(T)
    avg_line.set_data(np.arange(len(T_avg_history))*dt, T_avg_history)
    
    return [im, time_text, avg_line]

#animation!
ani = FuncAnimation(fig, update, frames=nt, blit=True, interval=1, repeat=False)
plt.show()