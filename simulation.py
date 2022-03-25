"""
Adiabatic Shock (modified from https://github.com/evjhlee/phys432-w2022/blob/main/phys432_w2022_sound_wave.py)
@author: Yuliya Shpunarska

This code doesn't work as expected but here is my attempt anyways.
March 24th 2022
"""
import numpy as np
import matplotlib.pyplot as pl

# Set up the grid, time and grid spacing, and the sound speed squared
Ngrid = 100
Nsteps = 5000
dt = 0.01
dx = 2.0
gamma = 5/3 # Adiabatic index

x = np.arange(Ngrid) * dx # grid
f1 = np.ones(Ngrid) # rho
f2 = np.zeros(Ngrid) # rho x u
f3 = np.ones(Ngrid) # rho x e_tot

cs2 = np.ones(Ngrid) # sound speed squared

mach = f2/f1 / np.sqrt(cs2) # initial Mach number
u = np.zeros(Ngrid+1) # advective velocity (keep the 1st and last element zero)

def advection(f, u, dt, dx):
    # calculating flux terms
    J = np.zeros(len(f)+1) # keeping the first and the last term zero
    J[1:-1] = np.where(u[1:-1] > 0, f[:-1] * u[1:-1], f[1:] * u[1:-1])
    f = f - (dt / dx) * (J[1:] - J[:-1]) #update

    return f

# Apply initial Gaussian perturbation to energy only
Amp, sigma = 100, Ngrid/100
f3 = f3 + Amp * np.exp(-(x - x.max()/2) ** 2 / sigma ** 2)

# plotting
pl.ion()
fig, (ax1, ax2) = pl.subplots(2,1, sharex=True)
x1, = ax1.plot(x, f1, 'ro')
x2, = ax2.plot(x, mach, "ko")

ax1.set_xlim([0, dx*Ngrid+1])
ax1.set_ylim([-Amp, Amp])

ax1.set_ylabel('Density')
ax2.set_ylabel("Mach Number")
ax2.set_ylim([-Amp, Amp])

ax2.set_xlabel('x')

fig.canvas.draw()

for ct in range(Nsteps):
    # 1. Compute the advection velocity at the cell interfaces and at the simulation boundaries.
    u[1:-1] = 0.5 * ((f2[:-1] / f1[:-1]) + (f2[1:] / f1[1:]))

    # 2. Advect density, then momentum
    f1 = advection(f1, u, dt, dx)
    f2 = advection(f2, u, dt, dx)

    # 3. Compute pressure and apply the pressure gradient force to the momentum
    # equation. Make sure to apply the correct boundary condition for the source term.
    P = cs2*f1/gamma # pressure update

    # add the source term to momentum equation
    f2[1:-1] = f2[1:-1] - 0.5 * (dt / dx) * cs2[1:-1] * (f1[2:] - f1[:-2]) #* (P[2:] - P[:-2])

    # correct for source term at the boundary (reflective)
    f2[0] = f2[0] - 0.5 * (dt / dx) * cs2[0] * (f1[1] - f1[0]) #* (P[1] - P[0])
    f2[-1] = f2[-1] - 0.5 * (dt / dx) * cs2[-1] * (f1[-1] - f1[-2]) #* (P[-1] - P[-2])

    # 4. Re-calculate the advection velocities
    u[1:-1] = 0.5 * ((f2[:-1] / f1[:-1]) + (f2[1:] / f1[1:]))
    u[0] = 0
    u[-1] = 0

    # 5. Advect energy
    f3 = advection(f3, u, dt, dx)

    # 6. Re-compute pressure and apply the corresponding source term to the energy equation.
    P = cs2*f1/gamma # pressure update
    mach = f2/f1 / np.sqrt(cs2) # Mach number update

    f3[1:-1] = f3[1:-1] - 0.5 * (P[1:-1] / dx) * f2[1:-1]/f1[1:-1]

    # 7. Again, correct for the source term at the simulation boundaries for
    # the energy equation, using the correct boundary condition

    # reflective boundary condition
    f3[0] = f3[0] - 0.5 * (P[0] / dx) * f2[0]/f1[0]
    f3[-1] = f3[-1] - 0.5 * (P[-1] / dx) * f2[-1]/f1[-1]

    # 8. Before updating your plot, re-calculate pressure and sound speed.
    P = cs2*f1/gamma # pressure update
    cs2 = gamma*P/f1 # sound speed update

    # update the plot
    x1.set_ydata(f1)
    x2.set_ydata(mach)
    fig.canvas.draw()
    pl.pause(0.001)
