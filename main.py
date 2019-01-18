import numpy

from matplotlib import pyplot, cm
from matplotlib.colors import Normalize

# {nx, ny} represent the number of cells we are mapping out onto
nx = 41
ny = 41

# {Lx, Ly} represent the original spatial map we wish to represent
Lx = 2
Ly = 2

# {dx, dy} represent the spatial increments
dx = Lx / float(nx - 1)
dy = Lx / float(ny - 1)

# {u, v} represent the {x, y} components of the velocity vectors
u = numpy.ones((ny, nx))
v = numpy.ones((ny, nx))

# {dt} is the time evolution
sigma = 0.001
nu = 0.01
dt = sigma * dx * dy / nu


def equation_of_motion(u, v, dx, dy, dt, nu):
	# generate the next state as a function of the old state
	un = u.copy()
	vn = v.copy()

	u[1:-1, 1:-1] = (
	    un[1:-1, 1:-1] - dt / dx * un[1:-1, 1:-1] *
	    (un[1:-1, 1:-1] - un[1:-1, 0:-2]) - dt / dy * vn[1:-1, 1:-1] *
	    (un[1:-1, 1:-1] - un[0:-2, 1:-1]) + nu * dt / dx**2 *
	    (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
	    nu * dt / dy**2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1]))

	v[1:-1, 1:-1] = (
	    vn[1:-1, 1:-1] - dt / dx * un[1:-1, 1:-1] *
	    (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) - dt / dy * vn[1:-1, 1:-1] *
	    (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) + nu * dt / dx**2 *
	    (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
	    nu * dt / dy**2 * (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1]))

	return (u, v)


def boundary(u, v, nozzle_u, nozzle_v, nx, ny, t_step):
	u[0, :] = 0
	u[-1, :] = 0
	u[:, 0] = 0
	u[:, -1] = 0

	v[0, :] = 0
	v[-1, :] = 0
	v[:, 0] = 0
	v[:, -1] = 0

	# special nozzle boundary conditions
	u[nx // 2 - 2:nx // 2 + 2, 0] = nozzle_u[t_step]
	v[ny // 2 - 2:ny // 2 + 2, 0] = nozzle_v[t_step]

	return (u, v)

def evolve(u, v, dx, dy, dt, nu, nozzle_u, nozzle_v, nx, ny, steps):
	
	# Initialize the pyplot
	
	
	for i in range(steps):
		(u, v) = equation_of_motion(u, v, dx, dy, dt, nu)
		(u, v) = boundary(u, v, nozzle_u, nozzle_v, nx, ny, i)

		if (i % 50 == 0):
			ax = pyplot.figure()
			norm = Normalize()
			magnitude = numpy.sqrt(u[::2]**2 + v[::2]**2)
			pyplot.quiver(
			    u[::2], v[::2], norm(magnitude), scale=60, cmap=pyplot.cm.jet)
			ax.savefig('frame' + str(i).zfill(5) + '.png', dpi=300)
			ax.clear()
			pyplot.close()
	return (u, v)


# {nt} is the number of steps we are simulating
nt = 2510

# assigning initial conditions with {initial_u, initial_v}
initial_u = numpy.zeros((ny, nx))
initial_v = numpy.zeros((ny, nx))

# special boundary conditions for nozzle
# located at (0, 1)
nozzle_u = numpy.append(10 * numpy.ones(1000), numpy.zeros(nt))
nozzle_v = numpy.append(10 * numpy.ones(1000), numpy.zeros(nt))

(final_u, final_v) = evolve(initial_u, initial_v, dx, dy, dt, nu, nozzle_u,
                            nozzle_v, nx, ny, nt)
