import matplotlib.pyplot as plt
import numpy as np
import PIL
from math import *
import scipy
import numpy.linalg
from matplotlib.widgets import CheckButtons


def rotz(rotzg):  # Calculation of rotation
    return [[cos(rotzg), -sin(rotzg), 0], [sin(rotzg), cos(rotzg), 0], [0, 0, 1]]


def get_orbit_n(r, sol1):  # Calculation of orbital norm
    phi = 90 * pi / 180
    p1 = -r[1] / r[0]
    p2 = -cos(phi) * r[2] / r[0]

    a = p1 ** 2 + 1
    b = 2 * p1 * p2
    c = p2 ** 2 - sin(phi) * sin(phi)

    y1 = (-b + sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    y2 = (-b - sqrt(b ** 2 - 4 * a * c)) / (2 * a)

    x1 = p1 * y1 + p2
    x2 = p1 * y2 + p2

    z = cos(phi)

    n1 = [x1, y1, z]
    n2 = [x2, y2, z]
    if sol1:
        return n1
    else:
        return n2


def odefun(x, t):  # a differential equation for which we will find a solution
    return np.concatenate([x[3:6], (-G * M * x[:3]) / ((numpy.linalg.norm(x[:3])) ** 3)])


# Initial values

N = float(input('North cords: '))
E = float(input('East cords: '))
G = 6.67 * 10 ** -11
R = 6371000
M = 5.972 * 10 ** 24
height = float(input('Height of satellite, m: '))
satellite_velocity = float(input('Start velocity of satellite, m/s: '))
time = 90 * 60
am = float(input('Amount of moments of the time: '))
mass = 420000

# Calculation

North = N * pi / 180
East = E * pi / 180
init_pos = np.array([cos(North) * cos(East), cos(North) * sin(East), sin(North)])

orbit_n = np.array((get_orbit_n(init_pos, True)))
tau = np.cross(orbit_n, init_pos)
r0 = init_pos * (R + height)
v0 = tau * satellite_velocity
x0 = np.concatenate([r0, v0])

tspan = np.linspace(0, am * time, 10 ** 5)

x = scipy.integrate.odeint(odefun, x0, tspan, rtol=1e-13, atol=1e-14)
trajectory = x[:, 0:3]
velocity = x[:, 3:6]

trajectory_corr1 = np.zeros(len(trajectory))
trajectory_corr2 = np.zeros(len(trajectory))
trajectory_corr3 = np.zeros(len(trajectory))

kinetic_energy = np.zeros(tspan.shape[0])
potential_energy = np.zeros(tspan.shape[0])

for i in range(len(tspan)):
    current_time = tspan[i]
    angle_Earth_rotation = -2 * pi * (current_time / (24 * 60 * 60))
    current_point = np.transpose(trajectory[i, :])
    current_point_corr = np.dot(rotz(angle_Earth_rotation), current_point)

    trajectory_corr1[i] = list(current_point_corr)[0]
    trajectory_corr2[i] = list(current_point_corr)[1]
    trajectory_corr3[i] = list(current_point_corr)[2]

    kinetic_energy[i] = 0.5 * mass * np.dot(velocity[i, :], velocity[i, :])
    potential_energy[i] = (-G * M * mass) / np.linalg.norm(current_point)

total_energy = potential_energy + kinetic_energy

# Earth and satellite simulation

fig = plt.figure('Simulation of')
ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.set_zlim(-8e6, 8e6)
ax.set_xlim(-8e6, 8e6)
ax.set_ylim(-8e6, 8e6)

ax.plot(0, 0, height + R, 'o', linewidth=2)

ax.plot3D(trajectory_corr1[:], trajectory_corr2[:], trajectory_corr3[:], linewidth=2)

ea = PIL.Image.open('earthicefree.jpg')
ea = np.array(ea.resize([int(d / 3) for d in ea.size])) / 256.
lons = np.linspace(-180, 180, ea.shape[1]) * np.pi / 180
lats = np.linspace(-90, 90, ea.shape[0])[::-1] * np.pi / 180
scale = 0.8
x = scale * R * np.outer(np.cos(lons), np.cos(lats)).T
y = scale * R * np.outer(np.sin(lons), np.cos(lats)).T
z = scale * R * np.outer(np.ones(np.size(lons)), np.sin(lats)).T
Earth = ax.plot_surface(x, y, z, rstride=3, cstride=3, facecolors=ea, visible=True)

# Earth on/off button

ax_checkbox = plt.axes([0.4, 0.74, 0.1, 0.1])


def Earth_alpha(label):
    Earth.set_visible(not Earth.get_visible())
    plt.draw


check = CheckButtons(ax_checkbox, ["Earth"], [1])
check.on_clicked(Earth_alpha)
# Checks:

first_space_velocity = np.sqrt((G * M) / (R + height))
second_space_velocity = first_space_velocity * sqrt(2)
# Fall on the Earth?
if satellite_velocity >= first_space_velocity:
    p1 = 'The satellite doesn\'t fall'
else:
    p1 = 'The satellite falls!!'

satellite_dist_from_Earth = np.sqrt(trajectory_corr1[:] ** 2 + trajectory_corr2[:] ** 2 + trajectory_corr3[:] ** 2) - R
# Is it in Earth's atmosphere?
count = 0
for i in range(len(satellite_dist_from_Earth)):
    if satellite_dist_from_Earth[i] <= 10000000:
        count += 0
    else:
        count += 1
if count == 0:
    p2 = "The satellite is in the Earth's atmosphere"
else:
    p2 = "The satellite is NOT in the Earth's atmosphere"
# Is it escape from Earth's orbit?
if satellite_velocity >= second_space_velocity:
    p3 = 'The satellite leaves the Earth\'s orbit'
else:
    p3 = 'The satellite doesn\'t leave the Earth\'s orbit'

plt.figtext(0.4, 0.65, f"Checks: \n"
                       f"1 : {p1} \n"
                       f"2: {p2} \n"
                       f"3: {p3}")
# Energy

ax0 = fig.add_subplot(3, 3, 1)
ax1 = fig.add_subplot(3, 3, 4)
ax2 = fig.add_subplot(3, 3, 7)
ax0.plot(tspan, potential_energy)
ax0.set_title('Potential')
ax1.plot(tspan, kinetic_energy)
ax1.set_title('Kinetic')
ax2.plot(tspan, total_energy)
ax2.set_title('Total')
ax2.set_ylim(-3e13, 3e13)

plt.show()
