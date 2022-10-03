import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import animation
import numpy as np
from scipy.special import jv, jn_zeros
from tqdm.auto import tqdm
from functools import lru_cache

RADIUS = 1
SPEED_OF_SOUND = 0.8
## jn_zeros(m,n), m is the order of bessel's function and n is the number of roots 
BESSEL_ROOTS = [jn_zeros(m, 10) for m in range(10)]
FPS = 25 ### here you specify the FPS 
TIME_PER_MODE = 3 #time the animation will spend on each mode
## Here are the modes to be animated, e.g mode = (1,0) is the mode where the bessel's function order is 1 and the first root 
## is being returned
## in case you want a specific mode to be animated, input the entry for that mode, in case you want more modes to be animated one
## after the other on the same animation, then put all the entries in order
MODES = (
    (0, 0),
    (1, 0),
    (2, 0),
    (0, 1),
    (3, 0),
    (1, 1),
    (4, 0),
    (2, 1),
    (0, 2),
    (3, 1),
    )

FRAMES = len(MODES) * TIME_PER_MODE * FPS ## number of frames 
## NOTE: the more frames are to be animated, the longer will the code take to run.

@lru_cache()
def lambda_mn(m, n, radius):
    return BESSEL_ROOTS[m][n - 1] / radius


@lru_cache()
def get_vmin_vmax(m, n):
    vmax = np.max(jv(m, np.linspace(0, BESSEL_ROOTS[m][n], 100)))
    return -vmax, vmax

## this is the circular membrane function 
def circular_membrane(r, theta, t, m, n, radius, speed_of_sound):
    l = lambda_mn(m, n, radius)

    T = np.sin(speed_of_sound * l * t)
    R = jv(m, l * r)
    Theta = np.cos(m * theta)
    ## the upper part of the function can be used to animate specific modes
    Theta_zero= np.cos(m*3.1415926/2)*0.75*jv(m,l*0.75)
    ## this part is needed to animate a summation of the modes, comment it out in case you are animating specific modes

    return R * T * Theta*Theta_zero ### if you need specific modes, comment out Theta_zero

### all the following code makes a 3D representation of a certain frame
## this part will be be fed to the animation object, this will serve as a framework and
## the animation object will make changes to this framework in order to make the animation
r = np.linspace(0, RADIUS, 100)
theta = np.linspace(0, 2 * np.pi, 100)


r, theta = np.meshgrid(r, theta)
x = np.cos(theta) * r
y = np.sin(theta) * r
z = circular_membrane(r, theta, 0, 0, 0, RADIUS, SPEED_OF_SOUND) 
for a in range(1,len(MODES)):
    	z += circular_membrane(r, theta, t, MODES[a][0], MODES[a][1], RADIUS, SPEED_OF_SOUND)
vmin, vmax = get_vmin_vmax(3, 1)

fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
ax = fig.add_axes([0, 0, 1, 1], projection='3d')
ax.set_axis_off()
### this function plots the first image 
plot = ax.plot_surface(
    x,
    y,
    z,
    linewidth=0,
    cmap='Spectral',
    vmin=vmin,
    vmax=vmax,
    rcount=100,
    ccount=100,
)
omega = SPEED_OF_SOUND * lambda_mn(0, 0, RADIUS)
## this text shows the angular frequency of specific modes, not so useful when making an animation that adds all the modes 
text = ax.text2D(
    0.5, 0.95,
    f'Circular membrane, m = {0}, n = {0}, ω={omega:.2f}',
    size=36, weight='bold', family='Fira Sans',
    va='top', ha='center',
    transform=ax.transAxes,
)


def init():
    pass

## this function adds the first ten modes 
def update(i, bar=None):
    global plot

    if bar is not None:
        bar.update()

    t = i / FPS
    m, n = MODES[9] ### for this case when the modes are added, the m,n can take any number
    ### following we have the addittion of the first ten modes, in case you need only one specific mode to be plotted 
    ## replace all the block that starts with "z=", with z = circular_membrane(r, theta, t, m, n, RADIUS, SPEED_OF_SOUND)

    z = circular_membrane(r, theta, t, 0, 0, RADIUS, SPEED_OF_SOUND) 
    for a in range(1,len(MODES)):
    	z += circular_membrane(r, theta, t, MODES[a][0], MODES[a][1], RADIUS, SPEED_OF_SOUND)
    vmin, vmax = get_vmin_vmax(m, n)
    plot.remove()
    plot = ax.plot_surface(
        x,
        y,
        z,
        linewidth=0,
        cmap='Spectral',
        vmin=vmin,
        vmax=vmax,
        rcount=100,
        ccount=100,
    )
    ax.set_zlim(-1.1, 1.1)
    ax.set_xlim(-0.75, 0.75)
    ax.set_ylim(-0.75, 0.75)
    omega = SPEED_OF_SOUND * lambda_mn(m, n, RADIUS)
    text.set_text(f'Circular membrane, m = {4}, n = {5}, ω={omega:.2f}') ## random numbers for now, when you do specific 
    ## modes you should put Circular membrane, m = {0}, n = {0}


bar = tqdm(total=FRAMES)
ani = FuncAnimation(fig, update, init_func=init, frames=FRAMES, interval=1000/FPS, repeat=False, fargs=(bar, ))
writergif = animation.PillowWriter(fps=20)
ani.save(
    ## the name of the file where the animation will be saved
    f'membrane_newtestagain.gif',
    writer=writergif
)