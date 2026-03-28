import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def ornstein_uhlenbeck_md(
    n_cells=30,
    dim=2,
    theta=1.0,
    sigma=0.5,
    mu=None,
    T=15.0,
    dt=0.02
):
    n_steps = int(T / dt)

    if mu is None:
        mu = np.zeros(dim)

    X = np.random.randn(n_cells, dim)
    trajectories = np.zeros((n_steps, n_cells, dim))
    trajectories[0] = X

    for t in range(1, n_steps):
        dW = np.sqrt(dt) * np.random.randn(n_cells, dim)
        X = X + theta * (mu - X) * dt + sigma * dW
        trajectories[t] = X

    return trajectories


# Parametri
n_cells = 20
mu = np.array([0.0, 0.0])
traj = ornstein_uhlenbeck_md(
    n_cells=n_cells,
    dim=2,
    theta=0.9,
    sigma=0.6,
    mu=mu,
    T=20,
    dt=0.02
)

n_steps = traj.shape[0]

# Setup figura
fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_aspect("equal")
ax.set_title("OU multidimensionale con traiettorie")
ax.set_xlabel("x")
ax.set_ylabel("y")

# Centro attrattivo
ax.scatter(mu[0], mu[1], c='red', s=120, label='Centro tumore')

# Scatter cellule
scat = ax.scatter(traj[0,:,0], traj[0,:,1], c='blue', s=40)

# Linee traiettorie complete (sfondo leggero)
full_lines = []
for i in range(n_cells):
    line, = ax.plot([], [], lw=0.5, alpha=0.3)
    full_lines.append(line)

# Linee scia recente
trail_length = 40
trail_lines = []
for i in range(n_cells):
    line, = ax.plot([], [], lw=1.5)
    trail_lines.append(line)

def update(frame):
    # aggiorna punti
    scat.set_offsets(traj[frame])

    for i in range(n_cells):
        # traiettoria completa
        full_lines[i].set_data(
            traj[:frame, i, 0],
            traj[:frame, i, 1]
        )

        # scia recente
        start = max(0, frame - trail_length)
        trail_lines[i].set_data(
            traj[start:frame, i, 0],
            traj[start:frame, i, 1]
        )

    return [scat] + full_lines + trail_lines


anim = FuncAnimation(
    fig,
    update,
    frames=n_steps,
    interval=30,
    blit=True
)

plt.legend()
plt.show()
