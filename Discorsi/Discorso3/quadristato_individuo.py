import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Parametri
dt = 0.05
T = 30
N = int(T / dt)

# Equilibrio individuale
O_individual = np.array([5.0, 2.0, 1.8, 2.1])
Lambda = np.diag([0.3, 0.3, 0.3, 0.3])
sigma = 0.25

# Stato dell'agente: (1 agente, N timesteps, 4 dimensioni)
X = np.zeros((1, N, 4))
X[0, 0, :] = O_individual + np.random.randn(4) * 0.2

# Simulazione
for t in range(N - 1):
    noise = sigma * np.random.randn(4)
    dX = -Lambda @ (X[0, t, :] - O_individual) + noise
    X[0, t + 1, :] = X[0, t, :] + dt * dX

# Plot 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel("Conoscenza k")
ax.set_ylabel("Scelta s")
ax.set_zlabel("Volontà w")
ax.set_title("Coscienza Singola")

line, = ax.plot([], [], [], lw=2, color='blue')
ax.scatter(O_individual[1], O_individual[2], O_individual[3],
           color='red', marker='*', s=200, label='O individuale')

# Indicatore del tempo percepito
time_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)

def update(frame):
    k = X[0, :frame, 1]
    s = X[0, :frame, 2]
    w = X[0, :frame, 3]
    line.set_data(k, s)
    line.set_3d_properties(w)

    # Tempo percepito della coscienza
    t_c = X[0, frame, 0]
    time_text.set_text(f"Tempo percepito t_c = {t_c:.2f}")

    return [line, time_text]

ani = FuncAnimation(fig, update, frames=N, interval=40)
plt.legend()
plt.show()
