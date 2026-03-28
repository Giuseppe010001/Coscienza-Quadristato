import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parametri biologici
r = 0.8        # forza di ritorno verso K
K = 100        # capacità portante
sigma = 15     # rumore ambientale

# Simulazione
T = 50
dt = 0.05
N = int(T/dt)

time = np.linspace(0, T, N)
X = np.zeros(N)
X[0] = 40  # popolazione iniziale

# Euler-Maruyama
for i in range(N-1):
    dW = np.sqrt(dt) * np.random.randn()
    X[i+1] = X[i] + r*(K - X[i])*dt + sigma*dW
    
    # Evita popolazioni negative
    X[i+1] = max(X[i+1], 0)

# --- Animazione ---
fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2, label="Popolazione")
ax.axhline(K, linestyle="--", color="red", label="Capacità portante K")

ax.set_xlim(0, T)
ax.set_ylim(0, max(X)*1.1)
ax.set_xlabel("Tempo")
ax.set_ylabel("Dimensione popolazione")
ax.legend()

def update(frame):
    line.set_data(time[:frame], X[:frame])
    return line,

ani = FuncAnimation(fig, update, frames=N, interval=20)
plt.show()
