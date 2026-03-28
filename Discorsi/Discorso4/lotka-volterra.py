import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parametri modello Lotka-Volterra
alpha = 1.0   # crescita prede
beta = 0.1    # predazione
delta = 0.075 # crescita predatori
gamma = 1.5   # morte predatori

# Intensità rumore (tipo Langevin)
sigma_prey = 0.2
sigma_pred = 0.2

# Simulazione
T = 50
dt = 0.02
N = int(T/dt)

time = np.linspace(0, T, N)
prey = np.zeros(N)
pred = np.zeros(N)

# Condizioni iniziali
prey[0] = 10
pred[0] = 5

# Simulazione SDE (Euler-Maruyama)
for i in range(N-1):
    dW1 = np.sqrt(dt) * np.random.randn()
    dW2 = np.sqrt(dt) * np.random.randn()

    prey[i+1] = prey[i] + (
        alpha*prey[i] - beta*prey[i]*pred[i]
    )*dt + sigma_prey*prey[i]*dW1

    pred[i+1] = pred[i] + (
        delta*prey[i]*pred[i] - gamma*pred[i]
    )*dt + sigma_pred*pred[i]*dW2

    # Evita valori negativi
    prey[i+1] = max(prey[i+1], 0)
    pred[i+1] = max(pred[i+1], 0)

# --- ANIMAZIONE ---

fig, ax = plt.subplots()
line1, = ax.plot([], [], label="Prede")
line2, = ax.plot([], [], label="Predatori")

ax.set_xlim(0, T)
ax.set_ylim(0, max(prey.max(), pred.max())*1.1)
ax.set_xlabel("Tempo")
ax.set_ylabel("Popolazione")
ax.legend()

def update(frame):
    line1.set_data(time[:frame], prey[:frame])
    line2.set_data(time[:frame], pred[:frame])
    return line1, line2

ani = FuncAnimation(fig, update, frames=N, interval=10)

plt.show()
