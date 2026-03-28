import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# =========================
# PARAMETRI INIZIALI
# =========================
dt = 0.05
T = 30
N = int(T/dt)

# INPUT: numero di coscienze
num_agents = int(input("Inserisci il numero di coscienze da simulare: "))

# Equilibri individuali casuali (O*) tra 0 e 5
np.random.seed(42)
O_individual = np.random.rand(num_agents, 4) * 5

# Tasso di ritorno all'equilibrio individuale
Lambda = np.diag([0.3]*4)

# Matrice di interazioni casuale (senza auto-interazione)
C = np.random.rand(num_agents, num_agents) * 0.3
np.fill_diagonal(C, 0)

# Rumore casuale
sigma = 0.25

# =========================
# INIZIALIZZAZIONE TRAIETTORIE
# =========================
X = np.zeros((num_agents, N, 4))
for i in range(num_agents):
    X[i,0,:] = O_individual[i] + np.random.randn(4)*0.2

# Evoluzione temporale
for t in range(N-1):
    for i in range(num_agents):
        interaction = np.zeros(4)
        for j in range(num_agents):
            interaction += C[i,j] * (X[j,t,:] - X[i,t,:])
        noise = sigma * np.random.randn(4)
        dX = (-Lambda @ (X[i,t,:] - O_individual[i])
              + interaction
              + noise)
        X[i,t+1,:] = X[i,t,:] + dt * dX

# =========================
# VISUALIZZAZIONE
# =========================
fig = plt.figure(figsize=(12,9))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel("Conoscenza k")
ax.set_ylabel("Scelta s")
ax.set_zlabel("Volontà w")
ax.set_title(f"{num_agents} Coscienze Interagenti con Tempo Esplicito (Diagramma 4D)")

# Genera colori distinti per le coscienze
if num_agents <= 20:
    colors = plt.cm.tab20.colors
else:
    colors = plt.cm.viridis(np.linspace(0,1,num_agents))

lines = []
scatters = []
for i in range(num_agents):
    line, = ax.plot([], [], [], lw=2, color=colors[i % len(colors)])
    lines.append(line)
    scatter = ax.scatter(O_individual[i,1], O_individual[i,2], O_individual[i,3],
                         color=colors[i % len(colors)], marker="*", s=80)
    scatters.append(scatter)

# Punto equilibrio collettivo
Ostar_line, = ax.plot([], [], [], 'ro', markersize=10)

# Indicatore del tempo percepito medio
time_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)

def update(frame):
    # Tempo percepito medio
    t_c_mean = np.mean(X[:,frame,0])
    time_text.set_text(f"Tempo percepito medio t_c = {t_c_mean:.2f}")

    for i in range(num_agents):
        k = X[i,:frame,1]
        s = X[i,:frame,2]
        w = X[i,:frame,3]
        lines[i].set_data(k, s)
        lines[i].set_3d_properties(w)

        # Manteniamo il colore distintivo per ogni coscienza
        lines[i].set_color(colors[i % len(colors)])

    # Equilibrio collettivo
    O_star = np.mean(X[:,frame,:], axis=0)
    Ostar_line.set_data([O_star[1]], [O_star[2]])
    Ostar_line.set_3d_properties([O_star[3]])

    return lines + [Ostar_line, time_text]

ani = FuncAnimation(fig, update, frames=N, interval=40)
plt.show()
