import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
import time

# Crear entorno
env = gym.make("LunarLander-v3", render_mode=None)

# Hiperparámetros
alpha = 0.1
gamma = 0.99
epsilon = 0.1
episodes = 2000

# Discretización de estados
bins = 10

# Estado -> x,y,vx,vy,angle,w,contactopata1,contactopata2
state_bins = [
    np.linspace(-1.5, 1.5, bins),
    np.linspace(-1.5, 1.5, bins),
    np.linspace(-2, 2, bins),
    np.linspace(-2, 2, bins),
    np.linspace(-3.14, 3.14, bins),
    np.linspace(-5, 5, bins),
    np.array([0, 1]),
    np.array([0, 1])
]

def discretize(state):
    disc = []
    for i in range(len(state)):
        if i < 6:
            disc.append(int(np.digitize(state[i], state_bins[i])))
        else:
            disc.append(int(state[i]))
    return tuple(disc)

# Q-table (x,y,vx,vy,angle,w) (patas) (acciones, 4)
Q = np.zeros([bins+1] * 6 + [2, 2] + [env.action_space.n])

# Política de acción
def epsilon_greedy(state):
    if random.uniform(0,1) < epsilon:
        return env.action_space.sample()
    return np.argmax(Q[state])

# Entrenamiento
rewards = []

for ep in range(episodes):
    state, _ = env.reset()
    state = discretize(state)
    total = 0

    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = epsilon_greedy(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = discretize(next_state)

        best = np.argmax(Q[next_state])

        Q[state][action] += alpha * (reward + gamma * Q[next_state][best] - Q[state][action])

        state = next_state
        total += reward

    rewards.append(total)

    if ep % 100 == 0:
        print(f"Episodio {ep} - Recompensa: {total}")

# Guardar modelo
with open("lunarlander_qtable.pkl", "wb") as f:
    pickle.dump(Q, f)

print("\nModelo guardado como lunarlander_qtable.pkl\n")

# Gráfico de recompensas
plt.plot(rewards)
plt.xlabel("Episodios")
plt.ylabel("Recompensa")
plt.title("Aprendizaje con Q-Learning - LunarLander")
plt.show()

env.close()
