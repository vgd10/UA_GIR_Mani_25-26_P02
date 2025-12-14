import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
import time
import argparse

# Crear entorno
env = gym.make("LunarLander-v3", render_mode=None)

# Hiperparámetros
gamma = 0.99

# Parámetros de entrenamiento (con decaimiento)
epsilon_start = 1.0
epsilon_min = 0.05
epsilon_decay = 0.995  # multiplicativo por episodio

alpha_start = 0.5
alpha_min = 0.05
alpha_decay = 0.995  # multiplicativo por episodio

# Parámetros Softmax (Boltzmann)
temperature_start = 1.0
temperature_min = 0.1
temperature_decay = 0.995

max_episodes = 20000  # límite superior; se usa parada temprana
target_mean_reward = 200  # objetivo estándar para LunarLander
window = 100  # tamaño de ventana para media móvil

# Discretización de estados
bins = 12

# Estado -> x,y,vx,vy,angle,w,contactopata1,contactopata2
state_bins = [
    np.linspace(-1.2, 1.2, bins),
    np.linspace(-1.2, 1.2, bins),
    np.linspace(-2.0, 2.0, bins),
    np.linspace(-2.0, 2.0, bins),
    np.linspace(-3.14, 3.14, bins),
    np.linspace(-5.0, 5.0, bins),
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
def epsilon_greedy(state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    return int(np.argmax(Q[state]))


def softmax_action(state, temperature):
    q_values = Q[state]
    # Evitar overflow: restar el max
    z = (q_values - np.max(q_values)) / max(temperature, 1e-6)
    probs = np.exp(z)
    probs /= np.sum(probs)
    return int(np.random.choice(len(q_values), p=probs))

# Entrenamiento
def train(policy="epsilon-greedy"):
    random.seed(42)
    np.random.seed(42)

    rewards = []
    best_mean = -np.inf
    best_Q = None

    epsilon = epsilon_start
    alpha = alpha_start
    temperature = temperature_start

    start_time = time.time()

    for ep in range(1, max_episodes + 1):
        state, _ = env.reset()
        state = discretize(state)
        total = 0.0

        terminated = False
        truncated = False

        while not (terminated or truncated):
            # Acción según política solicitada
            if policy == "epsilon-greedy":
                action = epsilon_greedy(state, epsilon)
            elif policy == "softmax":
                action = softmax_action(state, temperature)
            else:
                raise ValueError("Política no soportada. Usa 'epsilon-greedy' o 'softmax'.")
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = discretize(next_state)

            # Mejor acción en siguiente estado
            best_next = int(np.argmax(Q[next_state]))

            # Actualización Q-Learning
            td_target = reward + gamma * Q[next_state][best_next]
            td_error = td_target - Q[state][action]
            Q[state][action] += alpha * td_error

            state = next_state
            total += reward

        rewards.append(total)

        # Decaimiento de parámetros
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        alpha = max(alpha_min, alpha * alpha_decay)
        temperature = max(temperature_min, temperature * temperature_decay)

        # Estadísticas y parada temprana por media móvil
        if len(rewards) >= window:
            mean_recent = np.mean(rewards[-window:])
            if mean_recent > best_mean:
                best_mean = mean_recent
                best_Q = Q.copy()
            if mean_recent >= target_mean_reward:
                print(f"Parada temprana en episodio {ep}: media {mean_recent:.2f}")
                break

        if ep % 100 == 0:
            mean_recent = np.mean(rewards[-min(len(rewards), window):])
            elapsed = time.time() - start_time
            print(
                f"Episodio {ep} | Recompensa {total:.1f} | Media({min(len(rewards), window)}) "
                f"{mean_recent:.1f} | eps {epsilon:.3f} | temp {temperature:.3f} | alpha {alpha:.3f} | t {elapsed/60:.1f}m"
            )

    # Guardar el mejor modelo disponible
    to_save = best_Q if best_Q is not None else Q
    with open("lunarlander_qtable.pkl", "wb") as f:
        pickle.dump(to_save, f)
    print("\nModelo guardado como lunarlander_qtable.pkl\n")

    return rewards


def plot_rewards(rewards):
    plt.figure(figsize=(8, 4))
    plt.plot(rewards, label="Recompensa por episodio", alpha=0.7)
    if len(rewards) > window:
        mv = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, window-1+len(mv)), mv, label=f"Media móvil ({window})")
    plt.xlabel("Episodios")
    plt.ylabel("Recompensa")
    plt.title("Aprendizaje con Q-Learning - LunarLander")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Entrenar Q-Learning LunarLander")
    parser.add_argument("--policy", choices=["epsilon-greedy", "softmax"], default="epsilon-greedy", help="Política de selección de acciones")
    parser.add_argument("--episodes", type=int, default=None, help="Sobrescribir max_episodes si se indica")
    args = parser.parse_args()

    global max_episodes
    if args.episodes is not None:
        max_episodes = int(args.episodes)

    rewards = train(policy=args.policy)
    plot_rewards(rewards)
    env.close()


if __name__ == "__main__":
    main()
