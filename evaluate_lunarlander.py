import gymnasium as gym
import numpy as np
import pickle
import time
import argparse

# Cargar el modelo
def load_qtable(path="lunarlander_qtable.pkl"):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        raise SystemExit(f"No se encontró el modelo en '{path}'. Entrena primero el agente.")

bins = 12
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

# Entorno con render
def make_env(render=False):
    mode = "human" if render else None
    return gym.make("LunarLander-v3", render_mode=mode)

def epsilon_greedy(Q, state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(Q[state].shape[0])
    return int(np.argmax(Q[state]))


def softmax_action(Q, state, temperature):
    q_values = Q[state]
    z = (q_values - np.max(q_values)) / max(temperature, 1e-6)
    probs = np.exp(z)
    probs /= np.sum(probs)
    return int(np.random.choice(len(q_values), p=probs))


def evaluate(Q, episodes=5, render=True, sleep=0.02, policy="greedy", epsilon=0.0, temperature=1.0):
    env = make_env(render=render)
    scores = []
    for ep in range(episodes):
        state, _ = env.reset()
        state = discretize(state)
        total_reward = 0.0
        terminated = truncated = False

        print(f"\n--- Episodio {ep+1} ---")

        while not (terminated or truncated):
            if policy == "greedy":
                action = int(np.argmax(Q[state]))
            elif policy == "epsilon-greedy":
                action = epsilon_greedy(Q, state, epsilon)
            elif policy == "softmax":
                action = softmax_action(Q, state, temperature)
            else:
                raise ValueError("Política no soportada en evaluación.")
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = discretize(next_state)

            total_reward += reward
            state = next_state
            if render and sleep:
                time.sleep(sleep)   # para que el render se vea bien

        print(f"Recompensa total: {total_reward:.1f}")
        scores.append(total_reward)

    env.close()
    print(f"\nMedia de recompensas en {episodes} episodios: {np.mean(scores):.1f}")
    return scores


def main():
    parser = argparse.ArgumentParser(description="Evaluar Q-Learning LunarLander")
    parser.add_argument("--episodes", type=int, default=5, help="Número de episodios a evaluar")
    parser.add_argument("--no-render", action="store_true", help="Desactivar renderizado")
    parser.add_argument("--model", type=str, default="lunarlander_qtable.pkl", help="Ruta al modelo Q-table")
    parser.add_argument("--policy", choices=["greedy", "epsilon-greedy", "softmax"], default="greedy", help="Política de acción para evaluar")
    parser.add_argument("--epsilon", type=float, default=0.05, help="Epsilon para epsilon-greedy en evaluación")
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperatura para softmax en evaluación")
    args = parser.parse_args()

    Q = load_qtable(args.model)
    evaluate(Q, episodes=args.episodes, render=not args.no_render, policy=args.policy, epsilon=args.epsilon, temperature=args.temperature)


if __name__ == "__main__":
    main()
