import gymnasium as gym
import numpy as np
import pickle
import time

# Cargar el modelo
with open("lunarlander_qtable.pkl", "rb") as f:
    Q = pickle.load(f)

bins = 10
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

# Entorno con render
env = gym.make("LunarLander-v3", render_mode="human")

def evaluate(episodes=5):
    for ep in range(episodes):
        state, _ = env.reset()
        state = discretize(state)
        total_reward = 0
        terminated = truncated = False
        
        print(f"\n--- Episodio {ep+1} ---")

        while not (terminated or truncated):
            action = np.argmax(Q[state])
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = discretize(next_state)

            total_reward += reward
            state = next_state
            time.sleep(0.02)   # para que el render se vea bien

        print(f"Recompensa total: {total_reward}")

    env.close()


evaluate(5)
