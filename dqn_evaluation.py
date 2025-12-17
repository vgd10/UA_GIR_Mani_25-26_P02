import gymnasium as gym
import numpy as np
import time
import os
import sys
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


############################
# Parámetros editables
############################
MODEL_PATH = "src/trained_model.pth"  # Ruta del modelo DQN guardado
ENV_NAME = "LunarLander-v3"                     # Nombre del entorno Gymnasium
EPISODES = 10                                     # Número de episodios de evaluación
RENDER = True                                     # Mostrar render durante evaluación
DEVICE_STR = None                                 # Forzar dispositivo: "cpu" o "cuda" (None para auto)


def evaluate_dqn():
    # Implementación interna de DQNAgent (sin dependencias externas)
    class QNetwork(nn.Module):
        def __init__(self, state_size, action_size, hidden_size=128):
            super().__init__()
            self.fc1 = nn.Linear(state_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, action_size)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    class DQNAgent:
        def __init__(self, state_size, action_size, device="cpu", hidden_size=128):
            self.device = torch.device(device)
            self.state_size = state_size
            self.action_size = action_size
            self.q_local = QNetwork(state_size, action_size, hidden_size=hidden_size).to(self.device)
            self.q_target = QNetwork(state_size, action_size, hidden_size=hidden_size).to(self.device)
            self.q_target.load_state_dict(self.q_local.state_dict())
            self.epsilon = 0.01

        def act(self, state, greedy=True):
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            self.q_local.eval()
            with torch.no_grad():
                qvals = self.q_local(s)
            self.q_local.train()
            if greedy or random.random() > self.epsilon:
                return int(torch.argmax(qvals, dim=1).item())
            return int(random.randrange(self.action_size))

        def load(self, path):
            # Cargamos pesos únicamente (más seguro con modelos externos)
            self.q_local.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
            self.q_target.load_state_dict(self.q_local.state_dict())
            self.epsilon = 0.01

    device = torch.device(DEVICE_STR or ("cuda" if torch.cuda.is_available() else "cpu"))
    render_mode = "human" if RENDER else None
    env = gym.make(ENV_NAME, render_mode=render_mode)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size, device=device)
    agent.load(MODEL_PATH)
    agent.epsilon = 0.01  # greedy casi puro para evaluar

    scores = []
    for ep in range(1, EPISODES + 1):
        state, _ = env.reset()
        total = 0.0
        terminated = truncated = False
        print(f"\n--- Episodio {ep} (DQN) ---")
        while not (terminated or truncated):
            action = agent.act(state, greedy=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            total += reward
            state = next_state
            if RENDER:
                time.sleep(0.01)
        print(f"Recompensa total: {total:.1f}")
        scores.append(total)

    env.close()
    print(f"\nMedia de recompensas (DQN) en {EPISODES} episodios: {np.mean(scores):.1f}")
    return scores


def main():
    # Ejecuta evaluación con los parámetros definidos arriba
    evaluate_dqn()


if __name__ == "__main__":
    main()
