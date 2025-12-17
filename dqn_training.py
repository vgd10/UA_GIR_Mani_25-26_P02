import os
import sys
import time
from collections import deque
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


############################
# Parámetros editables
############################
ENV_NAME = "LunarLander-v3"           # Nombre del entorno Gymnasium
N_EPISODES = 2000                      # Número total de episodios de entrenamiento
MAX_T = 1000                           # Máximo de pasos por episodio
RECORD_VIDEO = False                   # Grabar vídeos del entrenamiento
VIDEO_EVERY = 10                       # Frecuencia de grabación: cada N episodios
MODEL_PATH = "src/trained_model.pth"  # Ruta de guardado del modelo entrenado
TARGET_MEAN_REWARD = None              # Parada temprana: media(window) >= objetivo (None para desactivar)
WINDOW = 100                           # Tamaño de la ventana para media móvil y early stopping
SEED = 42                              # Semilla para reproducibilidad (None para no fijar)
DEVICE_STR = None                      # Forzar dispositivo: "cpu" o "cuda" (None para auto)


def train_dqn():
    # Implementación interna de DQN (sin dependencias externas)
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

    class ReplayBuffer:
        def __init__(self, buffer_size=100000, batch_size=64, device="cpu"):
            self.memory = deque(maxlen=buffer_size)
            self.batch_size = batch_size
            self.device = torch.device(device)

        def add(self, s, a, r, s2, d):
            self.memory.append((
                np.array(s, dtype=np.float32),
                int(a),
                float(r),
                np.array(s2, dtype=np.float32),
                bool(d)
            ))

        def sample(self):
            idx = np.random.choice(len(self.memory), size=self.batch_size, replace=False)
            batch = [self.memory[i] for i in idx]
            states = torch.tensor([b[0] for b in batch], dtype=torch.float32, device=self.device)
            actions = torch.tensor([b[1] for b in batch], dtype=torch.long, device=self.device).unsqueeze(1)
            rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=self.device).unsqueeze(1)
            next_states = torch.tensor([b[3] for b in batch], dtype=torch.float32, device=self.device)
            dones = torch.tensor([b[4] for b in batch], dtype=torch.float32, device=self.device).unsqueeze(1)
            return states, actions, rewards, next_states, dones

        def __len__(self):
            return len(self.memory)

    class DQNAgent:
        def __init__(
            self,
            state_size,
            action_size,
            device="cpu",
            lr=1e-3,
            gamma=0.99,
            buffer_size=100000,
            batch_size=64,
            update_every=4,
            target_update_every=1000,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.996,
            hidden_size=128,
        ):
            self.device = torch.device(device)
            self.state_size = state_size
            self.action_size = action_size
            self.gamma = gamma
            self.update_every = update_every
            self.target_update_every = target_update_every
            self.batch_size = batch_size

            self.q_local = QNetwork(state_size, action_size, hidden_size=hidden_size).to(self.device)
            self.q_target = QNetwork(state_size, action_size, hidden_size=hidden_size).to(self.device)
            self.q_target.load_state_dict(self.q_local.state_dict())
            self.optimizer = optim.Adam(self.q_local.parameters(), lr=lr)
            self.criterion = nn.MSELoss()

            self.memory = ReplayBuffer(buffer_size=buffer_size, batch_size=batch_size, device=self.device)
            self.t_step = 0
            self.learn_step = 0

            self.epsilon = epsilon_start
            self.epsilon_end = epsilon_end
            self.epsilon_decay = epsilon_decay

        def act(self, state, greedy=False):
            if isinstance(state, np.ndarray):
                s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            else:
                s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            self.q_local.eval()
            with torch.no_grad():
                qvals = self.q_local(s)
            self.q_local.train()
            if greedy or random.random() > self.epsilon:
                return int(torch.argmax(qvals, dim=1).item())
            return int(random.randrange(self.action_size))

        def step(self, s, a, r, s2, done):
            self.memory.add(s, a, r, s2, done)
            self.t_step += 1
            if self.t_step % self.update_every == 0 and len(self.memory) >= self.batch_size:
                self.learn()
                if self.learn_step % self.target_update_every == 0 and self.learn_step > 0:
                    self.q_target.load_state_dict(self.q_local.state_dict())

        def learn(self):
            states, actions, rewards, next_states, dones = self.memory.sample()
            next_actions = self.q_local(next_states).argmax(dim=1, keepdim=True)
            q_next = self.q_target(next_states).gather(1, next_actions)
            targets = rewards + (self.gamma * q_next * (1.0 - dones))
            q_expected = self.q_local(states).gather(1, actions)
            loss = self.criterion(q_expected, targets.detach())
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_local.parameters(), 1.0)
            self.optimizer.step()
            self.learn_step += 1

        def decay_epsilon(self):
            if self.epsilon > self.epsilon_end:
                self.epsilon *= self.epsilon_decay
                self.epsilon = max(self.epsilon_end, self.epsilon)

        def save(self, path):
            torch.save(self.q_local.state_dict(), path)

        def load(self, path):
            self.q_local.load_state_dict(torch.load(path, map_location=self.device))
            self.q_target.load_state_dict(self.q_local.state_dict())
            self.epsilon = 0.01

    device = torch.device(DEVICE_STR or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    if SEED is not None:
        torch.manual_seed(SEED)

    render_mode = "rgb_array" if RECORD_VIDEO else None
    env = gym.make(ENV_NAME, render_mode=render_mode)
    

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size=state_size, action_size=action_size, device=device)

    scores = []
    scores_window = deque(maxlen=WINDOW)
    start_time = time.time()

    for i_episode in range(1, N_EPISODES + 1):
        state, _ = env.reset()
        score = 0.0
        for _ in range(MAX_T):
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break

        scores.append(score)
        scores_window.append(score)
        agent.decay_epsilon()

        if i_episode % 10 == 0 or i_episode == 1:
            mean_recent = sum(scores_window) / len(scores_window)
            elapsed = (time.time() - start_time) / 60
            print(
                f"Episode {i_episode} | Score: {score:.1f} | Mean({len(scores_window)}): {mean_recent:.1f} | Epsilon: {agent.epsilon:.3f} | t {elapsed:.1f}m"
            )

        if TARGET_MEAN_REWARD is not None and len(scores_window) == WINDOW:
            mean_recent = sum(scores_window) / len(scores_window)
            if mean_recent >= TARGET_MEAN_REWARD:
                print(f"Early stopping at episode {i_episode}: mean({WINDOW})={mean_recent:.1f} >= {TARGET_MEAN_REWARD}")
                break

    env.close()
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    agent.save(MODEL_PATH)
    print(f"Training finished. Model saved to {MODEL_PATH}.")

    return scores


def main():
    # Ejecuta entrenamiento con los parámetros definidos arriba
    train_dqn()


if __name__ == "__main__":
    main()
