import gymnasium as gym
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from gymnasium import RewardWrapper

import time
import pickle

import os
from copy import copy

# Bounds to ask for parameters
USERMODE_BOUND = [0,5,0]

ALPHA_BOUND = [0.3,0.9,1]
SPACE_BOUND = [1,10,0]


GAMMA_BOUND = [0.8,0.95,1]

POLICY_BOUND = [0,1,0]
ALGORITHM_BOUND = [1,2,0]
ENVIRONMENT_BOUND = [0,1,0]

VPOLICY_BOUND = [0.1,0.999,1]
MINVPOLICY_BOUND = [0.01,0.001,1]
DECAYVPOLICY_BOUND = [0.9,0.999,1]

EPISODESENV0_BOUND = [20,1000,0]
EPISODESENV1_BOUND = [500,5000,0]

WINDOW_BOUND = [10,200,0]

def inputBoundedParameter(text,bound):
    datatype = "int"
    if bound[2] > 0:
        datatype = "float"

    value = input(text+f" [MIN: {bound[0]}, MAX: {bound[1]}, TYPE: {datatype}]: ")
    if bound[2] < 1:
        return min(max(int(value),bound[0]),bound[1])
    return min(max(float(value),bound[0]),bound[1])


def inputParameter(text,bounds):
    values = []
    v_parameters = []

    mode = inputBoundedParameter(text+" input mode",USERMODE_BOUND)

    if len(bounds) > 0:
        if mode < 1:
            values.append(inputBoundedParameter(text,bounds[0]))
        elif mode < 2 and len(bounds) > 2:
            mod = ["maximum","minimum","bins"]
            for i in range(3):
                v_parameters.append(inputBoundedParameter(text+" ("+mod[i]+")",bounds[i]))
        else:
            for j in range(mode):
                values.append(inputBoundedParameter(text+f" ({j})",bounds[0]))
    
    if len(v_parameters) > 0:
        if len(v_parameters) > 2 and v_parameters[0] > v_parameters[1]:
            values = np.linspace(v_parameters[0],v_parameters[1],v_parameters[2])
        else:
            values.append(v_parameters[0])

    return values,mode


# Intrinsic RL parameters
#usermode = inputBoundedParameter("Set user mode (Manual: 0 // Recover: 1 // Automatic: 2)",USERMODE_BOUND)

print("REMAINDER: inputs mode dictates on which way parameters are expected")
print("0 == Only 1 value expected // 1 == 3 values expected: A maximum, minimum and bins to space them)  //  >1 == A costum amount of values equal to the mode\n")

alphas,_ = inputParameter("Set learning rate",[ALPHA_BOUND,ALPHA_BOUND,SPACE_BOUND])        # Learning rate parameters
gammas,_ = inputParameter("Set the discount reward",[GAMMA_BOUND,GAMMA_BOUND,SPACE_BOUND])  # Discount parameter

environment = inputBoundedParameter("Set environment to use (FrozenLake-v1_4x4_noslip: 0 // LunarLander-v3: 1)",ENVIRONMENT_BOUND)
algorithm = inputBoundedParameter("Set algorithm to train (Q-learning: 1 // SARSA: 2)",ALGORITHM_BOUND)
policy = inputBoundedParameter("Set policy to use (Epsilon-greedy: 0 // Softmax: 1)",POLICY_BOUND)

if algorithm < 2:
    s1 = "starting"
else:
    s1 = "permanent"

if policy < 1:
    s2 = "epsilon"
else:
    s2 = "temperature"

v_policies = []

v_policies.append([])
v_policies[-1].append(inputBoundedParameter(f"Set a {s1} {s2}",VPOLICY_BOUND))
if algorithm < 2:
    v_policies[-1].append(inputBoundedParameter(f"Set a minimum {s2}",MINVPOLICY_BOUND))
    v_policies[-1].append(inputBoundedParameter(f"Set a decay {s2}",DECAYVPOLICY_BOUND))

# Total episodes of the training
if environment < 1:
    envname = "FrozenLake-v1"
    max_episodes,_ = inputParameter("Set max episodes",[EPISODESENV0_BOUND,EPISODESENV0_BOUND,SPACE_BOUND])
else:
    envname = "LunarLander-v3"
    max_episodes,_ = inputParameter("Set max episodes",[EPISODESENV1_BOUND,EPISODESENV1_BOUND,SPACE_BOUND])
    
window = inputBoundedParameter("Set a window size (used to calculate mean of accumulated rewards)",WINDOW_BOUND) 

if policy < 1:
    epsilon = v_policies[-1][0]
    if algorithm < 2:
        min_epsilon = v_policies[-1][1]
        decay_epsilon = v_policies[-1][2]
else:
    temperature = v_policies[-1][0]
    if algorithm < 2:
        min_temperature = v_policies[-1][1]
        decay_temperature = v_policies[-1][2]

# Action policies

def epsilon_greedy(state,epsilon):
    if random.uniform(0,1) < epsilon:
        return env.action_space.sample()
    return np.argmax(Q[state])

def softmax(state, temperature):
    q_values = Q[state]
    # Evitar overflow: restar el max
    z = (q_values - np.max(q_values)) / max(temperature, 1e-6)
    probs = np.exp(z)
    probs /= np.sum(probs)
    return int(np.random.choice(len(q_values), p=probs))



def select_action(state,policy,parameter_policy):
    if policy < 1:
        return epsilon_greedy(state,parameter_policy)
    return softmax(state,parameter_policy)



# =========== LUNAR-LANDER ONLY ===========

bins = 12 # Amount of subdivisions space is divided

# State discretization values -> x,y,vx,vy,angle,w,contactleg1,contactleg2
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


# State discretization function
def discretize(state):
    disc = []
    for i in range(len(state)):
        if i < 6:
            disc.append(int(np.digitize(state[i], state_bins[i])))
        else:
            disc.append(int(state[i]))
    return tuple(disc)

target_mean_reward = 200  # Standard objective

# ================ Lunar-Lander ================

# Storing variables

# Action, reward and next state matrix [i][j] (i -> Episode // j -> Step on that episode)
ev_act = []
ev_reward = []
ev_state = []

# Epsilon and accumulated reward matrix (length = episodes)
ev_epsilon = []
acc_reward = []

envtotal = 1
if environment > 1:
    envtotal = 2


for envnumber in range(envtotal):

    # Environment initialization (TODO: Need some Tweaking to support varied environments)
    if envnumber < 1:
        if environment < 1:
            env = gym.make(envname,desc=None,map_name="4x4",is_slippery=False,reward_schedule=(10, -1, -1),render_mode=None)
        else:
            env = gym.make(envname,render_mode=None)
    else:
        env = gym.make(envname,render_mode=None)
    
    for alpha in alphas:
        for gamma in gammas:
            for v_policyI in v_policies:
                for episodes in max_episodes:

                    v_policy = copy(v_policyI)

                    best_mean = -np.inf

                    # Q-table first initialization
                    if environment < 1:
                        Q = (-1)*np.ones([env.observation_space.n, env.action_space.n])
                    else:
                        Q = np.zeros([bins+1] * 6 + [2, 2] + [env.action_space.n])

                    best_Q = None

                    if environment < 1:
                        max_steps = 20  # Maximum steps allowed before terminating episode no matter what

                    mean_recent = -200

                    # Flag and episode number when objective is reached for first time
                    objective_reached = False
                    ep_ob_reached = -1

                    ev_epsilon.append([])
                    acc_reward.append([])
                    if environment < 1:
                        ev_act.append([])
                        ev_reward.append([])
                        ev_state.append([])

                    start_time = time.time()

                    # print(alpha,gamma,v_policy,episodes)

                    # Training phase
                    for episode in range(1,episodes+1):
                        # break

                        # Environment reset
                        state, _ =env.reset()

                        # Prepare store data for this episode (epsilon only used for Q-learning + ep_greedy)
                        if environment < 1:
                            ev_state[-1].append([])
                            ev_act[-1].append([])
                            ev_reward[-1].append([])
                            steps = 0
                        else:
                            state = discretize(state)

                        ev_epsilon[-1].append(epsilon)
                        acc_reward[-1].append(0.0)

                        # Set termination flags
                        terminated = False
                        truncated = False  # NOT USED
    
                        if algorithm >= 2:
                            act = select_action(state,policy,v_policy)
    
    
                        # Episode simulation
                        while not (terminated):


                            # Action calculation (not SARSA)
                            if algorithm < 2:
                                act = select_action(state,policy,v_policy[0])


                            # Step execution
                            next_state,reward,terminated,truncated,_ = env.step(act)
        
                            if environment > 1:
                                next_state = discretize(next_state)
        

                            # Next best action calculation (Motecarlo and Q-learning only)
                            if algorithm < 2:
                                next_act = np.argmax(Q[next_state])
                            # Next action calculation (SARSA)
                            else:
                                next_act = select_action(state,policy,v_policy[0])


                            # Q-table and state update
                            Q[state][act] += alpha * (reward + gamma * Q[next_state][next_act] - Q[state][act])
                            state = next_state
        
                            # Save all gathered step info
                            if environment < 1:
                                ev_act [-1][episode-1].append(act)
                                ev_reward[-1][episode-1].append(reward)
                                ev_state[-1][episode-1].append(state)
                            acc_reward[-1][episode-1] += reward


                            # Start to erode epsilon value once target is reached for first time (Q-learning+ep_greedy only)
                            if mean_recent >= -150.0 and not objective_reached:
                                objective_reached = True
                                ep_ob_reached = episode
        
                            # End condition, so each episode doesn't take absurdly long
                            if environment < 1:
                                if steps >= max_steps:
                                    terminated = True
                                steps += 1

                        if objective_reached and algorithm < 2:
                            v_policy[0] = max(v_policy[1],v_policy[0]*v_policy[2])
        
                        # Estadísticas y parada temprana por media móvil
                        if len(acc_reward[-1]) >= window:
                            mean_recent = np.mean(acc_reward[-1][-window:])
                            if mean_recent > best_mean:
                                best_mean = mean_recent
                                best_Q = Q.copy()
                            if mean_recent >= target_mean_reward:
                                print(f"Parada temprana en episodio {episode}: media {mean_recent:.2f}")
                                break

                        if episode % window == 0:
                            mean_recent = np.mean(acc_reward[-1][-min(len(acc_reward[-1]), window):])
                            elapsed = time.time() - start_time
                            print(
                                f"Episode {episode} | Reward {acc_reward[-1][episode-1]:.1f} | Mean({min(len(acc_reward[-1]), window)}) "
                                f"{mean_recent:.1f} | v_pol {v_policy[0]:.3f} | alpha {alpha:.3f} | t {elapsed/60:.1f}m"
                            )

                    # Guardar el mejor modelo disponible
                    to_save = best_Q if best_Q is not None else Q
                    if not os.path.exists("qtables"):
                        os.mkdir("qtables")

                    with open(f"qtables/{envname}_a{alpha:.3f}_g{gamma:.3f}_p{v_policy[0]:.3f}_e{episodes}_qtable.pkl", "wb") as f:
                        pickle.dump(to_save, f)
                    print(f"\nModelo guardado como {envname}_a{alpha:.3f}_g{gamma:.3f}_p{v_policy[0]:.3f}_e{episodes}_qtable.pkl\n")

                env.close() # Close environment
