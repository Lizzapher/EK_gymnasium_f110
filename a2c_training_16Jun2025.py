"""
Basic a2c Training Script for F1Tenth Racing Environment
"""

import os
import time
import yaml
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import torch as th
import warnings
import matplotlib.image as mpimg

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from f1tenth_gym.envs.integrator import RK4Integrator
import f1tenth_gym.envs.f110_env

#-----------------------------------
# User Warnings Suppressed
#-----------------------------------
warnings.filterwarnings("ignore", category=UserWarning)
print('User warnings suppressed')

#-----------------------------------
# Training Configuration
#-----------------------------------
_TEST_BATCH = "_Map_roulette_TEST"
_TOTAL_TIME_STEPS = 100
_NUM_EPISODES = 5
_SHOW_PATH = False
_MAP = 'IMS'

#-----------------------------------
# Environment Creation
#-----------------------------------
# Headless Training
def make_f1tenth_env_training():
    config = {
        "map":_MAP,
        "num_agents":1,
        "timestep":0.01,
        "integrator":"rk4",
        "control_input": ["speed", "steering_angle"],
        "params": {"mu": 1.0}
    }
    # build env
    env = gym.make(
        'f1tenth_gym:f1tenth-v0',
        config=config,
        render_mode=None
    )
    return env

# Render and evaluation
def make_f1tenth_env():
    config = {
        "map":_MAP,
        "num_agents":1,
        "timestep":0.01,
        "integrator":"rk4",
        "control_input": ["speed", "steering_angle"],
        "params": {"mu": 1.0}
    }
    # build env
    env = gym.make(
        'f1tenth_gym:f1tenth-v0',
        config=config, 
        render_mode="human"
    )
    return env

#-----------------------------------
# Plotting Car Path
#-----------------------------------
def plot_path(positions, total_steps, episode):
    x = []
    y = []
    coordinate = ()
    str_steps = str(total_steps)

    # -1 to remove the episode reset position
    for i in range(0, (len(positions)-1)):  
        coordinate = positions[i]  
        x.append(coordinate[0][0])
        y.append(coordinate[0][1])

    plt.figure()
    plt.title(f"Path for {str_steps} Total Time Steps: Evaluation {episode+1}")
    plt.plot(x, y, marker='o', markersize=2, color="#0d15f4")
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.axis('equal')
    plt.grid(True)
    os.makedirs(f'./plots/a2c/batch{_TEST_BATCH}-TimeStep{_TOTAL_TIME_STEPS}', exist_ok=True)
    plt.savefig(f"./plots/a2c/batch{_TEST_BATCH}-TimeStep{_TOTAL_TIME_STEPS}/car_path-{str_steps}_ep{episode+1}.png", dpi=600)
    plt.close()

#------------
# Main Method
#------------
def main():
    print("Starting A2C Training for F110...")

    # Vectorized environments
    env = make_vec_env(make_f1tenth_env_training, n_envs=1)

    policy_kwargs = dict(activation_fn=th.nn.ReLU,
                         net_arch=dict(pi=[1090, 1090], vf=[1090, 1090])
                        )
    
    model = A2C("MultiInputPolicy",
                env,
                device='cuda',
                policy_kwargs=policy_kwargs,
                verbose=0
                )
    
    print("Device:", model.device)

    model.learn(total_timesteps=_TOTAL_TIME_STEPS, progress_bar=True)
    os.makedirs('./models', exist_ok=True)
    model.save(f"./models/a2c_f1tenth-{_TOTAL_TIME_STEPS}-batch{_TEST_BATCH}")
    print("Training complete and model saved!")
    del model

    model = A2C.load(f"./models/a2c_f1tenth-{_TOTAL_TIME_STEPS}-batch{_TEST_BATCH}")
    env = make_vec_env(make_f1tenth_env, n_envs=1)

    #================
    # EVALUATION LOOP
    #================
    for evaluation in range(_NUM_EPISODES):

        start_time = time.time()
        elapsed_time = 0
        flat_positions = []
        obs = env.reset()

        print(f'Evaluation {evaluation + 1}')

        while True:
            elapsed_time = time.time() - start_time
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
    
            if _SHOW_PATH is True:
                positions = []
                flatten = []
                car_x = obs['poses_x'][0]
                car_y = obs['poses_y'][0]
                positions.append((car_x, car_y))
                flatten = np.array(positions).reshape(-1, 2)
                flat_positions.append(flatten)

            env.render() # optional to observe the car movement
            if done[0]:
                # print("Termination info:", info[0]) # optional to see additional data
                break
        env.close()
        print(f"...runtime: {elapsed_time:.2f}s\n", end='') 
        if _SHOW_PATH is True:
                plot_path(flat_positions, _TOTAL_TIME_STEPS, evaluation)
                # print('Total Episode steps: ', len(flat_positions))

#-----------------------------------
# Main guard
#-----------------------------------
if __name__ == "__main__":
    main()