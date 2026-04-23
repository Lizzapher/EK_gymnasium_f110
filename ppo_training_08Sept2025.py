"""
Basic PPO Training Script for F1Tenth Racing Environment
with Checkpoint-Based system
"""

import os
import time
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import torch as th
import warnings

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

#-------------------------
# User Warnings Suppressed
#-------------------------
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
print('User warnings and Runtime warnings suppressed!')
# The runtime warning is related to numeric overall from the LiDar before the environemnt resets and the map loads

#-----------------------
# Training Configuration
#-----------------------
_TEST_BATCH = "_LiDar_IMS_mscs_12000000_rep0"
_LOG_NAME = "0% Entropy - 12,000,000 total time steps - Brands Hatch - mscs_rep4"

policy_kwargs = dict(activation_fn=th.nn.ReLU,
                         net_arch=dict(pi=[512,128], vf=[512,128])
                        )

MAP = 'BrandsHatch'
TOTAL_TIME_STEPS = 12_000_000
NUM_EPISODES = 100
SHOW_PATH = False
ENT_COEF = 0.00
BATCH_SIZE = 8192

#---------------------
# Environment Creation
#---------------------
# Headless Mode
def make_f1tenth_env():
    config = {
        "map":MAP,
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

#------------------
# Plotting Car Path
#------------------
# TODO: overaly the car path onto the track map for appropraite context
def plot_path(positions, total_steps, test_name, episode):
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
    plt.plot(x, y, marker='o', markersize=1.0, color="#00c3ff")
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.axis('equal')
    plt.grid(True)
    os.makedirs(f'./plots/ppo/batch{test_name}-TimeStep{total_steps}', exist_ok=True)
    plt.savefig(f"./plots/ppo/batch{test_name}-TimeStep{total_steps}/map_{MAP}_car_path-{str_steps}_ep{episode+1}.png", dpi=400)
    plt.close()

#------------
# Main Method
#------------
def main():
    #=========
    # TRAINING
    #=========
    print("Starting PPO Training for F110...")
    # Vectorized environments
    env = make_vec_env(make_f1tenth_env, n_envs=6)
    
    model = PPO("MultiInputPolicy",
                env,
                device='cuda',
                policy_kwargs=policy_kwargs,
                ent_coef= ENT_COEF,
                verbose=1, 
                batch_size = BATCH_SIZE,
                tensorboard_log="./TensorBoard/"
                )
    
    print(f'\n[INFO] model name: PPO_f110-{TOTAL_TIME_STEPS}-batch{_TEST_BATCH}\nmap: {MAP}\nentropy: {ENT_COEF}\nCustom policy kwargs {policy_kwargs}\nbatch size: {BATCH_SIZE}\n\n')
 
    model.learn(total_timesteps=TOTAL_TIME_STEPS, progress_bar=True, tb_log_name= _LOG_NAME, reset_num_timesteps=False)

    os.makedirs('./models', exist_ok=True)
    model.save(f"./models/PPO_f110-{TOTAL_TIME_STEPS}-batch{_TEST_BATCH}")
    print("Training complete and model saved!")

    del model

    #===========
    # EVALUATION
    #===========
    model = PPO.load(f"./models/PPO_f110-{TOTAL_TIME_STEPS}-batch{_TEST_BATCH}")
    env = make_vec_env(make_f1tenth_env, n_envs=1)

    for evaluation in range(NUM_EPISODES):
        start_time = time.time()
        elapsed_time = 0
        flat_positions = []
        obs = env.reset()

        # print(f'Evaluation {evaluation + 1}')

        while True:
            elapsed_time = time.time() - start_time
            action, _states = model.predict(obs)
            obs, _, terminated, _ = env.step(action)
    
            if SHOW_PATH is True:
                positions = []
                flatten = []
                car_x = obs['poses_x'][0]
                car_y = obs['poses_y'][0]
                positions.append((car_x, car_y))
                flatten = np.array(positions).reshape(-1, 2)
                flat_positions.append(flatten)

            env.render("human") # optional to observe the car movement
            if terminated[0]:
                break
        env.close()
        # print(f"...runtime: {elapsed_time:.2f}s\n", end='') 
        if SHOW_PATH is True:
                # print(f'number of steps per episode @ 6.5 m/s: {len(flat_positions)}')
                plot_path(flat_positions, TOTAL_TIME_STEPS, _TEST_BATCH, evaluation)

#-----------------------------------
# Main guard
#-----------------------------------
if __name__ == "__main__":
    main()