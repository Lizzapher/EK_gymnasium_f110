"""
Basic PPO Training Script for F1Tenth Racing Environment
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

#-----------------------------------
# User Warnings Suppressed
#-----------------------------------
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
print('User warnings and Runtime warnings suppressed!')
# The runtime warning is related to numeric overall from the LiDar before the environemnt resets and the map loads

#-----------------------------------
# Training Configuration
#-----------------------------------
_ENT_COEF = 0.00
_NETWORK_ZIP_FILE = "PPO_f110-6000000-batch_LiDar_IMS_Exp25-2_rep4_e"
_TEST_BATCH = "_LiDar_IMS_Exp25-3_rep4_e_4"
_LOG_NAME = "0% Entropy - 8,000,000 total time steps - IMS - Exp.25-3_rep4_e_4"

#-----------------
# General training
#-----------------
_MAP = 'IMS'
_SHOW_PATH = True
_NUM_EPISODES = 50
_TOTAL_TIME_STEPS = 6_000_000
_ADDITIONAL_TRAINING_STEPS = 2_000_000
_NEW_TOTAL_STEPS = _TOTAL_TIME_STEPS + _ADDITIONAL_TRAINING_STEPS

#---------------------
# Environment Creation
#---------------------
# Headless Mode
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
    plt.savefig(f"./plots/ppo/batch{test_name}-TimeStep{total_steps}/map_{_MAP}_car_path-{str_steps}_ep{episode+1}.png", dpi=400)
    plt.close()

#------------
# Main Method
#------------
def main():
    #===================
    # CONTINUED TRAINING
    #===================
    print("Resuming PPO Training for F110...")
    print(f'\n[INFO] INPUT model name:  {_NETWORK_ZIP_FILE}.zip, map: {_MAP}, entropy: {_ENT_COEF}')
    print(f'\n[INFO] OUPUT model name: PPO_f110-{_NEW_TOTAL_STEPS}-batch{_TEST_BATCH}, map: {_MAP}, entropy: {_ENT_COEF}, previous training: {_TOTAL_TIME_STEPS}, additional training: {_ADDITIONAL_TRAINING_STEPS}, new total time steps: {_NEW_TOTAL_STEPS}\n')

    env = make_vec_env(make_f1tenth_env, n_envs=6)
    model = PPO.load(f"./models/{_NETWORK_ZIP_FILE}", env=env, tensorboard_log="./TensorBoard/", ent_coef=_ENT_COEF)

    model.learn(total_timesteps=_ADDITIONAL_TRAINING_STEPS, progress_bar=True, tb_log_name= _LOG_NAME, reset_num_timesteps=False)

    os.makedirs('./models', exist_ok=True)
    model.save(f"./models/PPO_f110-{_NEW_TOTAL_STEPS}-batch{_TEST_BATCH}")
    print("Training complete and UPDATED model saved!")
    del model

    #===========
    # EVALUATION
    #===========
    model = PPO.load(f"./models/PPO_f110-{_NEW_TOTAL_STEPS}-batch{_TEST_BATCH}")
    env = make_vec_env(make_f1tenth_env, n_envs=1)

    for evaluation in range(_NUM_EPISODES):
        start_time = time.time()
        elapsed_time = 0
        flat_positions = []
        obs = env.reset()

        print(f'Evaluation {evaluation + 1}')

        while True:
            elapsed_time = time.time() - start_time
            action, _ = model.predict(obs)
            obs, _, done, _ = env.step(action)
    
            if _SHOW_PATH is True:
                positions = []
                flatten = []
                car_x = obs['poses_x'][0]
                car_y = obs['poses_y'][0]
                positions.append((car_x, car_y))
                flatten = np.array(positions).reshape(-1, 2)
                flat_positions.append(flatten)

            env.render("human")
            if done[0]:
                break
        env.close()
        print(f"...runtime: {elapsed_time:.2f}s\n", end='') 
        if _SHOW_PATH is True:
                plot_path(flat_positions, _NEW_TOTAL_STEPS, _TEST_BATCH, evaluation)

#-----------------------------------
# Main guard
#-----------------------------------
if __name__ == "__main__":
    main()