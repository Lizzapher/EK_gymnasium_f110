"""
Basic Evaluation script for F110
"""
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import warnings
import time

from rich.live import Live
from rich.console import Console
from rich.table import Table

#-------------------------
# Evaluation Configuration
#-------------------------
_NETWORK_ZIP_FILE = "PPO_f110-10000000-batch_MultiTrack_Exp2-1_rep0"
_NUM_EPISODES = 10
_MAP = 'Montreal'

#------------------------
# User Warning supression
#------------------------
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
print('User warnings and runtime warnings suppressed!')
# The runtime warning is related to numeric values from the LiDar used for calculations before the environemnt resets and the map loads

#---------------------
# Environment Creation
#---------------------
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
        render_mode='human'  # 'human' pyqt renderer or 'none' for headless mode
    )
    return env

#-------------
# Console info
#-------------
def make_table(info: dict) -> Table:
    table = Table(title='\n:arrow_forward::arrow_forward::arrow_forward: [bold]AGENT INFORMATION[/] :arrow_backward::arrow_backward::arrow_backward:')
    table.add_column("INFO", justify='center') # To change color use style='...'
    table.add_column("VALUE", justify="center")
    table.add_row("[bold]Reward[bold]",   f"[bold]{info['reward']:.5f}[bold]", style="#00FFF7")
    table.add_row("Speed (m/s)",    f"{float(info['speed']):.3f}")
    table.add_row("Steer",    f"{float(info['steer']):.3f}")
    table.add_row("Throttle", f"{float(info['throttle']):.2f}")
    return table

#-----------------------------------
# Main Method
#-----------------------------------
def main():
    model = PPO.load(f"./models/{_NETWORK_ZIP_FILE}")
    env = make_vec_env(make_f1tenth_env, n_envs=1)

    # Console prep, live panel, and placeholder information
    initial_info = {"reward": 0.0, "speed": 0.0, "steer": 0.0, "throttle": 0.0}
    console = Console()
    update_every = 10
    with Live(make_table(initial_info),
                console=console,
                refresh_per_second=8,
                transient=False) as live:

        for evaluation in range(_NUM_EPISODES):
            start_time = time.time()
            elapsed_time = 0
            obs = env.reset()

            print(f'Evaluation {evaluation + 1}')
    
            step_count = 0
            while True:
                elapsed_time = time.time() - start_time

                action, _ = model.predict(obs)
                obs, _, terminated, info = env.step(action)
                info = info[0] # this is a list of dictionaries, single agent at index 0
                step_count += 1

                if step_count % update_every == 1:
                    live.update(make_table(info))

                env.render("human")
                if terminated[0]:
                    break
            print(f"...runtime: {elapsed_time:.2f}s\n", end='') 
        env.close()

#-----------------------------------
# Main guard
#-----------------------------------
if __name__ == "__main__":
    main()