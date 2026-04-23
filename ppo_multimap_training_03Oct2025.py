import os
import numpy as np
import gymnasium as gym
import torch as th
import warnings
import yaml
import json
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from f1tenth_gym.envs.track.track import Track
from f1tenth_gym.envs.rendering import make_renderer
from f1tenth_gym.envs.reset import make_reset_fn


#--------------
# User Warnings
#--------------
# The runtime warning is related to numeric overall from the LiDar before the environemnt resets and the map loads
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
print('User warnings and Runtime warnings suppressed!')

#-----------------------
# Training Configuration
#-----------------------
EXP = 'Exp4-1'
REPEAT = 'rep1'
_TOTAL_TIME_STEPS = 20_000_000
_ENT_COEF = 0.00
BATCH_SIZE = 12_288

_TEST_BATCH = f"_MultiTrack_{EXP}_{REPEAT}"
_LOG_NAME = f"20,000,000 total time steps - MultiTrack - {EXP} - {REPEAT}"
policy_kwargs = dict(activation_fn=th.nn.ReLU,
                         net_arch=dict(pi=[512,128], vf=[512,128])
                        )
MAP_LIST = ['Catalunya', 'Monza', 'Shanghai', 'Spa', 'YasMarina', 'BrandsHatch', 'Montreal', 'IMS']

#-----------------
# Map list wrapper
#-----------------
class Multi_Map(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.map = ''
        self.idx = 0

    def reset(self, seed=None, options=None):
        obs, info = super().reset()
        self.map = MAP_LIST[self.idx]
        self.idx = (self.idx + 1) % len(MAP_LIST)
        self.config["map"] = self.map
        self.sim.set_map(self.map, self.config["scale"])

        if isinstance(self.map, Track):
            self.track = self.map
        else:
            self.track = Track.from_track_name(
                self.map,
                track_scale=self.config["scale"],
            )

        #---------------------------
        # Check point list from JSON
        #---------------------------
        self.checkpoints = []
        cp_file = f'maps/{self.map}/{self.map}_map.json'
        if Path(cp_file).exists():
            with open(cp_file, 'r') as cf:
                cp_data = json.load(cf)
            checkpoints = cp_data.get('checkpoints', cp_data)
            for cp in checkpoints:
                (x1, y1), (x2, y2) = cp
                a = np.array([x1, y1])
                b = np.array([x2, y2])
                self.checkpoints.append(np.vstack((a, b)))
        if not self.checkpoints:
            raise ValueError("No valid checkpoints.")

        with open(f'maps/{self.map}/{self.map}_map.yaml', 'r') as yaml_stream:
            try:
                map_metadata = yaml.safe_load(yaml_stream)
                yaw = map_metadata['car_position']
            except yaml.YAMLError as ex:
                print(ex)

        poses = np.array([[0.0, 0.0, yaw]])
        assert isinstance(poses, np.ndarray) and poses.shape == (
            self.num_agents,
            3,
        ), "Initial poses must be a numpy array of shape (num_agents, 3)"

        self.start_xs = poses[:, 0]
        self.start_ys = poses[:, 1]
        self.start_thetas = poses[:, 2]
        self.start_rot = np.array(
            [
                [
                    np.cos(-self.start_thetas[self.ego_idx]),
                    -np.sin(-self.start_thetas[self.ego_idx]),
                ],
                [
                    np.sin(-self.start_thetas[self.ego_idx]),
                    np.cos(-self.start_thetas[self.ego_idx]),
                ],
            ]
        )
        self.sim.reset(poses)
        action = np.zeros((self.num_agents, 2))
        obs, _, _, _, info = self.step(action)

        #---------------------
        # Check point elements
        #--------------------- 
        self.all_cp_passed = False
        self.start_pos = np.array([0.0,0.0])
        self.curr_cp = 1
        cp  = self.checkpoints[self.curr_cp - 1]
        # finds the closest point on a line segment
        A, B = cp[0], cp[1]
        v = B - A
        split = v/8
        p2 = A + split 
        p3 = p2 + split
        p4 = p3 + split
        p5 = p4 + split
        p6 = p5 + split
        p7 = p6 + split
        p8 = p7 + split
        line_segment = [A, p2, p3, p4, p5, p6, p7, p8, B]
        dist = np.linalg.norm(self.start_pos-line_segment[0])
        for i in range(len(line_segment)):
            p_current = line_segment[i]
            p_current_dist = np.linalg.norm(self.start_pos-p_current)
            if p_current_dist < dist:
                dist = p_current_dist
        self.curr_cp_dist = dist
        self.checkpoint_time = 0

        # NOTE: not confident below is needed
        # reset modes
        self.reset_fn = make_reset_fn(
            **self.config["reset_config"], track=self.track, num_agents=self.num_agents
        )

        # match render_fps to integration timestep
        self.metadata["render_fps"] = int(1.0 / self.timestep)
        if self.render_mode == "human_fast":
            self.metadata["render_fps"] *= 10  # boost fps by default: 10x
        self.renderer, self.render_spec = make_renderer(
            params=self.params,
            track=self.track,
            agent_ids=self.agent_ids,
            render_mode=self.render_mode,
            render_fps=self.metadata["render_fps"],
        )
        print(f'[INFO] map: {self.map}')
        return obs, info

#---------------------
# Environment Creation
#---------------------
def make_f1tenth_env_multi():
    config = {
        "map": MAP_LIST[0],
        "num_agents":1,
        "timestep":0.01,
        "integrator":"rk4",
        "control_input": ["speed", "steering_angle"],
        "params": {"mu": 1.0}
    }
    env = gym.make(
        'f1tenth_gym:f1tenth-v0',
        config=config,
        render_mode= None
    )
    env = Multi_Map(env)
    return env

#------------
# Main Method
#------------
def main():
    #=========
    # TRAINING
    #=========
    print("Starting PPO Training for F110...")
    print('Set agent to train on a rotation of maps...')
    env = make_vec_env(make_f1tenth_env_multi, n_envs=6)
    
    model = PPO("MultiInputPolicy",
                env,
                device='cuda',
                policy_kwargs=policy_kwargs,
                ent_coef= _ENT_COEF,
                verbose=1,
                batch_size= BATCH_SIZE, 
                tensorboard_log="./TensorBoard/"
                )
    
    print(f'\n[INFO] model name: PPO_f110-{_TOTAL_TIME_STEPS}-batch{_TEST_BATCH},\nmap list: {MAP_LIST}, \nentropy: {_ENT_COEF},\nbatch size: {BATCH_SIZE},\nCustom policy kwargs {policy_kwargs}\n')

    model.learn(total_timesteps=_TOTAL_TIME_STEPS, progress_bar=True, tb_log_name= _LOG_NAME, reset_num_timesteps=False)

    os.makedirs('./models', exist_ok=True)
    model.save(f"./models/PPO_f110-{_TOTAL_TIME_STEPS}-batch{_TEST_BATCH}")
    print("Training complete and model saved!")
    del model

#-----------------------------------
# Main guard
#-----------------------------------
if __name__ == "__main__":
    main()