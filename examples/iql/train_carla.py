"""
AWR + SAC from demo experiment
"""

from rlkit.demos.source.hdf5_path_loader import HDF5PathLoader
from rlkit.launchers.experiments.awac.finetune_rl import experiment, process_args

from rlkit.launchers.launcher_util import run_experiment

from rlkit.torch.sac.policies import GaussianPolicy
from rlkit.torch.sac.iql_trainer import IQLTrainer

import random

import d4rl
import gym
import numpy as np


class CarlaLeaderboardEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Box(low=np.array([-0.5, -1.0]), high=np.array([0.5, 1.0]), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
             low=np.array([[-4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, -1.0, 0.0]]),
             high=np.array([[4.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0]]),
             dtype=np.float32)

        self.dataset = dict(np.load("/zfsauton/datasets/ArgoRL/zheh/transfuser_autopilot/gpu16_town1346_total1203108_maxtraj9228.npz"))
        self.length = len(self.dataset['observations'])

        self._max_episode_steps = 9228

    def get_dataset(self):
        return self.dataset

    def step(self, action):
        raise Exception("Step called on environment!")

    def reset(self):
        # raise Exception("Reset called on environment!")
        import ipdb; ipdb.set_trace()
        return self.observation_space.sample()

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        pass

# Num gradient steps / num_batches
num_offline_epochs = 1_000_000 / (1203108 / 256)


variant = dict(
    algo_kwargs=dict(
        start_epoch=-1000, # offline epochs
        num_epochs=1001, # online epochs
        batch_size=256,
        num_eval_steps_per_epoch=0,
        num_trains_per_train_loop=1000,
        num_expl_steps_per_train_loop=1000,
        min_num_steps_before_training=1000,
    ),
    max_path_length=1000,
    replay_buffer_size=int(2E6),
    layer_size=256,
    policy_class=GaussianPolicy,
    policy_kwargs=dict(
        hidden_sizes=[256, 256, ],
        max_log_std=0,
        min_log_std=-6,
        std_architecture="values",
    ),
    qf_kwargs=dict(
        hidden_sizes=[256, 256, ],
    ),

    algorithm="SAC",
    version="normal",
    collection_mode='batch',
    trainer_class=IQLTrainer,
    trainer_kwargs=dict(
        discount=0.99,
        policy_lr=3E-4,
        qf_lr=3E-4,
        reward_scale=1,
        soft_target_tau=0.005,

        policy_weight_decay=0,
        q_weight_decay=0,

        reward_transform_kwargs=None,
        terminal_transform_kwargs=None,

        beta=1.0 / 3,
        quantile=0.7,
        clip_score=100,
    ),
    launcher_config=dict(
        num_exps_per_instance=1,
        region='us-west-2',
    ),

    path_loader_class=HDF5PathLoader,
    path_loader_kwargs=dict(),
    add_env_demos=False,
    add_env_offpolicy_data=False,

    load_demos=False,
    load_env_dataset_demos=True,

    normalize_env=False,
    env_class = CarlaLeaderboardEnv,
    normalize_rewards_by_return_range=True,

    seed=random.randint(0, 100000),
)

def main():
    run_experiment(experiment,
        variant=variant,
        exp_prefix='train_leaderboard_seed_2',
        mode="here_no_doodad",
        unpack_variant=False,
        use_gpu=True,
    )

if __name__ == "__main__":
    main()
