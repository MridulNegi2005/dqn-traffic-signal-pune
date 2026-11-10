"""Time the simulator and DQN training on CPU and GPU on the current machine."""
import json
import os
import time

import torch
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from env import PuneCrossingEnv


def sim_speed(seconds=900):
    e = PuneCrossingEnv(episode_seconds=seconds)
    e.reset(seed=0)
    t, done = time.time(), False
    while not done:
        _, _, done, _, _ = e.step(0)
    e.close()
    return round(seconds / (time.time() - t), 1)


def dqn_time(device, envs, steps):
    env = make_vec_env(PuneCrossingEnv, n_envs=envs, seed=0,
                       vec_env_cls=SubprocVecEnv if envs > 1 else None,
                       env_kwargs=dict(episode_seconds=1200))
    m = DQN("MlpPolicy", env, device=device, learning_starts=500, batch_size=128,
            train_freq=4, gradient_steps=envs, buffer_size=50_000,
            policy_kwargs=dict(net_arch=[128, 128]), seed=0)
    t = time.time()
    m.learn(steps)
    env.close()
    return round(time.time() - t, 1)


if __name__ == "__main__":
    cpus = os.cpu_count()
    out = {"cpus": cpus, "cuda": torch.cuda.is_available(),
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "sim_seconds_per_wall_second": sim_speed()}
    envs = min(8, cpus)
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for d in devices:
        out[f"dqn_6000_steps_{envs}envs_{d}_s"] = dqn_time(d, envs, 6000)
    print(json.dumps(out, indent=2))
