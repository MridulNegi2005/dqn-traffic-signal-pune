import argparse
import json
import time

import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

import config as C
from env import PuneCrossingEnv


def main(a):
    device = "cuda" if torch.cuda.is_available() and not a.cpu else "cpu"
    out = C.ROOT / "results" / "runs" / a.name
    out.mkdir(parents=True, exist_ok=True)
    env = make_vec_env(PuneCrossingEnv, n_envs=a.envs, seed=a.seed,
                       vec_env_cls=SubprocVecEnv if a.envs > 1 else None,
                       env_kwargs=dict(dates=C.TRAIN_DATES, episode_seconds=a.episode,
                                       sublane=not a.no_sublane),
                       monitor_dir=str(out / "monitor"))
    model = DQN(
        "MlpPolicy", env, device=device, seed=a.seed, verbose=1,
        learning_rate=5e-4, buffer_size=100_000, learning_starts=2_000,
        batch_size=128, gamma=0.99, train_freq=4, gradient_steps=a.envs,
        target_update_interval=2_000, exploration_fraction=0.3,
        exploration_final_eps=0.02, policy_kwargs=dict(net_arch=[128, 128]),
    )
    t0 = time.time()
    model.learn(total_timesteps=a.steps, progress_bar=False,
                callback=CheckpointCallback(20_000 // a.envs, str(out / "ckpt")))
    model.save(C.ROOT / "results" / f"{a.name}.zip")
    meta = dict(vars(a), device=device, minutes=round((time.time() - t0) / 60, 1))
    (C.ROOT / "results" / f"{a.name}_meta.json").write_text(json.dumps(meta, indent=2))
    env.close()
    print(meta)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="dqn_pune")
    p.add_argument("--steps", type=int, default=150_000)
    p.add_argument("--envs", type=int, default=8)
    p.add_argument("--episode", type=int, default=3600)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no-sublane", action="store_true")
    main(p.parse_args())
