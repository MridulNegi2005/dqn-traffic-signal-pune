import argparse
import json
import time

import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

import config as C
from env import PuneCrossingEnv

ENV_KEYS = ["delta_t", "obs", "reward", "action_mode"]


def algo_class(name):
    if name == "qrdqn":
        from sb3_contrib import QRDQN
        return QRDQN
    return DQN


def warm_start(model, env, transitions):
    """Fill the replay buffer with decisions of the camera rule."""
    obs = env.reset()
    n, t0 = 0, time.time()
    while n < transitions:
        acts = np.array(env.env_method("teacher_action"))
        nxt, rew, done, infos = env.step(acts)
        real_next = nxt.copy()
        for i, d in enumerate(done):
            if d and "terminal_observation" in infos[i]:
                real_next[i] = infos[i]["terminal_observation"]
        model.replay_buffer.add(obs, real_next, acts, rew, done, infos)
        obs = nxt
        n += env.num_envs
        if n % 2000 < env.num_envs:
            print(f"warm-start {n}/{transitions} transitions, {time.time() - t0:.0f}s", flush=True)


def main(a):
    device = "cuda" if torch.cuda.is_available() and not a.cpu else "cpu"
    out = C.ROOT / "results" / "runs" / a.name
    out.mkdir(parents=True, exist_ok=True)
    env_kwargs = dict(dates=C.TRAIN_DATES, episode_seconds=a.episode,
                      scale_range=C.TRAIN_SCALE_RANGE, sublane=not a.no_sublane,
                      delta_t=a.delta_t, obs=a.obs, reward=a.reward, action_mode=a.action_mode)
    env = make_vec_env(PuneCrossingEnv, n_envs=a.envs, seed=a.seed,
                       vec_env_cls=SubprocVecEnv if a.envs > 1 else None,
                       env_kwargs=env_kwargs, monitor_dir=str(out / "monitor"))
    warm = a.warmstart > 0
    kw = dict(device=device, seed=a.seed, verbose=1, learning_rate=5e-4, buffer_size=200_000,
              learning_starts=0 if warm else 2_000, batch_size=128, gamma=0.99, train_freq=4,
              gradient_steps=a.envs, target_update_interval=2_000,
              exploration_initial_eps=0.2 if warm else 1.0,
              exploration_fraction=0.2 if warm else 0.3, exploration_final_eps=0.02,
              policy_kwargs=dict(net_arch=[256, 256]))
    model = algo_class(a.algo)("MlpPolicy", env, **kw)
    t0 = time.time()
    if warm:
        warm_start(model, env, a.warmstart)
    model.learn(total_timesteps=a.steps, log_interval=4)
    model.save(C.ROOT / "results" / f"{a.name}.zip")
    meta = dict(vars(a), device=device, minutes=round((time.time() - t0) / 60, 1),
                env={k: env_kwargs[k] for k in ENV_KEYS})
    (C.ROOT / "results" / f"{a.name}_meta.json").write_text(json.dumps(meta, indent=2))
    env.close()
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="dqn")
    p.add_argument("--algo", choices=["dqn", "qrdqn"], default="dqn")
    p.add_argument("--steps", type=int, default=150_000)
    p.add_argument("--envs", type=int, default=4)
    p.add_argument("--episode", type=int, default=3600)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--delta-t", type=int, default=C.DELTA_T)
    p.add_argument("--obs", choices=["camera", "full"], default="camera")
    p.add_argument("--reward", choices=["combined", "wait"], default="combined")
    p.add_argument("--action-mode", choices=["choose", "keepswitch"], default="choose")
    p.add_argument("--warmstart", type=int, default=0, help="teacher transitions before RL")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no-sublane", action="store_true")
    main(p.parse_args())
