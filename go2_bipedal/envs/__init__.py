# Copyright (c) 2024-2026. All rights reserved.
# Gym 環境の登録 (Go2 Bipedal)

import gymnasium as gym
from . import go2_bipedal_env_cfg

# 1. 学習用環境
gym.register(
    id="Isaac-Velocity-Rough-Unitree-Go2-Bipedal-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": go2_bipedal_env_cfg.Go2BipedalEnvCfg,
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:UnitreeGo2BipedalPPORunnerCfg",
    },
)

# 2. 評価・可視化用環境
gym.register(
    id="Isaac-Velocity-Rough-Unitree-Go2-Bipedal-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": go2_bipedal_env_cfg.Go2BipedalEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:UnitreeGo2BipedalPPORunnerCfg",
    },
)
