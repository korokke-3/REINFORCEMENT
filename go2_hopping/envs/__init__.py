# Copyright (c) 2024-2026. All rights reserved.
# Gym 環境の登録

import gymnasium as gym

from . import go2_hopping_env_cfg

# 1. 学習用環境 (Headless / 大規模並列実行用)
gym.register(
    id="Isaac-Velocity-Rough-Unitree-Go2-Hopping-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": go2_hopping_env_cfg.Go2HoppingEnvCfg,
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:UnitreeGo2HoppingPPORunnerCfg",
    },
)

# 2. 評価・可視化用環境 (GUI表示用)
gym.register(
    id="Isaac-Velocity-Rough-Unitree-Go2-Hopping-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": go2_hopping_env_cfg.Go2HoppingEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:UnitreeGo2HoppingPPORunnerCfg",
    },
)
