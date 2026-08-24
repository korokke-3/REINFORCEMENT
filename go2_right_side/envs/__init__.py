import gymnasium as gym
from . import go2_right_side_env_cfg

gym.register(
    id="Isaac-Velocity-Rough-Unitree-Go2-RightSide-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": go2_right_side_env_cfg.Go2RightSideEnvCfg,
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:UnitreeGo2RightSidePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Unitree-Go2-RightSide-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": go2_right_side_env_cfg.Go2RightSideEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:UnitreeGo2RightSidePPORunnerCfg",
    },
)
