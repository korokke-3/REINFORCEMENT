import gymnasium as gym
from . import go2_single_leg_env_cfg

gym.register(
    id="Isaac-Velocity-Rough-Unitree-Go2-SingleLeg-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": go2_single_leg_env_cfg.Go2SingleLegEnvCfg,
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:UnitreeGo2SingleLegPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Unitree-Go2-SingleLeg-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": go2_single_leg_env_cfg.Go2SingleLegEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:UnitreeGo2SingleLegPPORunnerCfg",
    },
)
