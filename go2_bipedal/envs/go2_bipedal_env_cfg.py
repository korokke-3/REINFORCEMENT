# Copyright (c) 2024-2026. All rights reserved.
# Unitree Go2 "後足2本立ち歩行" 環境設定ファイル (修正版: 直立スポーン & 前進強制)

from __future__ import annotations

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    CommandsCfg,
    ActionsCfg,
    ObservationsCfg,
    EventsCfg,
    RewardsCfg,
    TerminationsCfg,
    CurriculumCfg,
)
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

from . import go2_bipedal_rewards as custom_rewards


@configclass
class Go2BipedalRewardsCfg(RewardsCfg):
    """2本足歩行用の報酬設計"""

    # 1. 速度追従 (前進を強く促す)
    track_lin_vel_xy_exp = RewTerm(
        func=custom_rewards.track_lin_vel_xy_exp,
        weight=2.5,
        params={"std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=custom_rewards.track_ang_vel_z_exp,
        weight=0.75,
        params={"std": math.sqrt(0.25)},
    )

    # 2. 静止ペナルティ (止まったらマイナス)
    stand_still_penalty = RewTerm(
        func=custom_rewards.stand_still_penalty,
        weight=-1.5,
        params={"threshold": 0.15},
    )

    # 3. 前足浮かせ & 後足歩行 (前進時のみプラス)
    front_feet_contact = RewTerm(
        func=custom_rewards.front_feet_contact_penalty,
        weight=-3.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_foot", "FR_foot"])},
    )
    front_feet_height = RewTerm(
        func=custom_rewards.front_feet_height_reward,
        weight=1.5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["FL_foot", "FR_foot"]), "target_height": 0.30},
    )
    biped_feet_air_time = RewTerm(
        func=custom_rewards.biped_feet_air_time,
        weight=2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["RL_foot", "RR_foot"]),
            "command_name": "base_velocity",
            "threshold": 0.08,
        },
    )

    # 4. 直立姿勢 (前進時のみプラス)
    biped_base_height = RewTerm(
        func=custom_rewards.biped_base_height_exp,
        weight=1.0,
        params={"target_height": 0.38, "std": 0.08},
    )
    biped_orientation = RewTerm(
        func=custom_rewards.biped_orientation_reward,
        weight=1.5,
        params={"target_pitch_deg": 55.0, "std": 0.2},
    )

    # 5. エネルギー抑制
    lin_vel_z_l2 = RewTerm(func=custom_rewards.lin_vel_z_l2, weight=0.0)
    dof_torques_l2 = RewTerm(func=custom_rewards.dof_torques_l2, weight=-0.0002)
    dof_acc_l2 = RewTerm(func=custom_rewards.dof_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=custom_rewards.action_rate_l2, weight=-0.01)

    feet_air_time = None
    flat_orientation_l2 = None
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_thigh"), "threshold": 1.0},
    )


@configclass
class Go2BipedalEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Go2 2本足立ち歩行環境設定"""

    rewards: Go2BipedalRewardsCfg = Go2BipedalRewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        # ロボットモデルの設定
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"

        # ★ 初期直立スポーン姿勢 (ピッチ55度、後足接地、前足折りたたみ)
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.36)
        self.scene.robot.init_state.rot = (0.0, 0.4618, 0.0, 0.8870)  # ピッチ +55度
        self.scene.robot.init_state.joint_pos = {
            ".*L_hip_joint": 0.1,
            ".*R_hip_joint": -0.1,
            "F[L,R]_thigh_joint": -1.0,  # 前足を胸元へ引き込む
            "F[L,R]_calf_joint": -1.0,
            "R[L,R]_thigh_joint": 0.7,   # 後足でしっかり支える
            "R[L,R]_calf_joint": -1.4,
        }

        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.01, 0.03)
            self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.03)

        self.actions.joint_pos.scale = 0.25
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"


@configclass
class Go2BipedalEnvCfg_PLAY(Go2BipedalEnvCfg):
    """可視化用設定"""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None

        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.viewer.eye = (2.2, 2.2, 1.2)
        self.viewer.lookat = (0.0, 0.0, 0.4)
