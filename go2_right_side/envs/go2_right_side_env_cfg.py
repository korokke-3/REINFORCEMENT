# Copyright (c) 2024-2026. All rights reserved.
# Unitree Go2 "右側2脚歩行" 環境設定ファイル

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

from . import go2_right_side_rewards as custom_rewards


@configclass
class Go2RightSideRewardsCfg(RewardsCfg):
    """右側2脚歩行用 報酬設計"""

    # 1. 速度追従
    track_lin_vel_xy_exp = RewTerm(
        func=custom_rewards.track_lin_vel_xy_exp,
        weight=3.0,
        params={"std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=custom_rewards.track_ang_vel_z_exp,
        weight=0.75,
        params={"std": math.sqrt(0.25)},
    )

    # 2. 右側2脚ステップ滞空時間
    right_side_feet_air_time = RewTerm(
        func=custom_rewards.right_side_feet_air_time,
        weight=2.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FR_foot", "RR_foot"]),
            "command_name": "base_velocity",
            "threshold": 0.08,
        },
    )

    # 3. 左側脚・胴体の接地ペナルティ (引きずり根絶)
    illegal_left_contacts = RewTerm(
        func=custom_rewards.illegal_left_contacts_penalty,
        weight=-4.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_.*", "RL_.*"])},
    )
    left_legs_height = RewTerm(
        func=custom_rewards.left_legs_height_reward,
        weight=1.5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["FL_foot", "RL_foot"]), "target_height": 0.20},
    )

    # 4. 静止ペナルティ
    stand_still_penalty = RewTerm(
        func=custom_rewards.stand_still_penalty,
        weight=-2.0,
        params={"threshold": 0.15},
    )

    # 5. 姿勢・高さ
    base_height_exp = RewTerm(
        func=custom_rewards.base_height_exp,
        weight=1.0,
        params={"target_height": 0.32, "std": 0.06},
    )
    right_tilt_orientation = RewTerm(
        func=custom_rewards.right_tilt_orientation_reward,
        weight=1.0,
        params={"target_roll_deg": 15.0, "std": 0.2},
    )

    # 6. エネルギー抑制
    lin_vel_z_l2 = RewTerm(func=custom_rewards.lin_vel_z_l2, weight=0.0)
    dof_torques_l2 = RewTerm(func=custom_rewards.dof_torques_l2, weight=-0.0002)
    dof_acc_l2 = RewTerm(func=custom_rewards.dof_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=custom_rewards.action_rate_l2, weight=-0.01)

    feet_air_time = None
    flat_orientation_l2 = None
    undesired_contacts = None


@configclass
class Go2RightSideEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Go2 右側2脚歩行環境設定"""

    rewards: Go2RightSideRewardsCfg = Go2RightSideRewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"

        # 初期スポーン姿勢 (右前FR・右後RR接地、左脚FL・RL引き込み、右傾き15度)
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.34)
        self.scene.robot.init_state.rot = (0.1305, 0.0, 0.0, 0.9914)  # ロール +15度
        self.scene.robot.init_state.joint_pos = {
            "FR_hip_joint": -0.1,
            "RR_hip_joint": -0.1,
            "FL_hip_joint": 0.2,
            "RL_hip_joint": 0.2,
            "FL_thigh_joint": -0.8,
            "FL_calf_joint": -1.0,
            "RL_thigh_joint": 0.0,
            "RL_calf_joint": -1.0,
            "FR_thigh_joint": 0.8,
            "FR_calf_joint": -1.5,
            "RR_thigh_joint": 0.8,
            "RR_calf_joint": -1.5,
        }

        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.01, 0.03)
            self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.03)

        self.actions.joint_pos.scale = 0.25

        # 転倒・這い歩き終了条件
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"
        self.terminations.root_height_below_minimum = DoneTerm(
            func=mdp.root_height_below_minimum,
            params={"minimum_height": 0.20},
        )


@configclass
class Go2RightSideEnvCfg_PLAY(Go2RightSideEnvCfg):
    """可視化・動画録画用の設定"""

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
        self.viewer.lookat = (0.0, 0.0, 0.3)
