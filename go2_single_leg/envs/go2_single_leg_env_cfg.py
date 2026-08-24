# Copyright (c) 2024-2026. All rights reserved.
# Unitree Go2 一本足アクティブ・ホッピング (Active Continuous Hopping) 環境設定

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

from . import go2_single_leg_rewards as custom_rewards


@configclass
class Go2SingleLegRewardsCfg(RewardsCfg):
    """一本足アクティブ・ホッピング 報酬設計"""

    # 1. ★ 特大滞空時間報酬 (地面を蹴って跳ねる動作への最大インセンティブ) ★
    single_leg_air_time = RewTerm(
        func=custom_rewards.single_leg_air_time_reward,
        weight=15.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="RR_foot"), "threshold": 0.04},
    )

    # 2. ★ 接地キック推力報酬 (上向き速度の生成) ★
    push_off_thrust = RewTerm(
        func=custom_rewards.push_off_thrust_reward,
        weight=8.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="RR_foot")},
    )

    # 3. ★ 生存報酬 (耐え続けることへの基本報酬) ★
    alive = RewTerm(func=custom_rewards.alive_reward, weight=3.0)

    # 4. ★ 重心直下姿勢維持報酬 ★
    posture = RewTerm(
        func=custom_rewards.precise_orientation_reward,
        weight=4.0,
        params={"target_roll": math.radians(28.5), "target_pitch": math.radians(-30.5)},
    )

    # 5. 3脚の空中保持
    disabled_3legs_retraction = RewTerm(
        func=custom_rewards.disabled_3legs_retraction_reward,
        weight=2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["FL_foot", "FR_foot", "RL_foot"]), "min_height": 0.16},
    )

    # 6. 不正接触ペナルティ
    illegal_contact = RewTerm(
        func=custom_rewards.illegal_contact_penalty,
        weight=-6.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Head_.*", "FL_.*", "FR_.*", "RL_.*", "RR_thigh", "base"])},
    )

    # 7. トルク正則化
    dof_torques_l2 = RewTerm(func=custom_rewards.dof_torques_l2, weight=-0.0001)

    track_lin_vel_xy_exp = None
    track_ang_vel_z_exp = None
    lin_vel_z_l2 = None
    ang_vel_xy_l2 = None
    feet_air_time = None
    flat_orientation_l2 = None
    undesired_contacts = None


@configclass
class Go2SingleLegEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Go2 一本足アクティブ・ホッピング学習環境"""

    rewards: Go2SingleLegRewardsCfg = Go2SingleLegRewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"

        # ★ 精密重心直下アライメント・ソフト接地スポーン (Roll=+28.5°, Pitch=-30.5°)
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.360)
        self.scene.robot.init_state.rot = (0.2447, -0.2505, -0.0658, 0.9341)
        
        # 初期状態: 他3脚はコンパクトに折りたたみ、支持脚(RR)で構える
        self.scene.robot.init_state.joint_pos = {
            "FL_hip_joint": 0.1,
            "FR_hip_joint": -0.1,
            "RL_hip_joint": 0.2,
            "RR_hip_joint": -0.1,
            "FL_thigh_joint": -1.2,
            "FR_thigh_joint": -1.2,
            "RL_thigh_joint": 1.5,
            "RR_thigh_joint": 0.70,
            "FL_calf_joint": -1.0,
            "FR_calf_joint": -1.0,
            "RL_calf_joint": -2.5,
            "RR_calf_joint": -2.00,
        }

        # リセット時の初期外乱ゼロ化
        self.events.reset_base.params = {
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        }
        self.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.base_com = None

        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.005, 0.01)
            self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.005, 0.01)

        # ★ アクション制御: 支持脚3関節、アクティブホッピングのための可動幅(scale=0.20)
        self.actions.joint_pos.joint_names = ["RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"]
        self.actions.joint_pos.scale = 0.20

        # ----------------------------------------------------
        # 終了条件 (Terminations)
        # ----------------------------------------------------
        self.terminations.base_contact = None

        # 1. 右後足以外が地面に強く接触したら終了 (閾値 5.0N)
        self.terminations.illegal_parts_contact = DoneTerm(
            func=custom_rewards.strict_illegal_contact_termination,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Head_.*", "FL_.*", "FR_.*", "RL_.*", "RR_thigh", "base"]),
                "threshold": 5.0,
            },
        )

        # 2. 姿勢が大きく崩れたら終了 (ホッピング許容角 35度)
        self.terminations.orientation_deviation = DoneTerm(
            func=custom_rewards.orientation_deviation_termination,
            params={"max_angle_error": 0.60, "target_roll": math.radians(28.5), "target_pitch": math.radians(-30.5)},
        )

        # 3. 胴体高さが 0.16m 未満で終了
        self.terminations.root_height_below_minimum = DoneTerm(
            func=mdp.root_height_below_minimum,
            params={"minimum_height": 0.16},
        )


@configclass
class Go2SingleLegEnvCfg_PLAY(Go2SingleLegEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None

        self.observations.policy.enable_corruption = False

        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.viewer.eye = (2.2, 2.2, 1.2)
        self.viewer.lookat = (0.0, 0.0, 0.3)
