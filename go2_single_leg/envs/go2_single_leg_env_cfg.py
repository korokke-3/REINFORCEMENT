# Copyright (c) 2024-2026. All rights reserved.
# Unitree Go2 真の一本足けんけん (Strict Single-Leg Forward Hopping) 環境設定

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
class Go2SingleLegCommandsCfg(CommandsCfg):
    """前進けんけん速度コマンド設定"""
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.0,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.4, 0.8), # 前進 0.4〜0.8 m/s
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=(0.0, 0.0),
        ),
    )


@configclass
class Go2SingleLegRewardsCfg(RewardsCfg):
    """一本足けんけん 報酬設計"""

    # 1. ★ 特大滞空時間報酬 (地面を蹴ってピョンと跳ねる動作) ★
    single_leg_hop_air_time = RewTerm(
        func=custom_rewards.single_leg_hop_air_time_reward,
        weight=15.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="RR_foot"), "threshold": 0.04},
    )

    # 2. ★ 前進キック推力報酬 (床を蹴って前・上へ打ち出す) ★
    forward_thrust = RewTerm(
        func=custom_rewards.single_leg_forward_thrust_reward,
        weight=8.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="RR_foot")},
    )

    # 3. ★ 前進速度追従報酬 (前へ跳ね進む) ★
    track_lin_vel_xy_exp = RewTerm(
        func=custom_rewards.track_forward_vel_exp,
        weight=10.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    # 4. 生存報酬
    alive = RewTerm(func=custom_rewards.single_leg_alive_reward, weight=4.0)

    # 5. 他3脚の完全高空保持報酬
    disabled_3legs_high = RewTerm(
        func=custom_rewards.disabled_3legs_high_airborne_reward,
        weight=3.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["FL_foot", "FR_foot", "RL_foot"]), "min_height": 0.08},
    )

    # 6. 不正接触ペナルティ
    illegal_contact = RewTerm(
        func=custom_rewards.strict_illegal_contact_penalty,
        weight=-25.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Head_.*", "FL_.*", "FR_.*", "RL_.*", "base"])},
    )

    # 7. アクション滑らかさ
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    # 8. トルク正則化
    dof_torques_l2 = RewTerm(func=custom_rewards.dof_torques_l2, weight=-0.0001)

    track_ang_vel_z_exp = None
    lin_vel_z_l2 = None
    ang_vel_xy_l2 = None
    feet_air_time = None
    flat_orientation_l2 = None
    undesired_contacts = None


@configclass
class Go2SingleLegEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Go2 真の一本足けんけん学習環境"""

    commands: Go2SingleLegCommandsCfg = Go2SingleLegCommandsCfg()
    rewards: Go2SingleLegRewardsCfg = Go2SingleLegRewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"

        # ★ 精密重心直下アライメント・他3脚完全高空保持 (Roll=+25.3°, Pitch=-20.0°)
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.330)
        self.scene.robot.init_state.rot = (0.2201, -0.2079, -0.0480, 0.9518)
        
        # 初期状態: 他3脚は上空へ完全に引き上げ固定、支持脚(RR)のみ接地構え
        self.scene.robot.init_state.joint_pos = {
            "FL_hip_joint": 0.1,
            "FR_hip_joint": -0.1,
            "RL_hip_joint": 0.2,
            "RR_hip_joint": 0.05,
            "FL_thigh_joint": -1.4,
            "FR_thigh_joint": -1.4,
            "RL_thigh_joint": 1.8,
            "RR_thigh_joint": 0.42,
            "FL_calf_joint": -0.9,
            "FR_calf_joint": -0.9,
            "RL_calf_joint": -2.6,
            "RR_calf_joint": -1.50,
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

        # ★ アクション制御: 支持脚(RR)の3関節、けんけん跳躍のための十分な可動幅(scale=0.25)
        self.actions.joint_pos.joint_names = ["RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"]
        self.actions.joint_pos.scale = 0.25

        # ----------------------------------------------------
        # 厳格な終了条件 (Terminations)
        # ----------------------------------------------------
        self.terminations.base_contact = None

        # 1. ★ 右後足(RR)以外が地面に 3.0N でも触れたら即座に終了！ ★
        self.terminations.illegal_parts_contact = DoneTerm(
            func=custom_rewards.absolute_strict_illegal_contact_termination,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Head_.*", "FL_.*", "FR_.*", "RL_.*", "base"]),
                "threshold": 3.0,
            },
        )

        # 2. ★ 姿勢が崩れたら即座に終了 ★
        self.terminations.orientation_deviation = DoneTerm(
            func=custom_rewards.hopping_orientation_deviation_termination,
            params={"max_angle_error": 0.65, "target_roll": math.radians(25.3), "target_pitch": math.radians(-20.0)},
        )

        # 3. ★ 胴体高さが 0.16m 未満で終了 ★
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
