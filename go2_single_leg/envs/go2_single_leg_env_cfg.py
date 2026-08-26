# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
    CommandsCfg,
    ActionsCfg,
    ObservationsCfg,
    EventsCfg,
    TerminationsCfg,
    CurriculumCfg,
)
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

from . import go2_single_leg_rewards as custom_rewards


@configclass
class Go2SingleLegActionsCfg(ActionsCfg):
    """Action specifications for Right-Rear Single Leg + Front Leg Tail Balancers."""
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["RR_hip_joint", "RR_thigh_joint", "RR_calf_joint", "FL_thigh_joint", "FR_thigh_joint"],
        scale=0.5,
        use_default_offset=True,
    )


@configclass
class Go2SingleLegRewardsCfg(RewardsCfg):
    # 親の不要報酬を無効化
    track_lin_vel_xy_exp = None
    track_ang_vel_z_exp = None
    feet_air_time = None
    undesired_contacts = None
    flat_orientation_l2 = None
    base_height_l2 = None
    lin_vel_z_l2 = None
    ang_vel_xy_l2 = None

    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-0.0001)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    # 1. 弾性リバウンド打ち上げ報酬
    explosive_launch = RewTerm(
        func=custom_rewards.pure_toe_explosive_rebound_reward,
        weight=25.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="RR_foot")},
    )

    # 2. 完全滞空クリアランス報酬
    flight_clearance = RewTerm(
        func=custom_rewards.pure_toe_air_clearance_reward,
        weight=25.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="RR_foot"), "min_clearance": 0.04},
    )

    # 3. Raibert Hopper 型 重心直下足先配置報酬 (着地転倒モーメントゼロ化)
    foot_placement = RewTerm(
        func=custom_rewards.raibert_foot_placement_reward,
        weight=15.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="RR_foot")},
    )

    # 4. 空中姿勢アクティブ安定化報酬 (腕振りバランス)
    arm_balance = RewTerm(
        func=custom_rewards.midair_arm_balance_orientation_reward,
        weight=10.0,
    )

    # 5. 接地引きずり・寝そべり禁止ペナルティ
    ground_stagnation = RewTerm(
        func=custom_rewards.pure_toe_ground_stagnation_penalty,
        weight=-10.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="RR_foot")},
    )

    # 6. 生存報酬 & 他脚高空格納報酬
    alive = RewTerm(func=custom_rewards.pure_toe_alive_reward, weight=4.0)
    disabled_3legs_high = RewTerm(
        func=custom_rewards.disabled_3legs_high_airborne_reward,
        weight=4.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["FL_foot", "FR_foot", "RL_foot"]), "min_height": 0.12},
    )

    # 7. 膝・他パーツ接触の超厳格ペナルティ
    illegal_contact = RewTerm(
        func=custom_rewards.strict_illegal_contact_penalty,
        weight=-30.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[".*thigh.*", ".*calf.*", ".*base.*", "FL_foot", "FR_foot", "RL_foot"]
            )
        },
    )


@configclass
class Go2SingleLegTerminationsCfg(TerminationsCfg):
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = None
    
    # 膝・他脚・胴体が床に 3.0N でも触れたら即座に終了 (足先ゴム球のみ許容)
    illegal_parts_contact = DoneTerm(
        func=custom_rewards.pure_toe_strict_illegal_contact_termination,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[".*thigh.*", ".*calf.*", ".*base.*", "FL_foot", "FR_foot", "RL_foot"]
            ),
            "threshold": 3.0,
        },
    )
    
    orientation_deviation = DoneTerm(
        func=custom_rewards.pure_toe_orientation_deviation_termination,
        params={"max_angle_error": 0.70},
    )
    
    root_height_below_minimum = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.12},
    )


@configclass
class Go2SingleLegCurriculumCfg(CurriculumCfg):
    terrain_levels = None


@configclass
class Go2SingleLegEnvCfg(LocomotionVelocityRoughEnvCfg):
    actions: Go2SingleLegActionsCfg = Go2SingleLegActionsCfg()
    rewards: Go2SingleLegRewardsCfg = Go2SingleLegRewardsCfg()
    terminations: Go2SingleLegTerminationsCfg = Go2SingleLegTerminationsCfg()
    curriculum: Go2SingleLegCurriculumCfg = Go2SingleLegCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"

        self.decimation = 4
        self.episode_length_s = 5.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation

        # ★ 足を最大伸展させた完全スポーン姿勢 (Roll=+22.0°, Pitch=-16.0°) ★
        opt_roll = math.radians(22.0)
        opt_pitch = math.radians(-16.0)
        cr = math.cos(opt_roll * 0.5); sr = math.sin(opt_roll * 0.5)
        cp = math.cos(opt_pitch * 0.5); sp = math.sin(opt_pitch * 0.5)
        qw = cr * cp; qx = sr * cp; qy = cr * sp; qz = -sr * sp

        self.scene.robot.init_state.pos = (0.0, 0.0, 0.415)
        self.scene.robot.init_state.rot = (qx, qy, qz, qw)
        self.scene.robot.init_state.joint_pos = {
            "FL_hip_joint": 0.1,
            "FR_hip_joint": -0.1,
            "RL_hip_joint": 0.2,
            "RR_hip_joint": 0.05,
            "FL_thigh_joint": -1.4,
            "FR_thigh_joint": -1.4,
            "RL_thigh_joint": 1.8,
            "RR_thigh_joint": -0.20,
            "FL_calf_joint": -0.9,
            "FR_calf_joint": -0.9,
            "RL_calf_joint": -2.6,
            "RR_calf_joint": -0.838, # 最大伸展限界
        }

        # コマンドゼロ
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        # フラット地形
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}


@configclass
class Go2SingleLegEnvCfg_PLAY(Go2SingleLegEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
