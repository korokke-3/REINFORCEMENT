# Copyright (c) 2024-2026. All rights reserved.
# Unitree Go2 "片足けんけん" (Hopping / 3-Legged Locomotion) 環境設定ファイル

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

from . import go2_hopping_rewards as custom_rewards


@configclass
class Go2HoppingRewardsCfg(RewardsCfg):
    """
    【ユーザー編集領域】報酬設計 (Reward Terms)
    
    「片足けんけん」を成立させるために、プラス報酬とペナルティの重み（weight）を調整してください。
    正の値: その動作を促す (プラス報酬)
    負の値: その動作を抑制する (ペナルティ)
    """

    # -------------------------------------------------------------
    # 1. 速度追従 (コマンドに従って前進/旋回する)
    # -------------------------------------------------------------
    track_lin_vel_xy_exp = RewTerm(
        func=custom_rewards.track_lin_vel_xy_exp,
        weight=2.0,
        params={"std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=custom_rewards.track_ang_vel_z_exp,
        weight=0.75,
        params={"std": math.sqrt(0.25)},
    )

    # -------------------------------------------------------------
    # 2. ★ けんけん / 片足浮かせ用 カスタム報酬・ペナルティ ★
    # -------------------------------------------------------------
    # (A) 浮かせる脚 (例: 右前足 'FR_foot') が地面に接触したときのペナルティ
    disabled_leg_contact = RewTerm(
        func=custom_rewards.disabled_leg_contact_penalty,
        weight=-2.5,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="FR_foot")},
    )

    # (B) 浮かせる脚の高さを一定以上に維持するプラス報酬 (例: 地面から 0.15m 以上)
    disabled_leg_height = RewTerm(
        func=custom_rewards.disabled_leg_height_reward,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="FR_foot"), "target_height": 0.15},
    )

    # (C) 支え脚 (FL, RL, RR) の滞空時間・周期的な跳躍を促す報酬 (ホッピング推進)
    hopping_air_time = RewTerm(
        func=custom_rewards.hopping_feet_air_time,
        weight=1.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_foot", "RL_foot", "RR_foot"]),
            "command_name": "base_velocity",
            "threshold": 0.08,
        },
    )

    # -------------------------------------------------------------
    # 3. 安定性・姿勢制御 (転倒防止・過度な傾きの抑制)
    # -------------------------------------------------------------
    # 胴体の高さ維持 (低すぎず高すぎず)
    base_height_exp = RewTerm(
        func=custom_rewards.base_height_exp,
        weight=0.5,
        params={"target_height": 0.28, "std": 0.05},
    )

    # 胴体のピッチ/ロール傾きペナルティ (3本足でバランスを保つため重要)
    flat_orientation_l2 = RewTerm(
        func=custom_rewards.flat_orientation_l2,
        weight=-1.0,
    )

    # 胴体のZ軸方向（上下）の振動ペナルティ (跳躍・ホッピングを妨げないよう 0.0 に設定)
    lin_vel_z_l2 = RewTerm(
        func=custom_rewards.lin_vel_z_l2,
        weight=0.0,
    )

    # -------------------------------------------------------------
    # 4. エネルギー効率・滑らかさ (モータへの負荷軽減)
    # -------------------------------------------------------------
    dof_torques_l2 = RewTerm(
        func=custom_rewards.dof_torques_l2,
        weight=-0.0002,
    )
    dof_acc_l2 = RewTerm(
        func=custom_rewards.dof_acc_l2,
        weight=-2.5e-7,
    )
    action_rate_l2 = RewTerm(
        func=custom_rewards.action_rate_l2,
        weight=-0.01,
    )

    # -------------------------------------------------------------
    # 5. 親クラスの不要な項の無効化・Go2用上書き
    # -------------------------------------------------------------
    feet_air_time = None
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_thigh"), "threshold": 1.0},
    )


@configclass
class Go2HoppingEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Go2 片足けんけん強化学習環境の全体設定"""

    # カスタム報酬クラスを割り当て
    rewards: Go2HoppingRewardsCfg = Go2HoppingRewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        # ロボットモデルに Unitree Go2 を設定
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # 高さスキャナ (RayCaster) の基準をベースに設定
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"

        # 平地または緩やかな地形から始める (最初は flat 推奨)
        # 必要に応じて rough terrain の難易度を調整
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.01, 0.04)
            self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.03)

        # アクションスケール (PDコントローラの目標角度変化量)
        self.actions.joint_pos.scale = 0.25

        # 転倒時のエピソード終了条件 (ベース胴体が地面に接触したら終了)
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"


@configclass
class Go2HoppingEnvCfg_PLAY(Go2HoppingEnvCfg):
    """学習済みモデルの評価・可視化用 (画面描画・録画用) の設定"""

    def __post_init__(self):
        super().__post_init__()

        # 可視化用のロボット数を 1体に設定
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5

        # 平地にして安定したスポーンとカメラアングルを確保
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None

        # ノイズやランダムな外力付加を無効化
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        # ロボット追従カメラ設定 (ロボットの斜め後方上空から追尾)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.viewer.eye = (2.2, 2.2, 1.2)
        self.viewer.lookat = (0.0, 0.0, 0.3)
