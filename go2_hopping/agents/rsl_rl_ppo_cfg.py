from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class UnitreeGo2HoppingPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """
    【ユーザー編集領域】PPO アルゴリズム設定
    
    学習の進み具合に合わせて、反復回数 (max_iterations) や学習率 (learning_rate) を変更できます。
    """
    num_steps_per_env = 24
    max_iterations = 1500  # 学習の総イテレーション数 (1000〜2000で十分な歩行を獲得可能)
    save_interval = 100    # チェックポイントの保存間隔
    experiment_name = "go2_hopping"

    # 方策ネットワーク (Actor / Critic) の設定
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],   # Actor (方策) の隠れ層
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],   # Critic (価値関数) の隠れ層
        activation="elu",
        obs_normalization=False,
    )

    # PPO 最適化設定
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,         # 探索を促すエントロピー係数
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,      # 学習率
        schedule="adaptive",       # 適応的学習率スケジューラ
        gamma=0.99,                # 割引率
        lam=0.95,                  # GAE lambda
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

