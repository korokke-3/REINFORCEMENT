import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize

# 各リンクのbase座標系における質量と重心位置（Go2の公称CAD値）
links = [
    {"name": "base",       "mass": 6.369, "pos": np.array([0.0, 0.0, 0.0])},
    {"name": "Head_upper", "mass": 0.001, "pos": np.array([0.28, 0.0, 0.05])},
    {"name": "FL_hip",     "mass": 0.678, "pos": np.array([0.1934, 0.0465, 0.0])},
    {"name": "FR_hip",     "mass": 0.678, "pos": np.array([0.1934, -0.0465, 0.0])},
    {"name": "RL_hip",     "mass": 0.678, "pos": np.array([-0.1934, 0.0465, 0.0])},
    {"name": "RR_hip",     "mass": 0.678, "pos": np.array([-0.1934, -0.0465, 0.0])},
]

# 順運動学（FK）で各リンクの位置を計算して、合成重心とRR足先位置のXYオフセットを最小化する
def compute_balance(pitch_deg, roll_deg, rr_thigh, rr_calf, fl_thigh, fr_thigh, rl_thigh):
    # 簡易モデルでの重心計算
    # 目的: CoM_xy == RR_foot_xy
    pass

