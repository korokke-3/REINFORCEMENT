import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize

# Go2のリンク質量とオフセット（URDF仕様）
BASE_MASS = 14.47  # kg
RR_HIP_POS = np.array([-0.1934, -0.0465, 0.0])
RR_THIGH_LEN = 0.213
RR_CALF_LEN = 0.213

# 関節角度からRR足先およびRR膝の位置を計算
def forward_kinematics(q_hip, q_thigh, q_calf):
    # Hip joint rotation (X axis)
    R_hip = R.from_euler('x', q_hip).as_matrix()
    p_thigh_joint = RR_HIP_POS + R_hip @ np.array([0, -0.0955, 0])
    
    # Thigh joint rotation (Y axis)
    R_thigh = R_hip @ R.from_euler('y', q_thigh).as_matrix()
    p_knee_joint = p_thigh_joint + R_thigh @ np.array([0, 0, -RR_THIGH_LEN])
    
    # Calf joint rotation (Y axis)
    R_calf = R_thigh @ R.from_euler('y', q_calf).as_matrix()
    p_foot = p_knee_joint + R_calf @ np.array([0, 0, -RR_CALF_LEN])
    
    return p_thigh_joint, p_knee_joint, p_foot

# 重心(CoM)が膝と足先を結ぶ直線（地面上の線分）の真上にあるかを最適化
def loss_function(params):
    roll, pitch, q_hip, q_thigh, q_calf = params
    
    # 姿勢回転
    R_base = R.from_euler('xyz', [roll, pitch, 0], degrees=True).as_matrix()
    
    # FK in body frame
    _, p_knee, p_foot = forward_kinematics(q_hip, q_thigh, q_calf)
    
    # World frame positions
    p_knee_w = R_base @ p_knee
    p_foot_w = R_base @ p_foot
    
    # 1. 膝と足先が同じ高さ（地面 Z=0 に平行に接地）
    z_diff = np.square(p_knee_w[2] - p_foot_w[2])
    
    # 2. 重心(0,0,Z)のXY座標が、膝と足先を結ぶ線分(p_knee_w -> p_foot_w)の真上にあるか
    # 線分上の最近傍点との距離
    line_vec = p_foot_w[:2] - p_knee_w[:2]
    line_len_sq = np.dot(line_vec, line_vec)
    if line_len_sq < 1e-6:
        dist_to_line = 10.0
    else:
        # 重心XY=(0,0)から線分への射影パラメータ t
        t = np.clip(np.dot(-p_knee_w[:2], line_vec) / line_len_sq, 0.0, 1.0)
        proj_point = p_knee_w[:2] + t * line_vec
        dist_to_line = np.linalg.norm(proj_point)
    
    return dist_to_line * 100.0 + z_diff * 50.0

res = minimize(loss_function, [25.0, -25.0, 0.0, 0.8, -2.5], bounds=[(10, 45), (-45, -10), (-0.5, 0.5), (0.2, 1.5), (-2.7, -1.5)])

opt_roll, opt_pitch, opt_q_hip, opt_q_thigh, opt_q_calf = res.x
print(f"Optimal Knee+Foot Stance Alignment:")
print(f"  Roll       : {opt_roll:+.2f} deg")
print(f"  Pitch      : {opt_pitch:+.2f} deg")
print(f"  RR_hip     : {opt_q_hip:+.2f} rad")
print(f"  RR_thigh   : {opt_q_thigh:+.2f} rad")
print(f"  RR_calf    : {opt_q_calf:+.2f} rad")
print(f"  Alignment Loss: {res.fun:.4f}")
