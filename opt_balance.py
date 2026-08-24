import numpy as np
from scipy.spatial.transform import Rotation as R

# inspect_go2_dynamics から得られた base座標系での各パーツ相対位置 (デフォルト屈曲時)
# (X, Y, Z, Mass)
body_data = [
    ("base",        0.000,  0.000,  0.000, 6.369),
    ("FL_hip",      0.193,  0.047,  0.000, 0.678),
    ("FR_hip",      0.193, -0.047,  0.000, 0.678),
    ("Head_upper",  0.280,  0.000,  0.050, 0.001),
    ("RL_hip",     -0.193,  0.047,  0.000, 0.678),
    ("RR_hip",     -0.193, -0.047,  0.000, 0.678),
    ("FL_thigh",    0.291,  0.210,  0.200, 1.152),
    ("FR_thigh",    0.511, -0.010,  0.200, 1.152),
    ("RL_thigh",   -0.216,  0.106, -0.100, 1.152),
    ("RR_thigh",   -0.193, -0.114, -0.050, 1.152),
    ("FL_calf",     0.291,  0.210,  0.300, 0.154),
    ("FR_calf",     0.511, -0.010,  0.250, 0.154),
    ("RL_calf",    -0.216,  0.106, -0.150, 0.154),
    ("RR_calf",    -0.100, -0.114, -0.150, 0.154),
    ("FL_foot",     0.291,  0.210,  0.415, 0.040),
    ("FR_foot",     0.511, -0.010,  0.286, 0.040),
    ("RL_foot",    -0.216,  0.106, -0.211, 0.040),
]

# RR足先の位置 (屈曲角度によって可変)
# RR_hip: (-0.1934, -0.0465, 0.0)
# Thigh length = 0.213m, Calf length = 0.213m
# RR_hip_rot = roll=0, RR_thigh = th, RR_calf = cal

def forward_kinematics_rr(thigh_angle, calf_angle):
    hip_pos = np.array([-0.1934, -0.114, 0.0]) # hip offset including abduction
    # 2D in X-Z plane relative to hip
    # thigh: vector from hip to knee
    knee_x = -0.213 * np.sin(thigh_angle)
    knee_z = -0.213 * np.cos(thigh_angle)
    
    # calf: vector from knee to foot
    foot_x = knee_x - 0.213 * np.sin(thigh_angle + calf_angle)
    foot_z = knee_z - 0.213 * np.cos(thigh_angle + calf_angle)
    
    foot_pos_base = hip_pos + np.array([foot_x, 0.0, foot_z])
    return foot_pos_base

# ベース基準の合成CoM
total_mass = sum(b[4] for b in body_data) + 1.152 + 0.154 + 0.040 # RR links
com_base = np.zeros(3)
for b in body_data:
    com_base += b[4] * np.array([b[1], b[2], b[3]])
com_base /= total_mass

print(f"Base-frame CoM (approx): {com_base}")

# 最適な Roll, Pitch 角を探索 (ワールド座標系で CoM_xy == RR_foot_xy)
best_diff = 1e9
best_angles = None

for roll_deg in np.linspace(-45, 45, 181):
    for pitch_deg in np.linspace(-70, 0, 141):
        r = R.from_euler('xyz', [roll_deg, pitch_deg, 0], degrees=True)
        rot_mat = r.as_matrix()
        
        # RR足先位置 (thigh=0.7, calf=-2.0)
        foot_base = forward_kinematics_rr(0.7, -2.0)
        
        # ワールド座標系での相対位置
        com_w = rot_mat @ com_base
        foot_w = rot_mat @ foot_base
        
        diff_xy = np.linalg.norm(com_w[:2] - foot_w[:2])
        if diff_xy < best_diff:
            best_diff = diff_xy
            best_angles = (roll_deg, pitch_deg)

print(f"Optimal Orientation: Roll = {best_angles[0]:.2f} deg, Pitch = {best_angles[1]:.2f} deg")
print(f"Residual XY Offset = {best_diff*1000:.2f} mm")
