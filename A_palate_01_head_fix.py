#実験室のパソコンに入っていたスクリプトをPython形式にしたもの
#一人の話者の全発話の全フレームのUI-ND-NAの三角形を001発話の最初のフレームの三角形に合わせる。

import os
import numpy as np

# =====================================================================
#  1. 設定項目
# =====================================================================
BASE_DIR = "20250304"
D_DATE = "20250304"

# 💡 基準（テンプレート枠）にするフォルダ
STANDARD_FOLDER = f"S{D_DATE}_D{D_DATE}20001mov"
STANDARD_DATA_NO = 0  # 基準にするフレーム番号 (0コマ目)

# 💡 処理対象となる全503個のIDリスト
ID_LIST = (
    list(range(10051, 10250 + 1)) +
    list(range(10301, 10503 + 1)) +
    list(range(10701, 10703 + 1)) +
    list(range(20001, 20050 + 1)) +
    list(range(30251, 30300 + 1))
)

FILE_NO = 0               
HF_FILE_NAME_HEAD = "hf"  
FILE_NAME_AFTER_CH_NO = "POS_angle"

START_CH_NO = 1
END_CH_NO = 12
WITHOUT_CH = 2  # 2chは除外

# 基準チャンネル (ch3=原点, ch1=方向軸, ch5=平面ロック)
REF_CH_NO_1 = 12
REF_CH_NO_2 = 5
REF_CH_NO_3 = 1

POS_AXIS_NUM = 3
DATA_AXIS_NUM = 5

# =====================================================================
#  2. 幾何学計算ユーティリティ関数
# =====================================================================
def make_direct_unit_vector(a):
    dist = np.linalg.norm(a)
    return a / dist if dist != 0 else np.zeros_like(a)

def make_direct_unit_vector2(angle):
    duv = np.zeros(3)
    duv[0] = np.cos(angle[0]) * np.cos(angle[1])
    duv[1] = np.cos(angle[0]) * np.sin(angle[1])
    duv[2] = -np.sin(angle[0])
    return duv

def make_theta_vector_from_direct_unit_vector(duv):
    theta0 = np.arcsin(-duv[2])
    cos_t0 = np.cos(theta0)
    cos_val = duv[0] / cos_t0 if abs(cos_t0) > 1e-9 else 1.0
    cos_val = np.clip(cos_val, -1.0, 1.0)
    if duv[1] >= 0.0:
        theta1 = np.arccos(cos_val)
    else:
        theta1 = -np.arccos(cos_val)
    return np.array([theta0, theta1])

def make_rotate_matrix(theta):
    t0, t1 = theta[0], theta[1]
    R = np.zeros((3, 3))
    R[0, 0] = np.cos(t0) * np.cos(t1)
    R[0, 1] = -np.sin(t1)
    R[0, 2] = np.sin(t0) * np.cos(t1)
    R[1, 0] = np.cos(t0) * np.sin(t1)
    R[1, 1] = np.cos(t1)
    R[1, 2] = np.sin(t0) * np.sin(t1)
    R[2, 0] = -np.sin(t0)
    R[2, 1] = 0.0
    R[2, 2] = np.cos(t0)
    return R

def make_inv_rotate_matrix(theta):
    t0, t1 = theta[0], theta[1]
    R_inv = np.zeros((3, 3))
    R_inv[0, 0] = np.cos(t0) * np.cos(t1)
    R_inv[0, 1] = np.cos(t0) * np.sin(t1)
    R_inv[0, 2] = -np.sin(t0)
    R_inv[1, 0] = -np.sin(t1)
    R_inv[1, 1] = np.cos(t1)
    R_inv[1, 2] = 0.0
    R_inv[2, 0] = np.sin(t0) * np.cos(t1)
    R_inv[2, 1] = np.sin(t0) * np.sin(t1)
    R_inv[2, 2] = np.cos(t0)
    return R_inv

def make_coefficient_of_mapping_plane2(rel_pos_ch2, rel_pos_ch3):
    A = np.array([
        [rel_pos_ch2[0], rel_pos_ch2[1]],
        [rel_pos_ch3[0], rel_pos_ch3[1]]
    ])
    B = np.array([-rel_pos_ch2[2], -rel_pos_ch3[2]])
    c_xy = np.linalg.solve(A, B)
    return np.array([c_xy[0], c_xy[1], 1.0])

def make_out_pos_ref3(pos_ref_ch_no_2, pos_ref_ch_no_3, coefficient_of_mapping_plane):
    dist102 = np.sum(pos_ref_ch_no_2**2)
    dist202 = np.sum(pos_ref_ch_no_3**2)
    relative_ref32 = pos_ref_ch_no_3 - pos_ref_ch_no_2
    dist212 = np.sum(relative_ref32**2)

    a, b, c = coefficient_of_mapping_plane
    x1, y1, z1 = pos_ref_ch_no_2
    
    d = c * x1 - a * z1
    e = b * x1 - a * y1
    f = c * y1 - b * z1
    
    g = c * d + b * e
    h = dist202 - dist212 + dist102
    p = d**2 + e**2 + f**2

    sqrt_val = np.sqrt(max(0.0, (g**2) * (h**2) + p * (4.0 * (f**2) * dist202 - (b**2 + c**2) * (h**2))))

    cand1 = np.array([
        ((h * g) + sqrt_val) / (2.0 * p),
        (c * h - 2.0 * d * (((h * g) + sqrt_val) / (2.0 * p))) / (2.0 * f),
        (2.0 * e * (((h * g) + sqrt_val) / (2.0 * p)) - b * h) / (2.0 * f)
    ])
    cand2 = np.array([
        ((h * g) - sqrt_val) / (2.0 * p),
        (c * h - 2.0 * d * (((h * g) - sqrt_val) / (2.0 * p))) / (2.0 * f),
        (2.0 * e * (((h * g) - sqrt_val) / (2.0 * p)) - b * h) / (2.0 * f)
    ])

    return cand1 if np.sum(pos_ref_ch_no_3 * cand1) >= np.sum(pos_ref_ch_no_3 * cand2) else cand2

def calc_rotate_angle_around_axis_a(rotated_pos, org_pos, a_vec):
    l, m, n = a_vec
    x1, y1, z1 = org_pos
    x2, y2 = rotated_pos[0], rotated_pos[1]
    b = l * x1 + m * y1 + n * z1
    e = (n * x1 - l * z1) * (x1 - l * b) - (m * z1 - n * y1) * (y1 - m * b)
    
    # ゼロ除算防止
    if abs(e) < 1e-12: e = 1e-12 
        
    costheta = ((n * x1 - l * z1) * (x2 - l * b) - (m * z1 - n * y1) * (y2 - m * b)) / e
    sintheta = (x1 * y2 - x2 * y1 + ((x2 - x1) * m + (y1 - y2) * l) * b) / e
    return np.arccos(np.clip(costheta, -1.0, 1.0)) if sintheta >= 0.0 else -np.arccos(np.clip(costheta, -1.0, 1.0))

def make_rotate_pos_around_axis_a(org_pos, a_vec, theta):
    R = np.zeros((3, 3))
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    a0, a1, a2 = a_vec
    R[0, 0] = a0**2 + (1.0 - a0**2) * cos_t
    R[0, 1] = a0 * a1 * (1.0 - cos_t) - a2 * sin_t
    R[0, 2] = a0 * a2 * (1.0 - cos_t) + a1 * sin_t
    R[1, 0] = a0 * a1 * (1.0 - cos_t) + a2 * sin_t
    R[1, 1] = a1**2 + (1.0 - a1**2) * cos_t
    R[1, 2] = a1 * a2 * (1.0 - cos_t) - a0 * sin_t
    R[2, 0] = a0 * a2 * (1.0 - cos_t) - a1 * sin_t
    R[2, 1] = a1 * a2 * (1.0 - cos_t) + a0 * sin_t
    R[2, 2] = a2**2 + (1.0 - a2**2) * cos_t
    return np.dot(R, org_pos)

# =====================================================================
#  3. メイン処理部
# =====================================================================
print("⏳ 1. 基準テンプレート（20001mov）の読み込み中...")
std_org_pos_data = {}

for i_ch in range(START_CH_NO, END_CH_NO + 1):
    if i_ch == WITHOUT_CH:
        continue
    std_file = f"{BASE_DIR}/{STANDARD_FOLDER}/{STANDARD_FOLDER}_{FILE_NO}_ch{i_ch}_{FILE_NAME_AFTER_CH_NO}.data"
    
    if not os.path.exists(std_file):
        print(f"❌ 基準ファイルが見つかりません: {std_file}")
        print("処理を中断します。20001movのデータが存在することを確認してください。")
        exit()
    
    raw_std_data = np.loadtxt(std_file)
    std_org_pos_data[i_ch] = raw_std_data[STANDARD_DATA_NO, :POS_AXIS_NUM]

# 基準空間における初期幾何学パラメータの計算
org_relative_pos_ch2 = std_org_pos_data[REF_CH_NO_2] - std_org_pos_data[REF_CH_NO_1]
org_relative_pos_ch3 = std_org_pos_data[REF_CH_NO_3] - std_org_pos_data[REF_CH_NO_1]

dir_org_relative_pos12 = make_direct_unit_vector(org_relative_pos_ch2)
theta_org12 = make_theta_vector_from_direct_unit_vector(dir_org_relative_pos12)
rotate_matrix1 = make_rotate_matrix(theta_org12)
coefficient_of_mapping_plane = make_coefficient_of_mapping_plane2(org_relative_pos_ch2, org_relative_pos_ch3)

print(f"⏳ 2. 全{len(ID_LIST)}フォルダの頭部固定を一括処理します...")

success_count = 0
missing_count = 0

for idx, target_id in enumerate(ID_LIST):
    target_folder = f"S{D_DATE}_D{D_DATE}{target_id}mov"
    folder_path = f"{BASE_DIR}/{target_folder}"
    
    if not os.path.exists(folder_path):
        missing_count += 1
        continue
        
    print(f"[{idx+1}/{len(ID_LIST)}] 🎬 処理中: {target_folder} ...")
    
    org_pos_data = {}
    err_data = {}
    out_pos_data = {}
    total_frames = 0
    valid_folder = True
    
    # データの読み込み
    for i_ch in range(START_CH_NO, END_CH_NO + 1):
        if i_ch == WITHOUT_CH:
            continue
            
        target_file = f"{folder_path}/{target_folder}_{FILE_NO}_ch{i_ch}_{FILE_NAME_AFTER_CH_NO}.data"
        if not os.path.exists(target_file):
            valid_folder = False
            break
            
        raw_data = np.loadtxt(target_file)
        org_pos_data[i_ch] = raw_data[:, :DATA_AXIS_NUM]
        err_data[i_ch] = raw_data[:, 5]
        total_frames = raw_data.shape[0]
        out_pos_data[i_ch] = np.zeros((total_frames, DATA_AXIS_NUM))
        
    if not valid_folder:
        print(f"   ⚠️ 一部のchファイルが不足しているためスキップします。")
        missing_count += 1
        continue

    # フレーム毎の幾何学変換
    theta_3_start = 0.0
    for i_datanum in range(total_frames):
        current_ch1 = org_pos_data[REF_CH_NO_2][i_datanum, :3] - org_pos_data[REF_CH_NO_1][i_datanum, :3]
        
        dir_relative_pos12 = make_direct_unit_vector(current_ch1)
        theta_new12 = make_theta_vector_from_direct_unit_vector(dir_relative_pos12)
        inv_rotate_matrix2 = make_inv_rotate_matrix(theta_new12)

        inter_mid_pos = {}
        for i_ch in range(START_CH_NO, END_CH_NO + 1):
            if i_ch == WITHOUT_CH: continue
            rel_pos = org_pos_data[i_ch][i_datanum, :3] - org_pos_data[REF_CH_NO_1][i_datanum, :3]
            inter_mid_pos[i_ch] = np.dot(rotate_matrix1, np.dot(inv_rotate_matrix2, rel_pos))

        out_pos_ref3 = make_out_pos_ref3(inter_mid_pos[REF_CH_NO_2], inter_mid_pos[REF_CH_NO_3], coefficient_of_mapping_plane)
        theta_3 = calc_rotate_angle_around_axis_a(out_pos_ref3, inter_mid_pos[REF_CH_NO_3], dir_org_relative_pos12)

        # 角度アンラップ処理
        if i_datanum == 0:
            theta_3_start = theta_3
        elif theta_3 - theta_3_start < -np.pi / 2:
            theta_3 += np.pi
            if theta_3 - theta_3_start < -np.pi / 2: theta_3 += np.pi
            if theta_3 - theta_3_start < -np.pi / 2: theta_3 += np.pi
        elif theta_3 - theta_3_start > np.pi / 2:
            theta_3 -= np.pi
            if theta_3 - theta_3_start > np.pi / 2: theta_3 -= np.pi
            if theta_3 - theta_3_start > np.pi / 2: theta_3 -= np.pi

        if theta_3 > np.pi: theta_3 -= 2.0 * np.pi
        if theta_3 < -np.pi: theta_3 += 2.0 * np.pi
        theta_3_start = theta_3
       
        # 全チャンネルへ補正適用
        for i_ch in range(START_CH_NO, END_CH_NO + 1):
            if i_ch == WITHOUT_CH: continue
            
            # 座標補正
            out_pos_relative = make_rotate_pos_around_axis_a(inter_mid_pos[i_ch], dir_org_relative_pos12, theta_3)
            out_pos_data[i_ch][i_datanum, :3] = out_pos_relative + std_org_pos_data[REF_CH_NO_1]
            
            # 角度補正
            receiver_angle = org_pos_data[i_ch][i_datanum, 3:5]
            receiver_dir_vector = make_direct_unit_vector2(receiver_angle)
            
            tmp_dir = np.dot(inv_rotate_matrix2, receiver_dir_vector)
            tmp_dir2 = np.dot(rotate_matrix1, tmp_dir)
            out_receiver_dir_vector = make_rotate_pos_around_axis_a(tmp_dir2, dir_org_relative_pos12, theta_3)
            out_pos_data[i_ch][i_datanum, 3:5] = make_theta_vector_from_direct_unit_vector(out_receiver_dir_vector)

    # データの書き出し
    for i_ch in range(START_CH_NO, END_CH_NO + 1):
        if i_ch == WITHOUT_CH: continue
        output_file_path = f"{folder_path}/{HF_FILE_NAME_HEAD}{target_folder}_{FILE_NO}_ch{i_ch}_{FILE_NAME_AFTER_CH_NO}.data"
        
        combined_output = np.hstack([out_pos_data[i_ch], err_data[i_ch].reshape(-1, 1)])
        fmt_list = ['%lf', '%lf', '%lf', '%lf', '%lf', '%e']
        np.savetxt(output_file_path, combined_output, delimiter='\t', fmt=fmt_list)
        
    success_count += 1

print("\n==================================================")
print("🎉 すべての頭部固定一括処理が完了しました！")
print("==================================================")
print(f"➔ 成功したフォルダ数: {success_count} / {len(ID_LIST)}")
print(f"➔ スキップされた数  : {missing_count}")
print("==================================================\n")
