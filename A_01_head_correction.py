#各話者のEMA_combined_xyz を読み込む
#  ↓
#各フレームで
# ・UIを原点へ移動
# ・NDがZ軸方向になるよう回転
# ・NAがx=0かつy<0側になるよう回転し、
#EMA_aligned_xyz に保存
#各フレームで補正を行っているが、もともとのデータが0フレーム目のUI-ND-NAの三角形に
#全フレームの三角形を合わせているため、0フレーム目の補正を各フレームに適用してもあまり変わらないと思われる。

#さらに、001番発話の第1フレームから変換行列を作り、その変換を口蓋データにも適用
#  ↓
#palate_aligned に保存

import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

# =====================================================================
#  1. 設定項目
# =====================================================================
SPEAKERS = ["M0102", "M0201", "W0401", "W0901"]
BASE_DIR = "Data"

ND_IDX, NA_IDX, UI_IDX = 0, 3, 6

# =====================================================================
#  2. 座標変換コア関数 (修正版)
# =====================================================================
def get_transform_matrices(frame_data):
    nd_pos = frame_data[ND_IDX : ND_IDX+3]
    na_pos = frame_data[NA_IDX : NA_IDX+3]
    ui_pos = frame_data[UI_IDX : UI_IDX+3]
    
    # --- Step 1: UI原点移動 ---
    v_nd = nd_pos - ui_pos
    v_na = na_pos - ui_pos

    # --- Step 2: NDを Z軸方向に向ける回転行列 R1 ---
    v_nd_unit = v_nd / np.linalg.norm(v_nd)
    z_target = np.array([0.0, 0.0, 1.0])
    
    rot_axis = np.cross(v_nd_unit, z_target)
    axis_norm = np.linalg.norm(rot_axis)
    
    if axis_norm > 1e-8:
        rot_axis = rot_axis / axis_norm
        angle1 = np.arccos(np.clip(np.dot(v_nd_unit, z_target), -1.0, 1.0))
        
        K = np.array([
            [0, -rot_axis[2], rot_axis[1]],
            [rot_axis[2], 0, -rot_axis[0]],
            [-rot_axis[1], rot_axis[0], 0]
        ])
        R1 = np.eye(3) + np.sin(angle1) * K + (1 - np.cos(angle1)) * (K @ K)
    else:
        R1 = np.eye(3)

    # --- Step 3: NAのxを 0, yをマイナス(y < 0) に向ける回転行列 R2 (修正部分) ---
    v_na_step2 = R1 @ v_na  # Step 2適用後のNA位置
    x_na, y_na = v_na_step2[0], v_na_step2[1]
    r = np.hypot(x_na, y_na)  # XY平面上のNAまでの距離
    
    if r > 1e-8:
        # NA(x, y) を (0, -r) に回転させる三角関数の値を直接定義
        cos_a2 = -y_na / r
        sin_a2 = -x_na / r
    else:
        cos_a2, sin_a2 = 1.0, 0.0
        
    R2 = np.array([
        [cos_a2, -sin_a2, 0],
        [sin_a2,  cos_a2, 0],
        [     0,       0, 1]
    ])

    return ui_pos, R1, R2


def apply_transformation(points, ui_pos, R1, R2):
    pts_shifted = points - ui_pos
    pts_r1 = (R1 @ pts_shifted.T).T
    pts_r2 = (R2 @ pts_r1.T).T
    
    return pts_r2


# =====================================================================
#  3. メイン処理ループ
# =====================================================================
print("🚀 修正版位置合わせ処理を開始します...\n")

for spk in SPEAKERS:
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"👤 話者: {spk}")
    
    ema_input_dir = os.path.join(BASE_DIR, spk, "EMA", "EMA_combined_xyz")
    ema_output_dir = os.path.join(BASE_DIR, spk, "EMA", "EMA_aligned_xyz")
    os.makedirs(ema_output_dir, exist_ok=True)
    
    csv_files = sorted(glob.glob(os.path.join(ema_input_dir, "*.csv")))
    if not csv_files:
        print(f"⚠️ {spk}: EMAデータが見つかりません。")
        continue

    # A. EMAデータの処理
    print(f"🎬 EMAデータ変換中 ({len(csv_files)} ファイル)...")
    for file_path in tqdm(csv_files, leave=False):
        df = pd.read_csv(file_path, header=None)
        raw_data = df.values
        num_frames = len(raw_data)
        
        aligned_data = np.zeros_like(raw_data)
        
        for f in range(num_frames):
            frame_raw = raw_data[f]
            sensors = frame_raw.reshape(-1, 3)
            
            ui_pos, R1, R2 = get_transform_matrices(frame_raw)
            sensors_aligned = apply_transformation(sensors, ui_pos, R1, R2)
            
            aligned_data[f] = sensors_aligned.flatten()
            
        out_path = os.path.join(ema_output_dir, os.path.basename(file_path))
        pd.DataFrame(aligned_data).to_csv(out_path, index=False, header=False)

    # B. 口蓋データの処理 (*_palate_xyz.data 対象)
    ref_ema_file = os.path.join(ema_input_dir, "001_combined.csv")
    if not os.path.exists(ref_ema_file):
        candidates = glob.glob(os.path.join(ema_input_dir, "001*.csv"))
        ref_ema_file = candidates[0] if candidates else csv_files[0]

    ref_df = pd.read_csv(ref_ema_file, header=None)
    frame0_data = ref_df.values[0]
    ui_pos_ref, R1_ref, R2_ref = get_transform_matrices(frame0_data)

    palate_input_dir = os.path.join(BASE_DIR, spk, "palate")
    palate_output_dir = os.path.join(BASE_DIR, spk, "palate_aligned")
    
    palate_files = sorted(glob.glob(os.path.join(palate_input_dir, "*_palate_xyz.data")))
                   
    if palate_files:
        os.makedirs(palate_output_dir, exist_ok=True)
        print(f"🏛️ 口蓋データ変換中 ({len(palate_files)} ファイル)...")
        
        for p_file in palate_files:
            palate_pts = np.loadtxt(p_file)
            aligned_palate = apply_transformation(palate_pts, ui_pos_ref, R1_ref, R2_ref)
            
            out_p_path = os.path.join(palate_output_dir, os.path.basename(p_file))
            np.savetxt(out_p_path, aligned_palate, fmt='%.6f', delimiter=' ')

print("\n🎉 修正完了しました！データを出力し直してご確認ください。")
