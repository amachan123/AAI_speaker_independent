#頭部補正がうまくできているかチェック

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =====================================================================
#  1. 設定項目
# =====================================================================
SPEAKERS = ["M0102", "M0201", "W0401", "W0901"]
BASE_DIR = "Data"

# センサーごとの表示色定義 (9個分)
SENSOR_COLORS = [
    "#f10f0f", # 0: ND (赤)
    '#ff7f0e', # 1: NA (オレンジ)
    "#0bec0b", # 2: UI (緑)
    "#1394f0", # 3: LI (青)
    "#850cf7", # 4: UL (紫)
    '#8c564b', # 5: LL (茶)
    "#f00eac", # 6: TT (ピンク)
    "#25cad6e0", # 7: TM (グレー)
    "#f1f113", # 8: TB (黄緑)
]

# =====================================================================
#  2. 1フレーム目の画像出力処理
# =====================================================================
print("📸 1フレーム目の画像出力処理を開始します...\n")

for spk in SPEAKERS:
    print(f"👤 話者: {spk} の画像を作成中...")
    
    # パスの設定
    ema_dir = os.path.join(BASE_DIR, spk, "EMA", "EMA_aligned_xyz")
    palate_dir = os.path.join(BASE_DIR, spk, "palate_aligned")
    output_dir = "Data/Visualization_Images"
    os.makedirs(output_dir, exist_ok=True)
    
    # 対象の EMA CSV ファイル
    ema_file = os.path.join(ema_dir, "001_combined.csv")
    if not os.path.exists(ema_file):
        candidates = glob.glob(os.path.join(ema_dir, "001*.csv"))
        if candidates:
            ema_file = candidates[0]
        else:
            print(f"⚠️ {spk}: 001_combined.csv が見つかりません。スキップします。")
            continue

    # 口蓋データの読み込み
    palate_files = sorted(glob.glob(os.path.join(palate_dir, "*.data")))
    palate_pts_list = []
    for pf in palate_files:
        try:
            pts = np.loadtxt(pf)
            if pts.ndim == 2 and pts.shape[1] == 3:
                palate_pts_list.append(pts)
        except Exception as e:
            pass
            
    if palate_pts_list:
        palate_pts = np.vstack(palate_pts_list)
    else:
        palate_pts = np.empty((0, 3))

    # EMA データの読み込み
    df = pd.read_csv(ema_file, header=None)
    raw_data = df.values
    
    # -----------------------------------------------------------------
    # 全フレームから表示範囲 (Axis Limits) を計算
    # ※他の画像や後で動画にした時と縮尺がズレないようにするため
    # -----------------------------------------------------------------
    all_x = raw_data[:, 0::3].flatten()
    all_y = raw_data[:, 1::3].flatten()
    all_z = raw_data[:, 2::3].flatten()
    
    if len(palate_pts) > 0:
        all_x = np.concatenate([all_x, palate_pts[:, 0]])
        all_y = np.concatenate([all_y, palate_pts[:, 1]])
        all_z = np.concatenate([all_z, palate_pts[:, 2]])
        
    margin = 5.0
    xlim_y = (all_y.min() - margin, all_y.max() + margin)
    xlim_x = (all_x.min() - margin, all_x.max() + margin)
    ylim_z = (all_z.min() - margin, all_z.max() + margin)

    # 1フレーム目（インデックス0）のデータを抽出
    frame_idx = 0
    frame_sensors = raw_data[frame_idx].reshape(-1, 3)

    out_img_path = os.path.join(output_dir, f"{spk}_001_frame1.png")
    
    # 描画キャンバスの初期化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), dpi=150)
    fig.subplots_adjust(top=0.85, bottom=0.15, left=0.1, right=0.95, wspace=0.3)

    # --- 左グラフ: Y-Z 平面 (矢状面) ---
    if len(palate_pts) > 0:
        ax1.scatter(palate_pts[:, 1], palate_pts[:, 2], c='gray', s=3, alpha=0.3, label='Palate')
        
    for i in range(len(frame_sensors)):
        ax1.scatter(frame_sensors[i, 1], frame_sensors[i, 2], 
                    c=SENSOR_COLORS[i % len(SENSOR_COLORS)], s=40, zorder=5)
        
    ax1.set_title("Sagittal View (Y-Z Plane)")
    ax1.set_xlabel("Y: Forward/Backward (mm)")
    ax1.set_ylabel("Z: Height (mm)")
    ax1.set_xlim(xlim_y)
    ax1.set_ylim(ylim_z)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # --- 右グラフ: X-Z 平面 (前頭面) ---
    if len(palate_pts) > 0:
        ax2.scatter(palate_pts[:, 0], palate_pts[:, 2], c='gray', s=3, alpha=0.3, label='Palate')
        
    for i in range(len(frame_sensors)):
        ax2.scatter(frame_sensors[i, 0], frame_sensors[i, 2], 
                    c=SENSOR_COLORS[i % len(SENSOR_COLORS)], s=40, zorder=5)
        
    ax2.set_title("Coronal/Frontal View (X-Z Plane)")
    ax2.set_xlabel("X: Left/Right (mm)")
    ax2.set_ylabel("Z: Height (mm)")
    ax2.set_xlim(xlim_x)
    ax2.set_ylim(ylim_z)
    ax2.grid(True, linestyle='--', alpha=0.5)

    fig.suptitle(f"Speaker: {spk} | File: 001_combined.csv | Frame: 1", fontsize=14, fontweight='bold')

    # PNG画像として保存
    plt.savefig(out_img_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  ➔ 保存完了: {out_img_path}")

print("\n🎉 全話者の画像出力が完了しました！")
