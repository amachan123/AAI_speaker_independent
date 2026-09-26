#W0901については口蓋データがうまくとれていなかったため、T1,T2,T3から口蓋形状を推定した。
#このスクリプトでは、全EMA発話からT1・T2・T3の軌跡を集めて、
#Y方向を2 mmごとに区切り、その区間で最も高いZ位置を取り出し、「舌が到達した上限形状」のような点列を作る

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# =====================================================================
#  1. 設定項目
# =====================================================================
SPEAKER = "W0901"
BASE_DIR = "Data"

# 読み込み先と保存先
INPUT_DIR = os.path.join(BASE_DIR, SPEAKER, "EMA", "EMA_aligned_yz")
OUTPUT_DIR = os.path.join(BASE_DIR, SPEAKER, "palate2")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_DATA = os.path.join(OUTPUT_DIR, "extracted_max_points_2.0_aligned.data")
OUTPUT_PLOT = os.path.join(OUTPUT_DIR, "extracted_max_points_plot_2.0_aligned.png")

BIN_STEP_MM = 2.0

# =====================================================================
#  2. データの読み込みと T1, T2, T3 の抽出
# =====================================================================
print(f"📂 フォルダ {INPUT_DIR} からCSVデータを探索します...")

csv_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.csv")))

if not csv_files:
    print("❌ CSVファイルが見つかりませんでした。")
    exit()

loaded_data = []

# T1, T2, T3 の列インデックス (0始まり)
# 12: T1_y, 13: T1_z
# 14: T2_y, 15: T2_z
# 16: T3_y, 17: T3_z
T1_COLS = [12, 13]
T2_COLS = [14, 15]
T3_COLS = [16, 17]

print(f"🔄 {len(csv_files)} ファイルから舌データ(T1, T2, T3)を抽出中...")
for file_path in tqdm(csv_files):
    try:
        df = pd.read_csv(file_path, header=None)
        raw_data = df.values
        
        # T1, T2, T3 の(Y, Z)を取り出して縦に積むためのリストに追加
        t1_data = raw_data[:, T1_COLS]
        t2_data = raw_data[:, T2_COLS]
        t3_data = raw_data[:, T3_COLS]
        
        loaded_data.extend([t1_data, t2_data, t3_data])
    except Exception as e:
        print(f"⚠️ {file_path} の読み込みスキップ: {e}")

# リストに溜まったすべての点群を1つの巨大なNumPy配列に結合
tongue_cloud = np.vstack(loaded_data)

# Y座標とZ座標に分割
y_as_x = tongue_cloud[:, 0]
z_raw = tongue_cloud[:, 1]

print(f"➔ 舌データ抽出・結合完了: 全 {len(tongue_cloud):,} 点")

# =====================================================================
#  3. 区間分割（Binning）による最大値（天井）抽出
# =====================================================================
print(f"🔍 区間分割アルゴリズムを実行中... (区間幅: {BIN_STEP_MM}mm)")

min_y = np.min(y_as_x)
max_y = np.max(y_as_x)

bins = np.arange(min_y, max_y + BIN_STEP_MM, BIN_STEP_MM)
digitized = np.digitize(y_as_x, bins)

bin_extracted_y = []
bin_extracted_z = []

valid_bins = [i for i in range(1, len(bins)) if np.any(digitized == i)]

if valid_bins:
    for i in valid_bins:
        bin_mask = digitized == i
        bin_y = y_as_x[bin_mask]
        bin_z = z_raw[bin_mask]
        
        # 区間内のZの最大値と、その値を持つインデックスを取得
        max_idx = np.argmax(bin_z)
        
        # 最大Zとその時の実際のY座標を抽出
        actual_max_y = bin_y[max_idx]
        actual_max_z = bin_z[max_idx]
        
        bin_extracted_y.append(actual_max_y)
        bin_extracted_z.append(actual_max_z)

bin_extracted_y = np.array(bin_extracted_y)
bin_extracted_z = np.array(bin_extracted_z)

# 両端の生データを取得
idx_min = np.argmin(y_as_x)  
idx_max = np.argmax(y_as_x)  
edge_y = np.array([y_as_x[idx_min], y_as_x[idx_max]])
edge_z = np.array([z_raw[idx_min], z_raw[idx_max]])

# =====================================================================
#  4. 抽出した点の整理と保存
# =====================================================================
# ビン代表点と両端の生データをすべて合流させる
all_y = np.concatenate([bin_extracted_y, edge_y])
all_z = np.concatenate([bin_extracted_z, edge_z])

# Y座標の昇順に並び替え
sort_idx = np.argsort(all_y)
all_y = all_y[sort_idx]
all_z = all_z[sort_idx]

# 抽出した「点」のデータをそのまま保存
np.savetxt(OUTPUT_DATA, np.column_stack((all_y, all_z)), delimiter='\t', fmt='%.6f')
print(f"💾 抽出した最大値ポイントを保存しました: {OUTPUT_DATA}")

# =====================================================================
#  5. グラフ描画
# =====================================================================
plt.figure(figsize=(10, 6))

# 点が多すぎる場合描画が重くなるので、間引いてプロット
plot_stride = max(1, len(y_as_x) // 100000) 
plt.scatter(y_as_x[::plot_stride], z_raw[::plot_stride], color='gray', alpha=0.05, s=1, label='Raw Tongue Data (T1, T2, T3)')

# 抽出した最大値の点を分かりやすく赤色で強調してプロット
plt.scatter(all_y, all_z, color='red', s=35, edgecolor='black', zorder=5, label='Extracted Max Points')

plt.xlabel('Y Position [mm]')
plt.ylabel('Z Position [mm]')
plt.title(f'Extracted Palate Points with Binning (Speaker: {SPEAKER}, Bin: {BIN_STEP_MM}mm)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.savefig(OUTPUT_PLOT, dpi=150)
plt.close()
print(f"🎨 確認用グラフを保存しました: {OUTPUT_PLOT}")
print("🎉 処理が完了しました！")
