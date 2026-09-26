#口蓋データを全部まとめる。
#x方向に1.5mmごとに区切る。
#各区間で最もZが高い点を選ぶ。
#その代表点列に3次スプラインを当てる。
#UI位置から後方まで指定した点数で滑らかな口蓋輪郭を出力。

import os
import pickle  # ★ 関数オブジェクトを保存するためのモジュールを追加
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

# =====================================================================
#  1. 設定項目
# =====================================================================
INPUT_FILES = [
    "Data/W0401/palate_aligned/1.05_701_palate.data",
    "Data/W0401/palate_aligned/1.05_702_palate.data",
    "Data/W0401/palate_aligned/1.05_703_palate.data"
]

UI_X_VALUE = 0.0  

OUTPUT_DATA_FILE = "Data/z_palate2/W0401_palate_1.05_1.5_deUI_10000.data"
OUTPUT_PLOT_FILE = "Data/W0401/palate/palate_contour_spline_plot_1.05_1.5_deUI.png"
# ★ 関数オブジェクトを保存するファイルのパスを追加 (.pkl 拡張子が一般的です)
OUTPUT_MODEL_FILE = "Data/W0401/palate/palate_contour_spline_1.05_1.5_deUI.pkl"

BIN_STEP_NUM = 1.5    # X軸の区間幅
SMOOTHING_FACTOR = 10.0  

print("🔄 スプライン平滑化による口蓋輪郭線の抽出を開始します...")

# =====================================================================
#  2. 複数データの読み込みと結合 (vstack)
# =====================================================================
loaded_data_list = []
trace_labels = []

for file_name in INPUT_FILES:
    if not os.path.exists(file_name):
        print(f"⚠️ ファイルが見つかりません: {file_name}")
        continue
        
    data = np.loadtxt(file_name)
    loaded_data_list.append(data)
    trace_labels.append(file_name)
    print(f"  📖 読み込み成功: {file_name} (データ数: {data.shape[0]})")

if len(loaded_data_list) == 0:
    print("❌ 有効なデータファイルが一つもありません。処理を終了します。")
    exit()

combined_data = np.vstack(loaded_data_list)
x_raw = combined_data[:, 0]
z_raw = combined_data[:, 1]

print(f"➔ 結合完了！ 総データポイント数: {combined_data.shape[0]}")

# =====================================================================
#  3. 区間（ビン）ごとの最大Z（一番高い天井）の抽出
# =====================================================================
min_val = np.min(x_raw)
max_val = np.max(x_raw)

bins = np.arange(min_val, max_val + BIN_STEP_NUM, BIN_STEP_NUM)
digitized = np.digitize(x_raw, bins)

bin_extracted_x = []
bin_extracted_z = []

valid_bins = [i for i in range(1, len(bins)) if np.any(digitized == i)]

for i in valid_bins:
    bin_mask = digitized == i
    bin_x = x_raw[bin_mask]
    bin_z = z_raw[bin_mask]
    
    max_idx = np.argmax(bin_z)
    
    actual_x = bin_x[max_idx]
    actual_z = bin_z[max_idx]
    
    bin_extracted_x.append(actual_x)
    bin_extracted_z.append(actual_z)

bin_extracted_x = np.array(bin_extracted_x)
bin_extracted_z = np.array(bin_extracted_z)

# 両端の生データを取得
idx_min = np.argmin(x_raw)  
idx_max = np.argmax(x_raw)  
edge_x = np.array([x_raw[idx_min], x_raw[idx_max]])
edge_z = np.array([z_raw[idx_min], z_raw[idx_max]])

# =====================================================================
#  4. スプライン平滑化の計算と保存
# =====================================================================
print(f"📐 スプライン平滑化を計算中... (平滑化係数 s={SMOOTHING_FACTOR})")

# 全ての点を合体
all_x = np.concatenate([bin_extracted_x, edge_x])
all_z = np.concatenate([bin_extracted_z, edge_z])

all_x, unique_indices = np.unique(all_x, return_index=True)
all_z = all_z[unique_indices]

# スプライン関数の作成
spline_func = UnivariateSpline(all_x, all_z, k=3, s=SMOOTHING_FACTOR)

# ★ 関数オブジェクトをそのままファイルとして保存（バイナリ書き込みモード 'wb'）
with open(OUTPUT_MODEL_FILE, 'wb') as f:
    pickle.dump(spline_func, f)
print(f"💾 関数モデルを保存しました: {OUTPUT_MODEL_FILE}")

# 点のデータ出力
#smooth_x = np.linspace(np.min(all_x), np.max(all_x), 3000)
smooth_x = np.linspace(UI_X_VALUE, np.max(all_x), 10000)
smooth_z = spline_func(smooth_x)

contour_data = np.column_stack((smooth_x, smooth_z))
np.savetxt(OUTPUT_DATA_FILE, contour_data, delimiter='\t', fmt='%.6f')
print(f"💾 輪郭線データを保存しました: {OUTPUT_DATA_FILE}")

# =====================================================================
#  5. 結果の視覚化（グラフ描画）
# =====================================================================
plt.figure(figsize=(10, 6))

colors = ['#2ca02c', '#ff7f0e', '#9467bd', '#1f77b4']
for idx, data in enumerate(loaded_data_list):
    color = colors[idx % len(colors)]
    plt.scatter(data[:, 0], data[:, 1], color=color, alpha=0.15, s=5, label=f'Raw: {trace_labels[idx]}')

plt.scatter(all_x, all_z, color='blue', s=25, edgecolor='white', zorder=4, label='Extracted Raw Points')

plt.plot(smooth_x, smooth_z, color='red', linewidth=2.5, zorder=5, label=f'Spline (s={SMOOTHING_FACTOR})')

plt.xlabel('X Position (Front-Back) [mm]')
plt.ylabel('Z Position (Height) [mm]')
plt.title(f'Palate Contour Extraction - Real Data Anchors (s={SMOOTHING_FACTOR})')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='lower right')

plt.savefig(OUTPUT_PLOT_FILE, dpi=150)
plt.close()
print(f"🎨 確認用グラフを保存しました: {OUTPUT_PLOT_FILE}")
print("🎉 すべての処理が完了しました！")
