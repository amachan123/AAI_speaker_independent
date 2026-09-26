

import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

# =====================================================================
#  1. 設定項目
# =====================================================================
INPUT_MAX_POINTS = "Data/W0901/palate2/extracted_max_points_2.0_aligned.data"
TONGUE_DIR = "Data/W0901/EMA/EMA_aligned_yz"
TONGUE_TARGET_COLS = (1, 2)  
SKIP_HEADER = 0

OUTPUT_BASE_DATA_FILE = "Data/W0901/palate2/palate_contour_spline_base.data"

OUTPUT_DATA_FILE = "Data/z_palate2/W0901_palate_1.0_1000.data"
OUTPUT_PLOT_FILE = "Data/W0901/palate2/palate_contour_spline_normal_offset_1.0_1000.png"
OUTPUT_MODEL_FILE = "Data/W0901/palate2/palate_contour_spline_normal_offset_1.0_1000.pkl"

# ★ 法線方向（外側）への押し出し量（ミリメートル）
OFFSET_MM = 1.0  

SMOOTHING_FACTOR = 5.0  

#zが小さい値はスキップする
ROWS_TO_DELETE =  [4] + [5] + [6]  + list(range(13, 18)) + list(range(19, 23)) +[25]
#ROWS_TO_DELETE = [4] + [5] + list(range(12, 17)) + [23]
#ROWS_TO_DELETE = [6] + list(range(11, 16)) +[29]
#OWS_TO_DELETE = [6] + list(range(12, 15))


print(f"🔄 フィルタリング ＆ 法線方向への拡大（+{OFFSET_MM}mm）を開始します...")

# =====================================================================
#  2. 生の舌データの探索と一括読み込み（背景描画用）
# =====================================================================
search_pattern = os.path.join(TONGUE_DIR, "**", "*_T[123].csv")
tongue_files = glob.glob(search_pattern, recursive=True)

loaded_t_points = []
if tongue_files:
    for file_path in tongue_files:
        try:
            data = np.loadtxt(file_path, delimiter=',', skiprows=SKIP_HEADER, usecols=TONGUE_TARGET_COLS)
            if data.size > 0:
                if data.ndim == 1:
                    data = data.reshape(1, -1)
                loaded_t_points.append(data)
        except Exception:
            pass
            
    if loaded_t_points:
        tongue_cloud = np.vstack(loaded_t_points)
        t_x = tongue_cloud[:, 0]
        t_z = tongue_cloud[:, 1]
        print(f"📖 生の舌データを読み込みました: 総データ数 {len(t_x):,} 点")
    else:
        print("⚠️ 舌データの読み込みに失敗しました。")
else:
    print("⚠️ 舌データのCSVが見つかりませんでした。")
    tongue_cloud = []

# =====================================================================
#  3. 最大Z値データの読み込みとフィルタリング
# =====================================================================
if not os.path.exists(INPUT_MAX_POINTS):
    print(f"❌ 最大値データが見つかりません: {INPUT_MAX_POINTS}")
    exit()

max_data = np.loadtxt(INPUT_MAX_POINTS)
original_count = len(max_data)

indices_to_delete = [r - 1 for r in ROWS_TO_DELETE]
valid_indices = [idx for idx in indices_to_delete if 0 <= idx < original_count]

deleted_data = max_data[valid_indices]
filtered_data = np.delete(max_data, valid_indices, axis=0)

x_filtered = filtered_data[:, 0]
z_filtered = filtered_data[:, 1]

# =====================================================================
#  4. ベーススプライン作成と、法線方向へのオフセット計算
# =====================================================================
print(f"📐 法線方向への拡張を計算中...")

sort_idx = np.argsort(x_filtered)
x_filtered = x_filtered[sort_idx]
z_filtered = z_filtered[sort_idx]

# ① まず、抽出した点を通る「ベースの曲線」を作る
base_spline = UnivariateSpline(x_filtered, z_filtered, k=3, s=SMOOTHING_FACTOR)

# ★ ベースのスプライン曲線を100点生成してテキスト保存
base_smooth_x = np.linspace(np.min(x_filtered), np.max(x_filtered), 100)
base_smooth_z = base_spline(base_smooth_x)
base_contour_data = np.column_stack((base_smooth_x, base_smooth_z))
np.savetxt(OUTPUT_BASE_DATA_FILE, base_contour_data, delimiter='\t', fmt='%.6f')
print(f"💾 ベース輪郭線データ（100点）を保存しました: {OUTPUT_BASE_DATA_FILE}")

# ② より精密に法線ベクトルを計算するため、細かく500点に分割
dense_x = np.linspace(np.min(x_filtered), np.max(x_filtered), 500)
dense_z = base_spline(dense_x)

# ③ 各点における「接線の傾き (dz/dx)」を取得
dz_dx = base_spline.derivative()(dense_x)

# ④ 法線ベクトル (nx, nz) の計算と正規化（長さを1にする）
# 接線ベクトルが (1, dz/dx) なので、垂直な上向きベクトルは (-dz/dx, 1) となります
norm = np.sqrt(1.0 + dz_dx**2)
nx = -dz_dx / norm
nz = 1.0 / norm

# ⑤ 法線方向に OFFSET_MM 分だけ点を押し出す
offset_x = dense_x + OFFSET_MM * nx
offset_z = dense_z + OFFSET_MM * nz

# =====================================================================
#  5. オフセットされた点から「新しい関数」を作り直して保存
# =====================================================================
# X軸順にソート（ドームの広がりでXが逆転する極端なケースを防ぐため）
sort_idx_off = np.argsort(offset_x)
offset_x = offset_x[sort_idx_off]
offset_z = offset_z[sort_idx_off]

# 新しいスプライン関数を作成（点はすでに滑らかなので s=0 または極小値で完全補間）
final_spline = UnivariateSpline(offset_x, offset_z, k=3, s=0.001)

# 関数オブジェクト（.pkl）の保存
with open(OUTPUT_MODEL_FILE, 'wb') as f:
    pickle.dump(final_spline, f)
print(f"💾 拡張関数モデルを保存しました: {OUTPUT_MODEL_FILE}")

# 口蓋曲線を生成してテキスト保存
smooth_x = np.linspace(np.min(offset_x), np.max(offset_x), 10000)
smooth_z = final_spline(smooth_x)

contour_data = np.column_stack((smooth_x, smooth_z))
np.savetxt(OUTPUT_DATA_FILE, contour_data, delimiter='\t', fmt='%.6f')
print(f"💾 拡張輪郭線データを保存しました: {OUTPUT_DATA_FILE}")

# =====================================================================
#  6. 結果の視覚化（グラフ描画）
# =====================================================================
plt.figure(figsize=(12, 7))

if len(tongue_cloud) > 0:
    plot_stride = max(1, len(t_x) // 100000)
    plt.scatter(t_x[::plot_stride], t_z[::plot_stride], 
                color='gray', alpha=0.02, s=1, zorder=1, label='Tongue Workspace')

if len(deleted_data) > 0:
    plt.scatter(deleted_data[:, 0], deleted_data[:, 1], 
                color='red', marker='x', s=60, zorder=3, label='Deleted Points')

plt.scatter(x_filtered, z_filtered, color='blue', s=25, edgecolor='white', zorder=4, label='Kept Points (Base Anchor)')

# 拡張する前の「ベースの曲線」を点線で表示（比較用）
plt.plot(dense_x, dense_z, color='blue', linestyle='--', linewidth=1.5, alpha=0.7, zorder=5, label='Base Spline')

# 法線方向に拡張された「最終的な口蓋アーチ」を赤い実線で表示
plt.plot(smooth_x, smooth_z, color='red', linewidth=3, zorder=6, label=f'Expanded Spline (Normal +{OFFSET_MM}mm)')

plt.xlabel('Horizontal Position [mm]')
plt.ylabel('Vertical Position [mm]')
plt.title(f'Palate Contour Expansion (Normal Vector Offset: +{OFFSET_MM}mm)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='lower right', markerscale=2.0)

plt.savefig(OUTPUT_PLOT_FILE, dpi=150)
plt.close()
print(f"🎨 確認用グラフを保存しました: {OUTPUT_PLOT_FILE}")
print("🎉 すべての処理が完了しました！")
