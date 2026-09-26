#舌が口蓋データを突き抜けていないかチェック

import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
#  1. 設定項目
# =====================================================================
# 💡 前のスクリプトで保存した口蓋の関数モデル（.pkl）のパス
PALATE_MODEL_FILE = "Data/M0201/palate/palate_contour_spline_1.25_1.5_deUI_14.pkl"

# 💡 突き抜けを確認したい舌データ（T1, T2, T3）が入っているフォルダ
TONGUE_DIR = "Data/M0201/EMA/EMA_aligned_yz"

OUTPUT_PLOT = "Data/M0201/palate/tongue_check_1.25_1.5_deUI_14.png"

# Pythonは0番目から数え始めるため、13列目はインデックス「12」になります
# T1(12, 13), T2(14, 15), T3(16, 17)
TARGET_COLS = (12, 13, 14, 15, 16, 17)
SKIP_HEADER = 0

print("🔄 舌センサ（T1, T2, T3）の口蓋突き抜けチェックを開始します...")

# =====================================================================
#  2. 口蓋のスプライン関数の読み込み
# =====================================================================
if not os.path.exists(PALATE_MODEL_FILE):
    print(f"❌ 関数モデルが見つかりません: {PALATE_MODEL_FILE}")
    exit()

with open(PALATE_MODEL_FILE, 'rb') as f:
    palate_func = pickle.load(f)

# スプライン関数が定義されているXの有効範囲（両端のノット）を取得
palate_knots = palate_func.get_knots()
palate_min_x = palate_knots[0]
palate_max_x = palate_knots[-1]

print(f"📖 口蓋関数モデルを読み込みました (有効X範囲: {palate_min_x:.2f} 〜 {palate_max_x:.2f} mm)")

# =====================================================================
#  3. 舌データの探索と一括読み込み
# =====================================================================
search_pattern = os.path.join(TONGUE_DIR, "*.csv")
tongue_files = glob.glob(search_pattern)

if not tongue_files:
    print(f"❌ 舌データが見つかりません。パスを確認してください: {TONGUE_DIR}")
    exit()

loaded_t_points = []
success_count = 0

for file_path in tongue_files:
    try:
        # T1, T2, T3 のデータを一括で読み込み
        data = np.loadtxt(file_path, delimiter=',', skiprows=SKIP_HEADER, usecols=TARGET_COLS)
        if data.size > 0:
            if data.ndim == 1:
                data = data.reshape(1, -1)
            
            # T1, T2, T3 を縦に積んで「舌の点群」としてまとめる
            # (X, Z) のペアになるように変形
            t1 = data[:, 0:2]
            t2 = data[:, 2:4]
            t3 = data[:, 4:6]
            
            loaded_t_points.append(np.vstack((t1, t2, t3)))
            success_count += 1
    except Exception as e:
        print(f"⚠️ 読み込みスキップ ({os.path.basename(file_path)}): {e}")

all_tongue_points = np.vstack(loaded_t_points)
t_x = all_tongue_points[:, 0]
t_z = all_tongue_points[:, 1]
total_points = len(t_x)

print(f"📖 舌データを読み込みました: {success_count}ファイル / 総データ数: {total_points:,} 点")

# =====================================================================
#  4. 突き抜け（Penetration）の判定
# =====================================================================
# 💡 ここで関数オブジェクトの強みを発揮！任意のX座標の天井の高さを一瞬で計算
expected_palate_z = palate_func(t_x)

# 口蓋のデータが存在するXの範囲内だけで判定を行う
valid_x_mask = (t_x >= palate_min_x) & (t_x <= palate_max_x)

# 突き抜けている条件: 舌のZ が 口蓋のZ（expected_palate_z）より大きい
penetration_mask = (t_z > expected_palate_z) & valid_x_mask

# 正常なデータと突き抜けたデータを分割
penetration_points = all_tongue_points[penetration_mask]
safe_points = all_tongue_points[~penetration_mask & valid_x_mask]

num_penetrations = len(penetration_points)
penetration_rate = (num_penetrations / total_points) * 100

print("\n==================================================")
print("📊 突き抜けチェック結果")
print("==================================================")
print(f"🔹 チェック対象点数 : {np.sum(valid_x_mask):,} 点")
print(f"🔹 突き抜け発生点数 : {num_penetrations:,} 点 ({penetration_rate:.3f} %)")

if num_penetrations > 0:
    # 突き抜けの最大深さ（どれくらい口蓋をオーバーしたか）を計算
    max_depth = np.max(penetration_points[:, 1] - palate_func(penetration_points[:, 0]))
    print(f"🔹 最大突き抜け深さ : {max_depth:.3f} mm")
print("==================================================\n")

# =====================================================================
#  5. 結果のグラフ描画
# =====================================================================
plt.figure(figsize=(12, 7))

# 描画が重くなるのを防ぐため、正常データは間引いてプロット
plot_stride = max(1, len(safe_points) // 50000)
if len(safe_points) > 0:
    plt.scatter(safe_points[::plot_stride, 0], safe_points[::plot_stride, 1], 
                color='gray', alpha=0.05, s=2, label='Normal Tongue Points (Subsampled)')

# 突き抜けたデータを赤色でプロット
if num_penetrations > 0:
    plt.scatter(penetration_points[:, 0], penetration_points[:, 1], 
                color='red', alpha=0.6, s=10, zorder=5, label='Penetrating Points')

# 口蓋ラインを描画するために、滑らかなX軸を作成
plot_x = np.linspace(palate_min_x, palate_max_x, 500)
plot_z = palate_func(plot_x)
plt.plot(plot_x, plot_z, color='black', linewidth=3, zorder=4, label='Estimated Palate Curve')

plt.xlabel('X Position (Front-Back) [mm]')
plt.ylabel('Z Position (Height) [mm]')
plt.title(f'Tongue Sensors (T1, T2, T3) Palate Penetration Check\nTotal Penetrations: {num_penetrations:,} ({penetration_rate:.2f}%)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.savefig(OUTPUT_PLOT, dpi=150)
plt.close()
print(f"🎨 確認用グラフを保存しました: {OUTPUT_PLOT}")
print("🎉 すべての処理が完了しました！")
