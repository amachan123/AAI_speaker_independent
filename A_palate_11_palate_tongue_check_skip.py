#W0401の異常がある発話分をスキップして舌が口蓋を突き抜けていないかチェック

import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
#  1. 設定項目
# =====================================================================
# 💡 前のスクリプトで保存した口蓋の関数モデル（.pkl）のパス
PALATE_MODEL_FILE = "Data/W0401/palate/palate_contour_spline_1.05_1.5_deUI.pkl"

# 💡 突き抜けを確認したい舌データ（T1, T2, T3）が入っているフォルダ
TONGUE_DIR = "Data/W0401/EMA/EMA_aligned_yz"

OUTPUT_PLOT = "Data/W0401/palate/tongue_check_1.05_1.5_10.png"

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
#  3. 舌データの探索と一括読み込み ＆ ファイルごとの突き抜けチェック
# =====================================================================
search_pattern = os.path.join(TONGUE_DIR, "*.csv")
tongue_files = glob.glob(search_pattern)

if not tongue_files:
    print(f"❌ 舌データが見つかりません。パスを確認してください: {TONGUE_DIR}")
    exit()

# スキップしたいファイル名
SKIP_FILES = ["017_combined.csv", "020_combined.csv", "029_combined.csv", "036_combined.csv"]

loaded_t_points = []
success_count = 0

# 💡 ファイルごとの突き抜け結果を記録するリスト
penetration_summary = []

for file_path in tongue_files:
    filename = os.path.basename(file_path)
    
    # スキップ判定
    if filename in SKIP_FILES:
        print(f"⏭️ 指定によりスキップしました: {filename}")
        continue

    try:
        # T1, T2, T3 のデータを一括で読み込み
        data = np.loadtxt(file_path, delimiter=',', skiprows=SKIP_HEADER, usecols=TARGET_COLS)
        if data.size > 0:
            if data.ndim == 1:
                data = data.reshape(1, -1)
            
            t1 = data[:, 0:2]
            t2 = data[:, 2:4]
            t3 = data[:, 4:6]
            file_points = np.vstack((t1, t2, t3))
            
            # 💡 このファイル単体での突き抜けチェックを行う
            fx = file_points[:, 0]
            fz = file_points[:, 1]
            
            f_valid_mask = (fx >= palate_min_x) & (fx <= palate_max_x)
            f_expected_z = palate_func(fx)
            f_penetration_mask = (fz > f_expected_z) & f_valid_mask
            
            f_num_penetrations = np.sum(f_penetration_mask)
            
            # 突き抜けが1点でもあれば記録
            if f_num_penetrations > 0:
                f_max_depth = np.max(fz[f_penetration_mask] - f_expected_z[f_penetration_mask])
                penetration_summary.append({
                    'filename': filename,
                    'count': f_num_penetrations,
                    'max_depth': f_max_depth
                })
            
            loaded_t_points.append(file_points)
            success_count += 1
            
    except Exception as e:
        print(f"⚠️ 読み込みスキップ ({filename}): {e}")

# 全データを統合（グラフ描画用）
all_tongue_points = np.vstack(loaded_t_points)
t_x = all_tongue_points[:, 0]
t_z = all_tongue_points[:, 1]
total_points = len(t_x)

print(f"📖 舌データを読み込みました: {success_count}ファイル / 総データ数: {total_points:,} 点")


# =====================================================================
#  4. 突き抜け（Penetration）の全体結果と内訳の表示
# =====================================================================
expected_palate_z = palate_func(t_x)
valid_x_mask = (t_x >= palate_min_x) & (t_x <= palate_max_x)
penetration_mask = (t_z > expected_palate_z) & valid_x_mask

penetration_points = all_tongue_points[penetration_mask]
safe_points = all_tongue_points[~penetration_mask & valid_x_mask]

num_penetrations = len(penetration_points)
penetration_rate = (num_penetrations / total_points) * 100

print("\n==================================================")
print("📊 突き抜けチェック結果 (全体)")
print("==================================================")
print(f"🔹 チェック対象点数 : {np.sum(valid_x_mask):,} 点")
print(f"🔹 突き抜け発生点数 : {num_penetrations:,} 点 ({penetration_rate:.3f} %)")

if num_penetrations > 0:
    max_depth = np.max(penetration_points[:, 1] - palate_func(penetration_points[:, 0]))
    print(f"🔹 最大突き抜け深さ : {max_depth:.3f} mm")
print("==================================================")

# 💡 どのファイルで突き抜けが起きたかを一覧表示
print("\n📂 突き抜けが発生したファイルの内訳:")
print("--------------------------------------------------")
if penetration_summary:
    for item in penetration_summary:
        print(f" 🔴 {item['filename']}: {item['count']:,} 点が突き抜け (最大深さ: {item['max_depth']:.3f} mm)")
else:
    print(" ✅ 突き抜けが発生したファイルはありません。")
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
