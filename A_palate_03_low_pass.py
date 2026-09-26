#口蓋データにローパスをかける。

import os
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

# =====================================================================
#  1. 設定項目（処理したいファイルを指定してください）
# =====================================================================
# 💡 処理したい入力ファイルを直接指定します
INPUT_FILE = "Data/W0401/palate/1.05_hfS20200220_D2020022010703mov_0_bar_tip_POS_angle.data"

dir_name = os.path.dirname(INPUT_FILE)   # 例: ../display_data/.../S20250304_D2025030410251mov
base_name = os.path.basename(INPUT_FILE)

OUTPUT_FILE = os.path.join(dir_name, f"lpf_{base_name}")

# サンプリング周波数
F_EMA = 250

# 💡 ローパスフィルタ(LPF)の設計 (MATLABの fir1(48, 0.1) と完全等価)
numtaps = 48 + 1
cutoff_hz = 12.5
blpf = signal.firwin(numtaps, cutoff_hz, fs=F_EMA, window='hamming')

# =====================================================================
#  2. データの読み込みとフィルタ処理
# =====================================================================
print(f"🔄 処理を開始します...")
print(f"📖 入力: {INPUT_FILE}")

if not os.path.exists(INPUT_FILE):
    print("❌ 入力ファイルが見つかりません。パスを確認してください。")
    exit()

# データの読み込み
M = np.loadtxt(INPUT_FILE)
O_M = M.copy()  # 4列目以降の角度データ等を保持するためにコピー

# 各列の平均値を計算 (端点効果対策)
col_means = np.nanmean(M, axis=0)

# 1〜3列目（X, Y, Z座標。Pythonのインデックスでは 0, 1, 2）に対して処理
for l in range(3):
    # 平均値を引き算
    M_a = M[:, l] - col_means[l]
    
    # ゼロ位相フィルタリング (位相がズレないフィルタ)
    filtered_col = signal.filtfilt(blpf, [1.0], M_a)
    
    # 平均値を足し戻して格納
    O_M[:, l] = filtered_col + col_means[l]

# =====================================================================
#  3. 結果の保存
# =====================================================================
# 6列構成（座標+角度+エラー値）を想定し、最終列だけ指数表記 %e、他は %lf にします
if O_M.shape[1] == 6:
    fmt_list = ['%lf', '%lf', '%lf', '%lf', '%lf', '%e']
else:
    fmt_list = '%lf'

np.savetxt(OUTPUT_FILE, O_M, delimiter='\t', fmt=fmt_list)
print(f"💾 保存完了: {OUTPUT_FILE}")

# =====================================================================
#  📊 4. フィルタ前後の視覚的確認（グラフ画像の保存）
# =====================================================================
nraw = O_M.shape[0]
time = np.arange(1, nraw + 1) / F_EMA

plt.figure(figsize=(10, 5))
# MATLABコードと同じく、3列目（Z軸の座標）をプロットして比較します
plt.plot(time, M[:, 2], label='Original (Z-axis)', alpha=0.5, color='gray')
plt.plot(time, O_M[:, 2], label='Filtered (LPF 12.5Hz)', color='blue', linewidth=1.5)

plt.xlabel('Time [s]')
plt.ylabel('Z Position [mm]')
plt.title('EMA LPF Verification (Single File)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

# グラフを画像ファイルとして保存
plot_output = os.path.join(dir_name, f"700_lpf_single_verify.png")
plt.savefig(plot_output, dpi=150)
plt.close()

print(f"🎨 確認用グラフを保存しました: {plot_output}")
print("🎉 すべての処理が正常に終了しました。")
