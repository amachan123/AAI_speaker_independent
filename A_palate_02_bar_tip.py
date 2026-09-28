#実験室のパソコンに入っていたスクリプトをPythonで使えるようにした。
#口蓋データを推定する。

import os
import numpy as np

# =====================================================================
#  1. 設定項目（ここを書き換えるだけで簡単に調整できます）
# =====================================================================
# 💡 倍率の調整（1.5倍にする場合は 1.5、元のコードと同じにする場合は 1.0）
# M0102：0.85, M0201：1.25, W0401：1.05
TIP_SCALE = 1.0

# チャンネル設定（先日判明した、ch1＝中間、ch5＝根本 のペアを指定）
BAR_CH_NO_1 = 10  # 中間に位置するチャンネル (C言語の BAR_CH_NO_1 相当)
BAR_CH_NO_2 = 6  # 根本側に位置するチャンネル (C言語の BAR_CH_NO_2 相当)

# ディレクトリ・ファイルパスの設定

ESTIMATED_DATA_DIR = "Data/W0401/S20200220_D2020022010703mov"          
HF_FILE_NAME = "hfS20200220_D2020022010703mov"              
FILE_NO = 0
FILE_NAME_AFTER_CH_NO = "POS_angle"

# =====================================================================
#  2. データの読み込み
# =====================================================================
print("データの読み込みを開始します...")

# 読み込み元ファイルパスの組み立て
ch1_file_path = f"{ESTIMATED_DATA_DIR}/{HF_FILE_NAME}_{FILE_NO}_ch{BAR_CH_NO_1}_{FILE_NAME_AFTER_CH_NO}.data"
ch2_file_path = f"{ESTIMATED_DATA_DIR}/{HF_FILE_NAME}_{FILE_NO}_ch{BAR_CH_NO_2}_{FILE_NAME_AFTER_CH_NO}.data"

if not os.path.exists(ch1_file_path) or not os.path.exists(ch2_file_path):
    print("❌ 入力ファイルが見つかりません。パスを確認してください。")
    print(f"探したパス1: {ch1_file_path}")
    print(f"探したパス2: {ch2_file_path}")
    exit()

# 最初の5列（X, Y, Zの座標 ＋ 2軸の角度）を読み込む
# ※C言語のコードが6列目のダミー数値を読み飛ばしていた挙動を再現しています
pos_ch1 = np.loadtxt(ch1_file_path, usecols=range(5))
pos_ch2 = np.loadtxt(ch2_file_path, usecols=range(5))

print(f"➔ 読み込み成功。総フレーム数: {len(pos_ch1)}")

# =====================================================================
#  3. 先端位置（bar_tip）の一括計算
# =====================================================================
print(f"先端位置を計算中... (設定倍率: {TIP_SCALE})")

# 💡 NumPyによる一括ベクトル演算（C言語のような2重ループは不要です）
# 全フレーム・全5軸分がこの1行で同時に計算されます
pos_tip = pos_ch1 + TIP_SCALE * (pos_ch1 - pos_ch2)

# =====================================================================
#  4. C言語仕様に合わせたフォーマットでファイル保存
# =====================================================================
output_file_path = f"Data/W0401/palate/1.05_{HF_FILE_NAME}_{FILE_NO}_bar_tip_{FILE_NAME_AFTER_CH_NO}.data"

# C言語の「6列目に %e で 0.0 を書き出す」仕様を再現するため、0の列を作成して結合
dummy_column = np.zeros((pos_tip.shape[0], 1))
output_data = np.hstack([pos_tip, dummy_column])

# 保存用の出力フォーマットを指定
# 最初の5列は通常の浮動小数点（%lf）、最後の6列目は指数表記（%e）
fmt_list = ['%lf', '%lf', '%lf', '%lf', '%lf', '%e']

print("計算結果をファイルに書き出しています...")
np.savetxt(output_file_path, output_data, delimiter='\t', fmt=fmt_list)

print(f"🎉 処理が完了しました！\n保存先: {output_file_path}")
