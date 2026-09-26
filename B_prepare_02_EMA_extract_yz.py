import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

# =====================================================================
#  1. 設定項目
# =====================================================================
SPEAKER = "W0901"
BASE_DIR = "Data"

# 入力元フォルダと出力先フォルダの設定
INPUT_DIR = os.path.join(BASE_DIR, SPEAKER, "EMA", "EMA_aligned_xyz")
OUTPUT_DIR = os.path.join(BASE_DIR, SPEAKER, "EMA", "EMA_aligned_yz")

# =====================================================================
#  2. メイン処理（Y, Z の抽出）
# =====================================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 入力ディレクトリ内のすべてのCSVファイルを取得
csv_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.csv")))

if not csv_files:
    print(f"⚠️ {INPUT_DIR} にCSVファイルが見つかりません。")
    exit()

print(f"🚀 (Y, Z) データの抽出を開始します... (対象: {len(csv_files)}ファイル)")

for file_path in tqdm(csv_files):
    # CSVデータの読み込み
    df = pd.read_csv(file_path, header=None)
    raw_data = df.values  # 形状: (フレーム数, 27)

    # -----------------------------------------------------------------
    # 抽出ロジック:
    # 1. (フレーム数, 9センサー, 3次元) に変形
    # 2. スライス[:, :, 1:3] で Y(1列目) と Z(2列目) だけを取得
    # 3. (フレーム数, 18) に再変形して [y0, z0, y1, z1...] の並びにする
    # -----------------------------------------------------------------
    yz_data = raw_data.reshape(-1, 9, 3)[:, :, 1:3].reshape(raw_data.shape[0], 18)

    # 出力ファイルパスの作成
    file_name = os.path.basename(file_path)
    out_path = os.path.join(OUTPUT_DIR, file_name)
    
    # ヘッダー・インデックスなしでCSVに保存
    pd.DataFrame(yz_data).to_csv(out_path, index=False, header=False)

print(f"\n🎉 抽出が完了しました！")
print(f"📂 保存先: {OUTPUT_DIR}")
