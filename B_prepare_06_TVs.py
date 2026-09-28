#EMAデータと口蓋データから声道変数(TVs)を求める

import os
import glob
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from tqdm import tqdm

# =====================================================================
#  1. 設定項目（※パスを実際の環境に合わせて変更してください）
# =====================================================================
PALATE_DATA_PATH = "Data/z_palate2/W0901_palate_1.0_1000.data"

INPUT_EMA_DIR = "Data/W0901/EMA/EMA_aligned_yz_cut" 
# 声道変数のみを保存するフォルダに変更
OUTPUT_TV_DIR = "Data/z_TV/W0901_TV_1000"

os.makedirs(OUTPUT_TV_DIR, exist_ok=True)

# CSVの列インデックス定義 (0始まり)
# 構成: ND(0,1), NA(2,3), UI(4,5), UL(6,7), LL(8,9), LJ(10,11), T1(12,13), T2(14,15), T3(16,17)
UL_Y, UL_Z = 6, 7
LL_Y, LL_Z = 8, 9
LJ_Y, LJ_Z = 10, 11
T1_Y, T1_Z = 12, 13
T2_Y, T2_Z = 14, 15
T3_Y, T3_Z = 16, 17

# =====================================================================
#  2. 口蓋データ (1000点) の読み込み
# =====================================================================
print("📖 口蓋データ (1000点) を読み込んでいます...")
palate_points = np.loadtxt(PALATE_DATA_PATH, delimiter='\t')

csv_files = sorted(glob.glob(os.path.join(INPUT_EMA_DIR, "*.csv")))
if len(csv_files) == 0:
    print("❌ CSVファイルが見つかりません。パスを確認してください。")
    exit()

# =====================================================================
#  3. パス1: 全ファイルを通した水平(y)座標の中央値（Global Median）を計算
# =====================================================================
print("🔍 全データの中央値を計算しています（パス1/2）...")

all_ll_y, all_t1_y, all_t2_y, all_t3_y = [], [], [], []

for file_path in tqdm(csv_files, desc="Median Calc"):
    df = pd.read_csv(file_path, header=None)
    all_ll_y.append(df[LL_Y].values)
    all_t1_y.append(df[T1_Y].values)
    all_t2_y.append(df[T2_Y].values)
    all_t3_y.append(df[T3_Y].values)

# 全フレームを1次配列に結合して中央値を算出
median_LL_Y = np.median(np.concatenate(all_ll_y))
median_T1_Y = np.median(np.concatenate(all_t1_y))
median_T2_Y = np.median(np.concatenate(all_t2_y))
median_T3_Y = np.median(np.concatenate(all_t3_y))

print(f"  - LL(y) 中央値: {median_LL_Y:.4f}")
print(f"  - T1(y) 中央値: {median_T1_Y:.4f}")
print(f"  - T2(y) 中央値: {median_T2_Y:.4f}")
print(f"  - T3(y) 中央値: {median_T3_Y:.4f}")

# =====================================================================
#  4. パス2: 声道変数の計算と保存（変数のみ抽出）
# =====================================================================
print("\n🚀 声道変数を計算し、独立したファイルとして保存しています（パス2/2）...")

for file_path in tqdm(csv_files, desc="TV Calc"):
    file_name = os.path.basename(file_path)
    out_path = os.path.join(OUTPUT_TV_DIR, file_name)
    
    df = pd.read_csv(file_path, header=None)
    
    # 舌の各センサーの座標配列を作成 (フレーム数, 2)
    t1_coords = df[[T1_Y, T1_Z]].values
    t2_coords = df[[T2_Y, T2_Z]].values
    t3_coords = df[[T3_Y, T3_Z]].values
    
    # 新しいデータフレーム（声道変数のみを格納）を作成
    tv_df = pd.DataFrame()
    
    # --- [1] 唇・顎の変数 (3つ) ---
    tv_df['LA'] = np.sqrt((df[UL_Y] - df[LL_Y])**2 + (df[UL_Z] - df[LL_Z])**2)
    tv_df['LP'] = df[LL_Y] - median_LL_Y
    tv_df['JA'] = np.sqrt((df[UL_Y] - df[LJ_Y])**2 + (df[UL_Z] - df[LJ_Z])**2)
    
    # --- [2] 舌の変数 (6つ: T1, T2, T3 それぞれの CD と CL) ---
    tv_df['TTCD'] = np.min(cdist(t1_coords, palate_points), axis=1)
    tv_df['TTCL'] = df[T1_Y] - median_T1_Y
    
    tv_df['TMCD'] = np.min(cdist(t2_coords, palate_points), axis=1)
    tv_df['TMCL'] = df[T2_Y] - median_T2_Y
    
    tv_df['TBCD'] = np.min(cdist(t3_coords, palate_points), axis=1)
    tv_df['TBCL'] = df[T3_Y] - median_T3_Y
    
    # 声道変数のみ（9列）の状態でCSVとして保存
    tv_df.to_csv(out_path, index=False, header=False)

print("\n🎉 すべての処理が完了しました！ 声道変数のみのデータセットが完成しました。")
