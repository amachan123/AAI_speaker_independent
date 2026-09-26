import os
import pandas as pd
import numpy as np

# --- 設定 ---
base_dir = "Data/M0102/20180911lpf_hf_poscsv"  # 001~503フォルダがある親ディレクトリのパス
output_dir = "Data/M0102/EMA/EMA_combined_xyz"     # 結合済みデータの保存先
os.makedirs(output_dir, exist_ok=True)

# センサのリスト
sensors = ["ND", "NA", "UI", "UL", "LL", "LJ", "T1", "T2", "T3"]

# 503文分ループ
for i in range(1, 504):
    folder_name = f"{i:03d}"      # "001", "002" ...
    folder_path = os.path.join(base_dir, folder_name)
    
    if not os.path.isdir(folder_path):
        continue
    
    combined_df = pd.DataFrame()
    
    for sensor in sensors:
        # ファイル名作成 (例: 1_ND.csv)
        file_name = f"{i}_{sensor}.csv"
        file_path = os.path.join(folder_path, file_name)
        
        if os.path.exists(file_path):
            # CSV読み込み（ヘッダーがない場合は header=None）
            df = pd.read_csv(file_path, header=None)
            
            # 1次元目(index 0)、2次元目(index 1)、3次元目(index 2)を抽出
            extracted = df.iloc[:, [0, 1, 2]]
            
            # 横に結合
            combined_df = pd.concat([combined_df, extracted], axis=1)
        else:
            print(f"Warning: {file_path} が見つかりません。")

    # 結果を保存
    output_file_csv = os.path.join(output_dir, f"{folder_name}_combined.csv")
    combined_df.to_csv(output_file_csv, index=False, header=False)

print("すべてのデータ統合が完了しました。")
