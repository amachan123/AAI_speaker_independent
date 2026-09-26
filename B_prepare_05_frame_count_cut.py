#音声特徴量とEMAデータのフレーム数を揃える。

import os
import shutil
import numpy as np
import pandas as pd

# === 1. 入力フォルダの設定（元のデータ） ===
IN_FEAT_DIR = "Data/M0102/Audio/audio_features_250Hz_xlsr"
IN_EMA_DIR = "Data/M0102/EMA/EMA_aligned_yz_cut"

# === 2. 出力フォルダの設定（長さを揃えた新しいデータ） ===
OUT_FEAT_DIR = "Data/M0102/Audio/audio_features_250Hz_xlsr_aligned"
OUT_EMA_DIR = "Data/M0102/EMA/EMA_aligned_yz_cut_aligned"

# 出力先のフォルダが存在しない場合は自動で作成する
os.makedirs(OUT_FEAT_DIR, exist_ok=True)
os.makedirs(OUT_EMA_DIR, exist_ok=True)

# ズレがあった場合のログを保存するリスト
mismatch_log = []

print("フレーム数の一致チェックと別フォルダへの出力処理を開始します...\n")

for i in range(1, 504):
    file_id = f"{i:03d}"
    
    # 元ファイルのパス
    feat_path = os.path.join(IN_FEAT_DIR, f"{file_id}_feat.npy")
    ema_path = os.path.join(IN_EMA_DIR, f"{file_id}.csv")
    
    # 保存先ファイルのパス
    out_feat_path = os.path.join(OUT_FEAT_DIR, f"{file_id}_feat.npy")
    out_ema_path = os.path.join(OUT_EMA_DIR, f"{file_id}.csv")
    
    if not os.path.exists(feat_path) or not os.path.exists(ema_path):
        continue
        
    # 特徴量(.npy)の読み込みとフレーム数取得
    feat_data = np.load(feat_path)
    feat_frames = feat_data.shape[0]
    
    # EMA(.csv)の読み込みとフレーム数取得
    ema_data = pd.read_csv(ema_path, header=None).values
    ema_frames = ema_data.shape[0]
    
    # 比較
    diff = feat_frames - ema_frames
    
    if diff == 0:
        print(f"[{file_id}] ✅ 一致 (Frames: {feat_frames}) -> そのままコピーします")
        # ズレがない場合は、データをそのまま新しいフォルダへコピー（処理が高速で安全です）
        shutil.copy(feat_path, out_feat_path)
        shutil.copy(ema_path, out_ema_path)
    else:
        # ズレがあった場合の処理
        min_len = min(feat_frames, ema_frames)
        print(f"[{file_id}] ✂️ ズレあり (音声: {feat_frames}, EMA: {ema_frames}) -> {min_len} に合わせてカットして保存します")
        
        # 短い方に合わせて配列をカット
        feat_data_trimmed = feat_data[:min_len, :]
        ema_data_trimmed = ema_data[:min_len, :]
        
        # 新しいフォルダに保存
        np.save(out_feat_path, feat_data_trimmed)
        pd.DataFrame(ema_data_trimmed).to_csv(out_ema_path, header=False, index=False)
        
        # ログに記録
        mismatch_log.append({
            "id": file_id,
            "original_feat": feat_frames,
            "original_ema": ema_frames,
            "trimmed_len": min_len
        })

print("\n=== すべての処理が完了しました ===")
if len(mismatch_log) == 0:
    print(f"🎉 すべて一致していました！データはそのまま {OUT_FEAT_DIR} および {OUT_EMA_DIR} にコピーされました。")
else:
    print(f"⚠️ {len(mismatch_log)} 件のファイルでズレを検出し、短い方に合わせてカットしました。")
    print(f"🎉 完全に長さが揃った綺麗なデータセットが {OUT_FEAT_DIR} と {OUT_EMA_DIR} に作成されました！")
