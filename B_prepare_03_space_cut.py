#データ前後の空白区間をカットする

import os
import librosa
import numpy as np
import soundfile as sf
import pandas as pd

# === 1. 設定（ご自身の環境に合わせて変更してください） ===

# フォルダの設定
AUDIO_DIR = "Data/W0901/20250304hpf_wav_16k"     # 元の音声データ(.wav)が入っているフォルダ
EMA_DIR = "Data/W0901/EMA/EMA_aligned_yz"        # 先ほど作成した統合済みEMAのフォルダ
OUT_AUDIO_DIR = "Data/W0901/Audio/audio_cut" # カット後の音声の保存先
OUT_EMA_DIR = "Data/W0901/EMA/EMA_aligned_cut"     # カット後のEMAの保存先

os.makedirs(OUT_AUDIO_DIR, exist_ok=True)
os.makedirs(OUT_EMA_DIR, exist_ok=True)

# パラメータの設定
EMA_SR = 250         # ★ EMAデータのサンプリング周波数(Hz)を入れてください
TOP_DB = 30          # 無音カットの閾値（数値を小さくするとより厳しくカットされます）

MARGIN_START_SEC = 0.1    # 発話前後に残すマージン（秒）
MARGIN_END_SEC = 0.1      # 発話前後に残すマージン（秒）

# =======================================================

for i in range(1, 504):
    file_id = f"{i:03d}" # 001, 002 ...
    
    # ★ 音声ファイルの名前が "001.wav" ではない場合は、ここを変更してください
    audio_path = os.path.join(AUDIO_DIR, f"{file_id}.wav") 
    ema_path = os.path.join(EMA_DIR, f"{file_id}_combined.csv")
    
    # ファイルが存在するかチェック
    if not os.path.exists(audio_path) or not os.path.exists(ema_path):
        continue
        
    print(f"処理中: {file_id} ...", end=" ")
        
    # --- 1. データの読み込み ---
    audio, audio_sr = librosa.load(audio_path, sr=None)
    ema_df = pd.read_csv(ema_path, header=None)
    ema_data = ema_df.values # numpy配列に変換
    
    # --- 2. 音声から発話区間を検出 ---
    # indexには [発話開始のサンプル位置, 発話終了のサンプル位置] が入る
    _, index = librosa.effects.trim(audio, top_db=TOP_DB, frame_length=4096, hop_length=1024)
    
    # --- 3. サンプル位置を「秒数」に変換し、マージンを追加 ---
    start_sec = max(0, (index[0] / audio_sr) - MARGIN_START_SEC)
    end_sec = min(len(audio) / audio_sr, (index[1] / audio_sr) + MARGIN_END_SEC)
    
    # --- 4. カットするインデックスを計算してスライス ---
    audio_start = int(start_sec * audio_sr)
    audio_end = int(end_sec * audio_sr)
    ema_start = int(start_sec * EMA_SR)
    ema_end = int(end_sec * EMA_SR)
    
    audio_cut = audio[audio_start:audio_end]
    ema_cut = ema_data[ema_start:ema_end, :]
    
    # --- 5. カットしたデータを保存 ---
    sf.write(os.path.join(OUT_AUDIO_DIR, f"{file_id}.wav"), audio_cut, audio_sr)
    pd.DataFrame(ema_cut).to_csv(os.path.join(OUT_EMA_DIR, f"{file_id}.csv"), index=False, header=False)
    
    print(f"完了 (音声: {len(audio_cut)} samples, EMA: {len(ema_cut)} frames)")

print("すべてのカット処理が完了しました。")
