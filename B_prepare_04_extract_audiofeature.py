import os
import torch
import librosa
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

# cudnnのベンチマークモードをオフ（再現性確保のため）
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

# === 設定 ===
AUDIO_DIR = "Data/W0901/Audio/audio_cut"           # カット済み音声のフォルダ
OUT_FEAT_DIR = "Data/W0901/Audio/audio_features_250Hz_xlsr" # 保存先フォルダ
os.makedirs(OUT_FEAT_DIR, exist_ok=True)

# 使用するモデル（多言語対応・大容量モデル）
MODEL_NAME = "facebook/wav2vec2-large-xlsr-53"
SR = 16000  # Wav2Vec 2.0 の必須サンプリングレート

# 抽出したい中間層のインデックス (1〜24)
# 調音推定には 12〜16 あたりが適していることが多いです
TARGET_LAYER = 12

# === デバイスの設定（GPUを使用） ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用デバイス: {device}")

# === モデルの読み込み ===
print(f"モデル ({MODEL_NAME}) を読み込んでいます...")
processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
model = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(device)
model.eval() # 評価モード（学習ストップ）

# シフトするサンプル数の計算 (0ms, 4ms, 8ms, 12ms, 16ms)
# 16000Hzの場合: [0, 64, 128, 192, 256] サンプル
shift_samples = [int(ms * (SR / 1000)) for ms in [0, 4, 8, 12, 16]]
max_shift = max(shift_samples)

print(f"特徴量抽出を開始します... (対象レイヤー: 第{TARGET_LAYER}層)")

# 503文をループ処理
for i in range(1, 504):
    file_id = f"{i:03d}"
    audio_path = os.path.join(AUDIO_DIR, f"{file_id}.wav")
    
    if not os.path.exists(audio_path):
        continue
        
    print(f"処理中: {file_id} ...", end=" ")
    
    # 1. 音声の読み込み
    audio, _ = librosa.load(audio_path, sr=SR)
    original_length = len(audio)
    
    # 2. ズラすための余白（ゼロ埋め）を後ろに追加
    padded_audio = np.pad(audio, (0, max_shift), mode='constant')
    
    # 3. 5パターンのズラした音声配列を作成
    batch_audio = []
    for shift in shift_samples:
        # すべて同じ長さ (original_length) で切り出す
        shifted = padded_audio[shift : original_length + shift]
        batch_audio.append(shifted)
        
    # リストをNumpy配列に変換 -> 形状: (5, 音声の長さ)
    batch_audio = np.array(batch_audio)
    
    # 4. GPUで一気に特徴量抽出
    inputs = processor(batch_audio, sampling_rate=SR, return_tensors="pt").to(device)
    
    with torch.no_grad():
        # output_hidden_states=True で全層の出力を取得
        outputs = model(**inputs, output_hidden_states=True)
        
    # 指定した中間層の出力を取得し、CPUメモリに戻してNumpy配列にする
    # 形状: (5, フレーム数, 1024)
    features = outputs.hidden_states[TARGET_LAYER].cpu().numpy()
    
    # 5. インターリーブ（合体）処理
    # (5, フレーム数, 1024) を (フレーム数, 5, 1024) に入れ替える
    features_transposed = features.transpose(1, 0, 2)
    
    # ファスナーを閉めるように順番に平坦化 -> 形状: (フレーム数 * 5, 1024)
    # Largeモデルなので 768 ではなく 1024 になります
    interleaved_features = features_transposed.reshape(-1, 1024)
    
    # 6. 保存 (.npy形式)
    out_path = os.path.join(OUT_FEAT_DIR, f"{file_id}_feat.npy")
    np.save(out_path, interleaved_features)
    
    print(f"完了! shape: {interleaved_features.shape}")

print("\nすべての特徴量抽出が完了しました！")
