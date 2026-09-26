#口蓋データとEMAを同時に表示する動画

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import japanize_matplotlib  # 必要に応じて残しています

# ==========================================
# 設定（データ構造に合わせて微調整してください）
# ==========================================
ema_file_path = "Data/W0901/EMA/EMA_aligned_cut/001.csv"
palate_file_path = "Data/z_palate2/W0901_palate_1.0_1000.data"  
output_video_path = "z100_palate_spline2/W0901_001_ema_palate_spline.mp4"

# 動画の1秒あたりのフレーム数（再生速度）
FPS = 30  
INTERVAL = int(1000 / FPS)

try:
    # 1. 口蓋データ (palate) の読み込み
    palate_df = pd.read_csv(palate_file_path, sep=r'\s+', engine='python', header=None)
    palate_x = palate_df[0].values
    palate_y = palate_df[1].values

    # 2. EMAデータ (001.csv) の読み込み
    ema_df = pd.read_csv(ema_file_path, header=None)
    
    num_cols = ema_df.shape[1]
    num_sensors = num_cols // 2  
    
    # 3. グラフとアニメーションの初期設定
    # 💡 凡例が右側にはみ出すため、横幅を少し広げました (8 -> 10)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_title("EMA Sensor Movement with Palate Profile")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")

    # 口蓋データの常時表示
    ax.plot(palate_x, palate_y, 'k-', lw=3, label='Palate (口蓋)', alpha=0.8) 

    # EMAセンサーの描画用オブジェクトを準備
    scatters = []
    colors = plt.cm.jet(np.linspace(0, 1, num_sensors)) if num_sensors > 0 else ['red']

    for i in range(num_sensors):
        sc, = ax.plot([], [], 'o', color=colors[i], markersize=8, label=f'Sensor {i+1}')
        scatters.append(sc)
    
    # 💡【修正ポイント】凡例をグラフの枠の外（右側）へ追い出す設定
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0)

    # 💡【修正ポイント】表示範囲を指定された固定値に変更
    ax.set_xlim(-30, 80)
    ax.set_ylim(-60, 80)
    ax.set_aspect('equal')  # 縦横比を1:1にして歪みを防ぐ

    # 右側に追い出した凡例が画面外に切れないようにレイアウトを最適化
    plt.tight_layout()

    time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12, verticalalignment='top')

    # アニメーション初期化関数
    def init():
        for sc in scatters:
            sc.set_data([], [])
        time_text.set_text('')
        return scatters + [time_text]

    # フレーム更新関数
    def update(frame):
        row_data = ema_df.iloc[frame].values
        
        for i in range(num_sensors):
            x_val = row_data[2 * i]
            y_val = row_data[2 * i + 1]
            
            if not np.isnan(x_val) and not np.isnan(y_val):
                scatters[i].set_data([x_val], [y_val])
            else:
                scatters[i].set_data([], [])
                
        time_text.set_text(f'Frame: {frame}')
        return scatters + [time_text]

    total_frames = len(ema_df)
    print(f"動画を生成中... (総フレーム数: {total_frames})")
    
    ani = animation.FuncAnimation(
        fig, update, frames=total_frames, init_func=init, blit=True, interval=INTERVAL
    )

    # 動画（MP4）として保存
    ani.save(output_video_path, writer='ffmpeg', fps=FPS)
    plt.close()
    
    print(f"動画の保存が完了しました！\n保存先: {output_video_path}")

except FileNotFoundError as e:
    print(f"ファイルが見つかりません。パスを確認してください:\n{e}")
except Exception as e:
    print(f"エラーが発生しました: {e}")
