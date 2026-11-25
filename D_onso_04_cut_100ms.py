"""JuliusのsilB/silEを基準に音声・EMA・音素アライメントを同期カットする。

・silB終了前を100 ms残す
・silE開始後を100 ms残す
・元データに100 ms存在しない場合は不足分を補う
    WAV : 0（無音）
    EMA : 端の座標を繰り返す
・切り出し後のJuliusアライメントをEMA 250 Hzフレームへ対応付ける
"""

from pathlib import Path
import csv
import math

import numpy as np
import pandas as pd
from scipy.io import wavfile


# =========================================================
# 設定
# =========================================================
ROOT = Path("Data")
OUTPUT = ROOT / "IPA_preprocessed"

SPEAKERS = {
    "M0102": "20180911",
    "M0201": "20181115",
    "W0401": "20200220",
    "W0901": "20250304",
}

EMA_SR = 250
WAV_SR = 16000

MARGIN_SEC = 0.100
MARGIN_FRAMES = int(MARGIN_SEC * EMA_SR)   # 25フレーム

NUM_FILES = 503

EMA_COLUMNS = [
    "ND_y", "ND_z",
    "NA_y", "NA_z",
    "UI_y", "UI_z",
    "UL_y", "UL_z",
    "LL_y", "LL_z",
    "LJ_y", "LJ_z",
    "T1_y", "T1_z",
    "T2_y", "T2_z",
    "T3_y", "T3_z",
]


# =========================================================
# 1発話分
# =========================================================
def process(speaker, date, number):

    name = f"{number:03d}"

    # -----------------------------------------------------
    # 入力パス
    # -----------------------------------------------------
    wav_path = (
        ROOT / "data" / speaker
        / f"{date}hpf_wav_16k"
        / f"{name}.wav"
    )

    ema_path = (
        ROOT / "data" / speaker
        / "EMA" / "EMA_aligned_yz"
        / f"{name}_combined.csv"
    )

    lab_path = (
        ROOT / "IPAlabel"
        / "JuliusAlignment_sp"
        / speaker
        / f"{name}.lab"
    )

    for path in (wav_path, ema_path, lab_path):
        if not path.is_file():
            raise ValueError(f"File missing: {path}")

    # =====================================================
    # WAV読み込み
    # =====================================================
    sr, wav = wavfile.read(wav_path)

    if sr != WAV_SR or wav.ndim != 1:
        raise ValueError(
            f"WAV must be mono {WAV_SR}Hz: {wav_path}"
        )

    samples_per_frame = sr // EMA_SR   # 16000 / 250 = 64

    # =====================================================
    # EMA読み込み
    # =====================================================
    ema = pd.read_csv(
        ema_path,
        header=None
    ).to_numpy()

    if ema.shape[1:] != (18,) or len(ema) == 0:
        raise ValueError(
            f"EMA shape invalid: {ema.shape}"
        )

    # =====================================================
    # Julius alignment読み込み
    # =====================================================
    lab = pd.read_csv(
        lab_path,
        sep=r"\s+",
        header=None,
        names=["start", "end", "phone"],
        encoding="utf-8-sig"
    )

    if (
        len(lab) < 3
        or lab.iloc[0]["phone"] != "silB"
        or lab.iloc[-1]["phone"] != "silE"
    ):
        raise ValueError(
            "Julius labels must begin with silB and end with silE"
        )

    starts = lab["start"].to_numpy(float)
    ends = lab["end"].to_numpy(float)

    # Julius境界チェック
    if (
        not np.isfinite(starts).all()
        or not np.isfinite(ends).all()
        or np.any(starts < 0)
        or np.any(ends <= starts)
        or np.any(np.abs(starts[1:] - ends[:-1]) > 0.01)
    ):
        raise ValueError(
            "Invalid or discontinuous Julius boundaries"
        )

    # =====================================================
    # WAVとEMAの共通時間範囲
    # =====================================================
    common_end = min(
        len(ema),
        len(wav) * EMA_SR // sr
    )

    if common_end <= 0:
        raise ValueError(
            f"No common WAV/EMA range: {speaker}/{name}"
        )

    # =====================================================
    # silB / silEの境界をEMAフレームへ変換
    # =====================================================
    # silB終了 = 発話開始
    speech_start_frame = math.floor(
        ends[0] * EMA_SR
    )

    # silE開始 = 発話終了
    speech_end_frame = math.ceil(
        starts[-1] * EMA_SR
    )

    speech_start_frame = max(
        0,
        min(speech_start_frame, common_end)
    )

    speech_end_frame = max(
        speech_start_frame,
        min(speech_end_frame, common_end)
    )

    # =====================================================
    # silB側：発話開始前100 ms
    # =====================================================
    first = max(
        0,
        speech_start_frame - MARGIN_FRAMES
    )

    available_before = (
        speech_start_frame - first
    )

    # 足りないフレーム数
    pad_before = max(
        0,
        MARGIN_FRAMES - available_before
    )

    # =====================================================
    # silE側：発話終了後100 ms
    # =====================================================
    last = min(
        common_end,
        speech_end_frame + MARGIN_FRAMES
    )

    available_after = (
        last - speech_end_frame
    )

    pad_after = max(
        0,
        MARGIN_FRAMES - available_after
    )

    if last <= first:
        raise ValueError(
            f"Invalid crop range: {speaker}/{name}"
        )

    # =====================================================
    # 元データを同期カット
    # =====================================================
    wav_first = first * samples_per_frame
    wav_last = last * samples_per_frame

    wav_cut = wav[
        wav_first:wav_last
    ]

    ema_cut = ema[
        first:last
    ]

    # =====================================================
    # 不足しているsilB / silEを補う
    # =====================================================

    # -------------------------
    # WAV：0（完全な無音）
    # -------------------------
    if pad_before > 0:

        wav_cut = np.pad(
            wav_cut,
            (
                pad_before * samples_per_frame,
                0
            ),
            mode="constant",
            constant_values=0
        )

    if pad_after > 0:

        wav_cut = np.pad(
            wav_cut,
            (
                0,
                pad_after * samples_per_frame
            ),
            mode="constant",
            constant_values=0
        )

    # -------------------------
    # EMA：端の座標を繰り返す
    # -------------------------
    if pad_before > 0:

        ema_cut = np.pad(
            ema_cut,
            (
                (pad_before, 0),
                (0, 0)
            ),
            mode="edge"
        )

    if pad_after > 0:

        ema_cut = np.pad(
            ema_cut,
            (
                (0, pad_after),
                (0, 0)
            ),
            mode="edge"
        )

    # =====================================================
    # WAV / EMA長さ確認
    # =====================================================
    if (
        len(wav_cut)
        != len(ema_cut) * samples_per_frame
    ):
        raise ValueError(
            f"WAV/EMA length mismatch after padding: "
            f"{speaker}/{name}"
        )

    # 元データ上で実際に切り出した範囲
    cut_start = first / EMA_SR
    cut_end = last / EMA_SR

    # 先頭に追加した無音時間
    time_offset = pad_before / EMA_SR

    # 切り出し後の全体時間
    output_duration = (
        len(ema_cut) / EMA_SR
    )

    # =====================================================
    # Julius alignmentを同じ範囲へ切り出す
    # =====================================================
    aligned = []

    for s, e, phone in zip(
        starts,
        ends,
        lab["phone"].astype(str)
    ):

        # 元データ内で切り出した範囲に収める
        a = max(s, cut_start)
        b = min(e, cut_end)

        if b <= a:
            raise ValueError(
                f"Label interval missing after cut: "
                f"{phone} ({s}, {e})"
            )

        # 切り出し後の0秒基準へ移動
        # 先頭paddingがある場合は、その分後ろへずらす
        aligned.append(
            (
                a - cut_start + time_offset,
                b - cut_start + time_offset,
                phone
            )
        )

    # =====================================================
    # 追加した無音区間もsilB / silEとして扱う
    # =====================================================

    # silBは必ず0秒から開始
    first_start, first_end, first_phone = aligned[0]

    aligned[0] = (
        0.0,
        first_end,
        first_phone
    )

    # silEは必ず出力音声の最後まで
    last_start, last_end, last_phone = aligned[-1]

    aligned[-1] = (
        last_start,
        output_duration,
        last_phone
    )

    # =====================================================
    # EMA各フレームへ音素ラベルを割り当て
    # =====================================================
    n_frames = len(ema_cut)

    frame_index = np.arange(
        n_frames
    )

    frame_time = (
        frame_index / EMA_SR
    )

    frame_phone = np.empty(
        n_frames,
        dtype=object
    )

    frame_phone[:] = None

    phone_index = np.full(
        n_frames,
        -1,
        dtype=int
    )

    # start <= time < end
    for i, (
        start,
        end,
        phone
    ) in enumerate(aligned):

        mask = (
            (frame_time >= start)
            & (frame_time < end)
        )

        frame_phone[mask] = phone
        phone_index[mask] = i

    # =====================================================
    # 未割り当てフレーム確認
    # =====================================================
    missing = (
        pd.isna(frame_phone)
        | (phone_index < 0)
    )

    if np.any(missing):

        missing_frames = np.where(
            missing
        )[0]

        raise ValueError(
            "Unassigned EMA frames found: "
            f"{speaker}/{name}, "
            f"frames="
            f"{missing_frames[:20].tolist()}"
        )

    # =====================================================
    # silB / silEの長さ確認
    # =====================================================
    silB_duration = (
        aligned[0][1] - aligned[0][0]
    )

    silE_duration = (
        aligned[-1][1] - aligned[-1][0]
    )

    if silB_duration < MARGIN_SEC - 1e-7:
        raise ValueError(
            f"silB < 100 ms after padding: "
            f"{speaker}/{name}, "
            f"{silB_duration:.4f}"
        )

    if silE_duration < MARGIN_SEC - 1e-7:
        raise ValueError(
            f"silE < 100 ms after padding: "
            f"{speaker}/{name}, "
            f"{silE_duration:.4f}"
        )

    # =====================================================
    # 保存先
    # =====================================================
    audio_out = (
        OUTPUT
        / "audio"
        / speaker
        / f"{name}.wav"
    )

    ema_out = (
        OUTPUT
        / "ema"
        / speaker
        / f"{name}_combined.csv"
    )

    lab_out = (
        OUTPUT
        / "julius_alignment"
        / speaker
        / f"{name}.lab"
    )

    frame_out = (
        OUTPUT
        / "frame_aligned"
        / speaker
        / f"{name}_frame_aligned.csv"
    )

    out_paths = [
        audio_out,
        ema_out,
        lab_out,
        frame_out
    ]

    for p in out_paths:
        p.parent.mkdir(
            parents=True,
            exist_ok=True
        )

    # =====================================================
    # 保存
    # =====================================================
    wavfile.write(
        audio_out,
        sr,
        wav_cut
    )

    pd.DataFrame(
        ema_cut
    ).to_csv(
        ema_out,
        index=False,
        header=False
    )

    with lab_out.open(
        "w",
        encoding="utf-8"
    ) as f:

        for a, b, phone in aligned:

            f.write(
                f"{a:.7f} "
                f"{b:.7f} "
                f"{phone}\n"
            )

    # =====================================================
    # EMAフレーム対応データ
    # =====================================================
    frame_df = pd.DataFrame({
        "frame": frame_index,
        "time_sec": frame_time,
        "phone_index": phone_index,
        "phone": frame_phone,
    })

    ema_df = pd.DataFrame(
        ema_cut,
        columns=EMA_COLUMNS
    )

    frame_df = pd.concat(
        [
            frame_df,
            ema_df
        ],
        axis=1
    )

    frame_df.to_csv(
        frame_out,
        index=False,
        encoding="utf-8-sig"
    )

    # =====================================================
    # ログ
    # =====================================================
    return {
        "status": "OK",

        "crop_start_s": cut_start,
        "crop_end_s": cut_end,

        "duration_s": output_duration,

        "wav_samples": len(wav_cut),
        "ema_frames": len(ema_cut),

        "phone_count": len(aligned),

        "silB_duration_s": silB_duration,
        "silE_duration_s": silE_duration,

        "pad_before_frames": pad_before,
        "pad_after_frames": pad_after,

        "pad_before_ms": (
            pad_before / EMA_SR * 1000
        ),

        "pad_after_ms": (
            pad_after / EMA_SR * 1000
        ),

        "message": "",
    }


# =========================================================
# 全話者・全発話
# =========================================================
def main():

    log = []

    for speaker, date in SPEAKERS.items():

        ok = 0

        for number in range(
            1,
            NUM_FILES + 1
        ):

            row = {
                "speaker": speaker,
                "utterance": f"{number:03d}"
            }

            try:

                row.update(
                    process(
                        speaker,
                        date,
                        number
                    )
                )

                ok += 1

            except Exception as exc:

                row.update(
                    status="ERROR",
                    message=str(exc)
                )

            log.append(
                row
            )

        print(
            f"{speaker}: "
            f"{ok}/{NUM_FILES} processed"
        )

    # =====================================================
    # ログ保存
    # =====================================================
    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    dest = (
        OUTPUT
        / "trim_log.csv"
    )

    cols = [
        "speaker",
        "utterance",
        "status",
        "crop_start_s",
        "crop_end_s",
        "duration_s",
        "wav_samples",
        "ema_frames",
        "phone_count",
        "silB_duration_s",
        "silE_duration_s",
        "pad_before_frames",
        "pad_after_frames",
        "pad_before_ms",
        "pad_after_ms",
        "message",
    ]

    with dest.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=cols
        )

        writer.writeheader()
        writer.writerows(log)

    print(
        f"Log saved: {dest}"
    )


if __name__ == "__main__":
    main()
