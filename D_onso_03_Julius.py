from pathlib import Path
import shutil
import subprocess

# =========================================================
# 設定
# =========================================================

SEGMENTATION_KIT = Path("/home/amahori/github/segmentation-kit")

WORK_ROOT = Path("Data/IPAlabel/JuliusWork_sp")

OUTPUT_ROOT = Path("Data/IPAlabel/JuliusAlignment_sp")

SPEAKERS = ["W0301", "W0501", "W0701", "W0801"]

UTTERANCE_START = 1
UTTERANCE_END = 503


# =========================================================
# segmentation-kit/wav を空にする
# =========================================================
def clean_kit_wav_dir(kit_wav_dir):

    extensions = {".wav", ".txt", ".lab", ".log", ".dfa", ".dict"}

    for path in kit_wav_dir.iterdir():

        if (
            path.is_file()
            and path.suffix in extensions
        ):
            path.unlink()


# =========================================================
# 1発話だけ Julius 実行
# =========================================================
def run_one_utterance(
    speaker,
    utt_id,
    src_dir,
    dst_dir,
    kit_wav_dir,
):

    src_wav = (
        src_dir
        / f"{utt_id}.wav"
    )

    src_txt = (
        src_dir
        / f"{utt_id}.txt"
    )

    # -----------------------------------------
    # 入力確認
    # -----------------------------------------
    if not src_wav.exists():

        print(
            f"[MISSING WAV] "
            f"{speaker}/{utt_id}"
        )

        return False

    if not src_txt.exists():

        print(
            f"[MISSING TXT] "
            f"{speaker}/{utt_id}"
        )

        return False

    # -----------------------------------------
    # Julius作業フォルダを空にする
    # -----------------------------------------
    clean_kit_wav_dir(
        kit_wav_dir
    )

    # -----------------------------------------
    # wav + txtコピー
    # -----------------------------------------
    shutil.copy2(
        src_wav,
        kit_wav_dir
        / f"{utt_id}.wav",
    )

    shutil.copy2(
        src_txt,
        kit_wav_dir
        / f"{utt_id}.txt",
    )

    # -----------------------------------------
    # Julius実行
    # -----------------------------------------
    result = subprocess.run(
        [
            "perl",
            "segment_julius.pl",
        ],
        cwd=SEGMENTATION_KIT,
        text=True,
        capture_output=True,
    )

    # segmentation-kit 自体が異常終了
    if result.returncode != 0:

        print(
            f"[ERROR] "
            f"{speaker}/{utt_id}"
        )

        print(result.stdout)
        print(result.stderr)

        return False

    # -----------------------------------------
    # 出力確認
    # -----------------------------------------
    lab_path = (
        kit_wav_dir
        / f"{utt_id}.lab"
    )

    log_path = (
        kit_wav_dir
        / f"{utt_id}.log"
    )

    # labが存在しない / 空
    if (
        not lab_path.exists()
        or lab_path.stat().st_size == 0
    ):

        print(
            f"[FAILED/EMPTY] "
            f"{speaker}/{utt_id}"
        )

        # logだけでも保存
        if log_path.exists():

            shutil.copy2(
                log_path,
                dst_dir
                / f"{utt_id}.log",
            )

        return False

    # -----------------------------------------
    # 結果保存
    # -----------------------------------------
    shutil.copy2(
        lab_path,
        dst_dir
        / f"{utt_id}.lab",
    )

    if log_path.exists():

        shutil.copy2(
            log_path,
            dst_dir
            / f"{utt_id}.log",
        )

    return True


# =========================================================
# 1話者分
# =========================================================
def run_one_speaker(speaker):

    src_dir = (
        WORK_ROOT
        / speaker
        / "wav"
    )

    dst_dir = (
        OUTPUT_ROOT
        / speaker
    )

    dst_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    kit_wav_dir = (
        SEGMENTATION_KIT
        / "wav"
    )

    if not src_dir.exists():

        raise FileNotFoundError(
            f"入力フォルダなし: {src_dir}"
        )

    success = 0
    failed = 0

    print()
    print("==============================")
    print(f"{speaker} 開始")
    print("==============================")

    for utt_num in range(
        UTTERANCE_START,
        UTTERANCE_END + 1,
    ):

        utt_id = f"{utt_num:03d}"

        ok = run_one_utterance(
            speaker=speaker,
            utt_id=utt_id,
            src_dir=src_dir,
            dst_dir=dst_dir,
            kit_wav_dir=kit_wav_dir,
        )

        if ok:

            success += 1

            print(
                f"[OK] "
                f"{speaker}/{utt_id}"
            )

        else:

            failed += 1

    print()
    print(
        f"{speaker}: "
        f"成功={success}, "
        f"失敗={failed}"
    )

    return success, failed


# =========================================================
# メイン
# =========================================================
def main():

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not SEGMENTATION_KIT.exists():

        raise FileNotFoundError(
            f"segmentation-kitなし: "
            f"{SEGMENTATION_KIT}"
        )

    total_success = 0
    total_failed = 0

    for speaker in SPEAKERS:

        success, failed = (
            run_one_speaker(
                speaker
            )
        )

        total_success += success
        total_failed += failed

    print()
    print("================================")
    print("全話者 Alignment 終了")
    print("================================")

    print(
        f"成功: {total_success}"
    )

    print(
        f"失敗: {total_failed}"
    )

    print(
        f"保存先: {OUTPUT_ROOT}"
    )


if __name__ == "__main__":
    main()
