from pathlib import Path
import shutil

# =========================================================
# 設定
# =========================================================
ROOT = Path("Data")

SPEAKERS = {
    "W0301": "20191201",
    "W0501": "20201209",
    "W0701": "20231226",
    "W0801": "20240312",
}

PHONEME_DIR = ROOT / "Onso_row" / "phoneme"
WORK_ROOT = ROOT / "IPAlabel" / "JuliusWork_sp"

NUM_FILES = 503


# =========================================================
# 1話者分
# =========================================================
def prepare_speaker(speaker, date):

    phoneme_file = PHONEME_DIR / f"{speaker}.txt"

    wav_dir = (
        ROOT
        / "data"
        / speaker
        / f"{date}hpf_wav_16k"
    )

    work_dir = (
        WORK_ROOT
        / speaker
        / "wav"
    )

    work_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # -----------------------------------------
    # 音素列読み込み
    # -----------------------------------------
    phone_dict = {}

    with phoneme_file.open(
        "r",
        encoding="utf-8-sig"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            utt_id, phones = line.split(
                maxsplit=1
            )

            phone_dict[utt_id] = phones

    # -----------------------------------------
    # 001～503
    # -----------------------------------------
    success = 0

    for number in range(1, NUM_FILES + 1):

        utt_id = f"{number:03d}"

        src_wav = (
            wav_dir
            / f"{utt_id}.wav"
        )

        dst_wav = (
            work_dir
            / f"{utt_id}.wav"
        )

        dst_txt = (
            work_dir
            / f"{utt_id}.txt"
        )

        if not src_wav.is_file():

            print(
                f"[MISSING WAV] "
                f"{speaker}/{utt_id}"
            )

            continue

        if utt_id not in phone_dict:

            print(
                f"[MISSING PHONE] "
                f"{speaker}/{utt_id}"
            )

            continue

        shutil.copy2(
            src_wav,
            dst_wav
        )

        dst_txt.write_text(
            phone_dict[utt_id] + "\n",
            encoding="utf-8"
        )

        success += 1

    print(
        f"{speaker}: "
        f"{success}/{NUM_FILES} prepared"
    )


# =========================================================
# main
# =========================================================
def main():

    for speaker, date in SPEAKERS.items():

        prepare_speaker(
            speaker,
            date
        )


if __name__ == "__main__":
    main()
