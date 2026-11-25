from pathlib import Path
import pandas as pd

# =========================================================
# 設定
# =========================================================
ROOT = Path("Data/IPA_preprocessed")

INPUT_ROOT = ROOT / "frame_aligned"
OUTPUT_ROOT = ROOT / "frame_aligned_12d"

SPEAKERS = [
    "W0301",
    "W0501",
    "W0701",
    "W0801",
]

NUM_FILES = 503

# 学習に使用する12次元EMA
EMA_12D_COLUMNS = [
    "UL_y", "UL_z",
    "LL_y", "LL_z",
    "LJ_y", "LJ_z",
    "T1_y", "T1_z",
    "T2_y", "T2_z",
    "T3_y", "T3_z",
]

# EMA以外で残す情報
INFO_COLUMNS = [
    "frame",
    "time_sec",
    "phone_index",
    "phone",
]


# =========================================================
# 1ファイル処理
# =========================================================
def process_file(speaker, number):

    name = f"{number:03d}"

    input_path = (
        INPUT_ROOT
        / speaker
        / f"{name}_frame_aligned.csv"
    )

    output_path = (
        OUTPUT_ROOT
        / speaker
        / f"{name}_frame_aligned.csv"
    )

    if not input_path.is_file():
        print(f"[MISSING] {input_path}")
        return False

    df = pd.read_csv(input_path)

    # 必要な列が存在するか確認
    required_columns = (
        INFO_COLUMNS
        + EMA_12D_COLUMNS
    )

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{speaker}/{name}: "
            f"missing columns = {missing}"
        )

    # 必要な列だけ取り出す
    df_12d = df[
        INFO_COLUMNS
        + EMA_12D_COLUMNS
    ].copy()

    # 保存
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df_12d.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    return True


# =========================================================
# 全話者・全発話
# =========================================================
def main():

    for speaker in SPEAKERS:

        count = 0

        for number in range(
            1,
            NUM_FILES + 1
        ):
            if process_file(
                speaker,
                number
            ):
                count += 1

        print(
            f"{speaker}: "
            f"{count}/{NUM_FILES} processed"
        )


if __name__ == "__main__":
    main()
