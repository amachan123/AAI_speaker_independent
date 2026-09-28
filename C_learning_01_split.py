from pathlib import Path
from typing import Dict

import pandas as pd


# =====================================================================
# 1. 基本設定
# =====================================================================

# 話者ごとのAudio・EMAが入っている場所
DATA_ROOT = Path("Data/data")

# TVデータが入っている場所
TV_ROOT = Path("Data/z_TV")

# 分割CSVの保存先
OUTPUT_DIR = Path("Data/splits_loso")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPEAKERS = ["M0102", "M0201", "W0401", "W0901"]

# W0401で除外する発話
INVALID_UTTERANCES: Dict[str, set[int]] = {
    "M0102": set(),
    "M0201": set(),
    "W0401": {17, 20, 29, 36, 191, 211},
    "W0901": set(),
}

# 文番号の分割
#
# 学習:
#   001～250
#   401～503
#
# 検証:
#   251～300
#
# テスト:
#   301～400
#
TRAIN_UTTERANCES = set(range(1, 251)) | set(range(401, 504))
VALIDATION_UTTERANCES = set(range(251, 301))
TEST_UTTERANCES = set(range(301, 401))

ALL_UTTERANCES = set(range(1, 504))


# =====================================================================
# 2. 固定パス生成関数
# =====================================================================

def get_feature_path(speaker: str, utterance: int) -> Path:
    """音響特徴量のパスを返す。"""
    utt_id = f"{utterance:03d}"

    return (
        DATA_ROOT
        / speaker
        / "Audio"
        / "audio_features_250Hz_xlsr_aligned"
        / f"{utt_id}_feat.npy"
    )


def get_ema_path(speaker: str, utterance: int) -> Path:
    """EMAデータのパスを返す。"""
    utt_id = f"{utterance:03d}"

    return (
        DATA_ROOT
        / speaker
        / "EMA"
        / "EMA_aligned_yz_cut_aligned"
        / f"{utt_id}.csv"
    )


def get_tv_path(speaker: str, utterance: int) -> Path:
    """TVデータのパスを返す。"""
    utt_id = f"{utterance:03d}"

    return (
        TV_ROOT
        / f"{speaker}_TV_1000_aligned"
        / f"{utt_id}.csv"
    )


# =====================================================================
# 3. 1発話分のレコード作成
# =====================================================================

def make_record(
    speaker: str,
    utterance: int,
    split: str,
    fold_test_speaker: str,
) -> dict:
    feature_path = get_feature_path(speaker, utterance)
    ema_path = get_ema_path(speaker, utterance)
    tv_path = get_tv_path(speaker, utterance)

    missing_files = []

    if not feature_path.exists():
        missing_files.append("feature")

    if not ema_path.exists():
        missing_files.append("ema")

    if not tv_path.exists():
        missing_files.append("tv")

    return {
        "fold_test_speaker": fold_test_speaker,
        "split": split,
        "speaker": speaker,
        "utterance": utterance,
        "utterance_id": f"{utterance:03d}",
        "feature_path": feature_path.as_posix(),
        "ema_path": ema_path.as_posix(),
        "tv_path": tv_path.as_posix(),
        "all_files_exist": len(missing_files) == 0,
        "missing_files": ",".join(missing_files),
    }


# =====================================================================
# 4. 1つのLOSO foldを作成
# =====================================================================

def build_fold(test_speaker: str) -> None:
    train_speakers = [
        speaker
        for speaker in SPEAKERS
        if speaker != test_speaker
    ]

    fold_dir = OUTPUT_DIR / f"test_{test_speaker}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    train_records = []
    validation_records = []
    test_records = []
    excluded_records = []

    # -----------------------------------------------------------------
    # 学習・検証データ
    # テスト話者以外の3話者を使用
    # -----------------------------------------------------------------
    for speaker in train_speakers:
        invalid_utterances = INVALID_UTTERANCES[speaker]

        # 学習データ
        for utterance in sorted(TRAIN_UTTERANCES):
            if utterance in invalid_utterances:
                excluded_records.append({
                    "speaker": speaker,
                    "utterance": utterance,
                    "utterance_id": f"{utterance:03d}",
                    "reason": "known_invalid_utterance",
                    "intended_split": "train",
                })
                continue

            train_records.append(
                make_record(
                    speaker=speaker,
                    utterance=utterance,
                    split="train",
                    fold_test_speaker=test_speaker,
                )
            )

        # 検証データ
        for utterance in sorted(VALIDATION_UTTERANCES):
            if utterance in invalid_utterances:
                excluded_records.append({
                    "speaker": speaker,
                    "utterance": utterance,
                    "utterance_id": f"{utterance:03d}",
                    "reason": "known_invalid_utterance",
                    "intended_split": "validation",
                })
                continue

            validation_records.append(
                make_record(
                    speaker=speaker,
                    utterance=utterance,
                    split="validation",
                    fold_test_speaker=test_speaker,
                )
            )

        # 学習側話者の301～400は使用しない
        for utterance in sorted(TEST_UTTERANCES):
            excluded_records.append({
                "speaker": speaker,
                "utterance": utterance,
                "utterance_id": f"{utterance:03d}",
                "reason": "reserved_test_text",
                "intended_split": "unused",
            })

    # -----------------------------------------------------------------
    # テストデータ
    # テスト話者の301～400だけを使用
    # -----------------------------------------------------------------
    test_invalid_utterances = INVALID_UTTERANCES[test_speaker]

    for utterance in sorted(TEST_UTTERANCES):
        if utterance in test_invalid_utterances:
            excluded_records.append({
                "speaker": test_speaker,
                "utterance": utterance,
                "utterance_id": f"{utterance:03d}",
                "reason": "known_invalid_utterance",
                "intended_split": "test",
            })
            continue

        test_records.append(
            make_record(
                speaker=test_speaker,
                utterance=utterance,
                split="test",
                fold_test_speaker=test_speaker,
            )
        )

    # テスト話者の301～400以外はすべて未使用
    for utterance in sorted(ALL_UTTERANCES - TEST_UTTERANCES):
        if utterance in test_invalid_utterances:
            reason = "known_invalid_utterance"
        else:
            reason = "held_out_speaker_not_used"

        excluded_records.append({
            "speaker": test_speaker,
            "utterance": utterance,
            "utterance_id": f"{utterance:03d}",
            "reason": reason,
            "intended_split": "unused",
        })

    # -----------------------------------------------------------------
    # DataFrame化
    # -----------------------------------------------------------------
    train_df = pd.DataFrame(train_records)
    validation_df = pd.DataFrame(validation_records)
    test_df = pd.DataFrame(test_records)
    excluded_df = pd.DataFrame(excluded_records)

    used_df = pd.concat(
        [train_df, validation_df, test_df],
        ignore_index=True,
    )

    missing_df = used_df[
        ~used_df["all_files_exist"]
    ].copy()

    # -----------------------------------------------------------------
    # CSV保存
    # -----------------------------------------------------------------
    train_df.to_csv(
        fold_dir / "train.csv",
        index=False,
        encoding="utf-8-sig",
    )

    validation_df.to_csv(
        fold_dir / "validation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    test_df.to_csv(
        fold_dir / "test.csv",
        index=False,
        encoding="utf-8-sig",
    )

    excluded_df.to_csv(
        fold_dir / "excluded.csv",
        index=False,
        encoding="utf-8-sig",
    )

    missing_df.to_csv(
        fold_dir / "missing_files.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------------------
    # 件数集計
    # -----------------------------------------------------------------
    summary_rows = []

    for split_name, split_df in [
        ("train", train_df),
        ("validation", validation_df),
        ("test", test_df),
    ]:
        for speaker in SPEAKERS:
            count = int(
                (split_df["speaker"] == speaker).sum()
            )

            if count > 0:
                summary_rows.append({
                    "fold_test_speaker": test_speaker,
                    "split": split_name,
                    "speaker": speaker,
                    "count": count,
                })

        summary_rows.append({
            "fold_test_speaker": test_speaker,
            "split": split_name,
            "speaker": "TOTAL",
            "count": len(split_df),
        })

    summary_df = pd.DataFrame(summary_rows)

    summary_df.to_csv(
        fold_dir / "summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------------------
    # コンソール表示
    # -----------------------------------------------------------------
    print("=" * 72)
    print(f"テスト話者 : {test_speaker}")
    print(f"学習話者   : {', '.join(train_speakers)}")
    print(f"学習件数   : {len(train_df)}")
    print(f"検証件数   : {len(validation_df)}")
    print(f"テスト件数 : {len(test_df)}")
    print(f"ファイル不足件数 : {len(missing_df)}")

    if not missing_df.empty:
        print("\nファイル不足があります。")
        print(
            missing_df[
                [
                    "split",
                    "speaker",
                    "utterance_id",
                    "missing_files",
                ]
            ]
            .head(30)
            .to_string(index=False)
        )


# =====================================================================
# 5. 分割内容の検証
# =====================================================================

def validate_fold(test_speaker: str) -> None:
    fold_dir = OUTPUT_DIR / f"test_{test_speaker}"

    train_df = pd.read_csv(fold_dir / "train.csv")
    validation_df = pd.read_csv(fold_dir / "validation.csv")
    test_df = pd.read_csv(fold_dir / "test.csv")

    def make_sample_keys(df: pd.DataFrame) -> set[tuple[str, int]]:
        return set(
            zip(
                df["speaker"].astype(str),
                df["utterance"].astype(int),
            )
        )

    train_keys = make_sample_keys(train_df)
    validation_keys = make_sample_keys(validation_df)
    test_keys = make_sample_keys(test_df)

    # 同一話者・同一文の重複確認
    assert train_keys.isdisjoint(validation_keys), (
        "trainとvalidationに同一サンプルがあります。"
    )

    assert train_keys.isdisjoint(test_keys), (
        "trainとtestに同一サンプルがあります。"
    )

    assert validation_keys.isdisjoint(test_keys), (
        "validationとtestに同一サンプルがあります。"
    )

    # テスト話者の混入確認
    assert test_speaker not in set(train_df["speaker"]), (
        f"trainにテスト話者 {test_speaker} が含まれています。"
    )

    assert test_speaker not in set(validation_df["speaker"]), (
        f"validationにテスト話者 {test_speaker} が含まれています。"
    )

    assert set(test_df["speaker"]) == {test_speaker}, (
        "testにテスト話者以外が含まれています。"
    )

    # 文番号範囲の確認
    assert set(train_df["utterance"]).issubset(
        TRAIN_UTTERANCES
    ), "trainに設定範囲外の文番号があります。"

    assert set(validation_df["utterance"]).issubset(
        VALIDATION_UTTERANCES
    ), "validationに設定範囲外の文番号があります。"

    assert set(test_df["utterance"]).issubset(
        TEST_UTTERANCES
    ), "testに設定範囲外の文番号があります。"

    # 学習・検証・テスト間で文番号自体も分離されているか確認
    train_text_ids = set(train_df["utterance"].astype(int))
    validation_text_ids = set(
        validation_df["utterance"].astype(int)
    )
    test_text_ids = set(test_df["utterance"].astype(int))

    assert train_text_ids.isdisjoint(validation_text_ids), (
        "trainとvalidationで文番号が重複しています。"
    )

    assert train_text_ids.isdisjoint(test_text_ids), (
        "trainとtestで文番号が重複しています。"
    )

    assert validation_text_ids.isdisjoint(test_text_ids), (
        "validationとtestで文番号が重複しています。"
    )

    # W0401の既知不具合データが使用されていないか確認
    invalid_w0401 = INVALID_UTTERANCES["W0401"]

    for split_name, split_df in [
        ("train", train_df),
        ("validation", validation_df),
        ("test", test_df),
    ]:
        w0401_rows = split_df[
            split_df["speaker"] == "W0401"
        ]

        used_invalid = (
            set(w0401_rows["utterance"].astype(int))
            & invalid_w0401
        )

        assert not used_invalid, (
            f"{split_name}にW0401の不具合文が含まれています: "
            f"{sorted(used_invalid)}"
        )

    print(f"分割チェック成功: test={test_speaker}")


# =====================================================================
# 6. 期待件数の表示
# =====================================================================

def print_expected_counts() -> None:
    print("\n期待される基本件数")
    print("-" * 40)
    print("欠損なし話者の学習文数 : 353")
    print("W0401の学習文数        : 347")
    print("1話者あたりの検証文数  : 50")
    print("テスト文数              : 100")
    print()

    for test_speaker in SPEAKERS:
        train_speakers = [
            s for s in SPEAKERS
            if s != test_speaker
        ]

        expected_train = 0

        for speaker in train_speakers:
            speaker_count = len(TRAIN_UTTERANCES)

            invalid_in_train = (
                INVALID_UTTERANCES[speaker]
                & TRAIN_UTTERANCES
            )

            speaker_count -= len(invalid_in_train)
            expected_train += speaker_count

        expected_validation = (
            len(train_speakers)
            * len(VALIDATION_UTTERANCES)
        )

        invalid_validation = sum(
            len(
                INVALID_UTTERANCES[speaker]
                & VALIDATION_UTTERANCES
            )
            for speaker in train_speakers
        )

        expected_validation -= invalid_validation

        expected_test = (
            len(TEST_UTTERANCES)
            - len(
                INVALID_UTTERANCES[test_speaker]
                & TEST_UTTERANCES
            )
        )

        print(
            f"test={test_speaker}: "
            f"train={expected_train}, "
            f"validation={expected_validation}, "
            f"test={expected_test}"
        )


# =====================================================================
# 7. 実行
# =====================================================================

def main() -> None:
    print_expected_counts()

    for test_speaker in SPEAKERS:
        build_fold(test_speaker)
        validate_fold(test_speaker)

    print("\nすべてのLOSO分割を作成しました。")
    print(f"保存先: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
