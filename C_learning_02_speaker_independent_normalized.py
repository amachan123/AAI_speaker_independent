import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# =====================================================================
# 基本設定
# =====================================================================

SPEAKERS = ["M0102", "M0201", "W0401", "W0901"]

EMA_CHANNELS = [
    "UL_y", "UL_z", "LL_y", "LL_z", "LJ_y", "LJ_z",
    "T1_y", "T1_z", "T2_y", "T2_z", "T3_y", "T3_z",
]

TV_CHANNELS = [
    "LA", "LP", "JA", "TTCD", "TTCL",
    "TMCD", "TMCL", "TBCD", "TBCL",
]


# =====================================================================
# パス・split CSV読み込み
# =====================================================================

def resolve_path(value, project_root, manifest_dir):
    path = Path(str(value))

    if path.is_absolute():
        candidate = path
    elif (project_root / path).exists():
        candidate = project_root / path
    else:
        candidate = manifest_dir / path

    if not candidate.exists():
        raise FileNotFoundError(f"ファイルが存在しません: {candidate}")

    return candidate.resolve()


def load_split_csv(path, project_root):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"split CSVが存在しません: {path}")

    df = pd.read_csv(path, dtype={"utterance_id": str})

    required = ["speaker", "utterance_id", "feature_path", "ema_path", "tv_path"]
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise KeyError(f"{path}に必要な列がありません: {missing}")

    df = df.copy()
    df["_speaker"] = df["speaker"].astype(str)

    df["_feature_resolved"] = df["feature_path"].map(
        lambda value: str(resolve_path(value, project_root, path.parent.resolve()))
    )
    df["_ema_original_resolved"] = df["ema_path"].map(
        lambda value: str(resolve_path(value, project_root, path.parent.resolve()))
    )
    df["_tv_original_resolved"] = df["tv_path"].map(
        lambda value: str(resolve_path(value, project_root, path.parent.resolve()))
    )

    return df


# =====================================================================
# データ読み込み
# =====================================================================

def load_feature(path):
    feature = np.asarray(np.load(path), dtype=np.float32)

    if feature.ndim != 2 or feature.shape[1] != 1024:
        raise ValueError(f"音響特徴量shapeが不正です: {path}, {feature.shape} (期待値: [T, 1024])")

    if not np.isfinite(feature).all():
        raise ValueError(f"音響特徴量にNaNまたはInfがあります: {path}")

    return feature


def load_ema(path):
    path = Path(path)

    if path.suffix.lower() == ".npy":
        ema = np.load(path)

    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path, header=None)

        if df.shape[1] == 18:
            ema = df.iloc[:, 6:18].to_numpy()
        elif df.shape[1] == 12:
            ema = df.to_numpy()
        else:
            raise ValueError(f"EMA列数が12または18ではありません: {path}, shape={df.shape}")

    else:
        raise ValueError(f"未対応のEMA形式です: {path}")

    ema = np.asarray(ema, dtype=np.float32)

    if ema.ndim != 2 or ema.shape[1] != 12:
        raise ValueError(f"EMA shapeが不正です: {path}, {ema.shape}")

    if not np.isfinite(ema).all():
        raise ValueError(f"EMAにNaNまたはInfがあります: {path}")

    return ema


def load_tv(path):
    path = Path(path)

    if path.suffix.lower() == ".npy":
        tv = np.load(path)
    else:
        tv = pd.read_csv(path, header=None).to_numpy()

    tv = np.asarray(tv, dtype=np.float32)

    if tv.ndim != 2 or tv.shape[1] != 9:
        raise ValueError(f"TV shapeが不正です: {path}, {tv.shape}")

    if not np.isfinite(tv).all():
        raise ValueError(f"TVにNaNまたはInfがあります: {path}")

    return tv


# =====================================================================
# 話者別スケーラー
# =====================================================================

def calculate_scaler(dataframe, resolved_column, loader, dim):
    total_sum = np.zeros(dim, dtype=np.float64)
    total_square_sum = np.zeros(dim, dtype=np.float64)
    total_frames = 0

    for path in dataframe[resolved_column]:
        values = loader(path).astype(np.float64)

        total_sum += values.sum(axis=0)
        total_square_sum += np.square(values).sum(axis=0)
        total_frames += len(values)

    if total_frames == 0:
        raise ValueError("スケーラー計算対象のデータがありません。")

    mean = total_sum / total_frames
    variance = total_square_sum / total_frames - np.square(mean)
    std = np.sqrt(np.maximum(variance, 1e-8))

    return mean.astype(np.float32), std.astype(np.float32), total_frames


def create_fold_scalers(train_df, test_df, test_speaker):
    ema_scalers = {}
    tv_scalers = {}
    metadata = {}

    # 学習話者：各話者のtrainデータのみから計算
    for speaker in sorted(train_df["_speaker"].unique()):
        rows = train_df[train_df["_speaker"] == speaker]

        ema_mean, ema_std, ema_frames = calculate_scaler(rows, "_ema_original_resolved", load_ema, 12)
        tv_mean, tv_std, tv_frames = calculate_scaler(rows, "_tv_original_resolved", load_tv, 9)

        if ema_frames != tv_frames:
            raise ValueError(
                f"{speaker}のEMAとTVの総フレーム数が一致しません: EMA={ema_frames}, TV={tv_frames}"
            )

        ema_scalers[speaker] = (ema_mean, ema_std)
        tv_scalers[speaker] = (tv_mean, tv_std)

        metadata[speaker] = {
            "source_split": "train",
            "num_utterances": int(len(rows)),
            "num_frames": int(ema_frames),
        }

    # テスト話者：test正解データから計算するOracle条件
    rows = test_df[test_df["_speaker"] == test_speaker]

    if len(rows) == 0:
        raise ValueError(f"test splitに{test_speaker}がありません。")

    ema_mean, ema_std, ema_frames = calculate_scaler(rows, "_ema_original_resolved", load_ema, 12)
    tv_mean, tv_std, tv_frames = calculate_scaler(rows, "_tv_original_resolved", load_tv, 9)

    if ema_frames != tv_frames:
        raise ValueError(
            f"{test_speaker}のEMAとTVの総フレーム数が一致しません: "
            f"EMA={ema_frames}, TV={tv_frames}"
        )

    ema_scalers[test_speaker] = (ema_mean, ema_std)
    tv_scalers[test_speaker] = (tv_mean, tv_std)

    metadata[test_speaker] = {
        "source_split": "test_oracle",
        "num_utterances": int(len(rows)),
        "num_frames": int(ema_frames),
    }

    return ema_scalers, tv_scalers, metadata


# =====================================================================
# スケーラー保存
# =====================================================================

def save_scalers(ema_scalers, tv_scalers, metadata, output_fold_dir):
    scaler_dir = output_fold_dir / "speaker_scalers"
    scaler_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for speaker in sorted(ema_scalers):
        ema_mean, ema_std = ema_scalers[speaker]
        tv_mean, tv_std = tv_scalers[speaker]

        np.savez(
            scaler_dir / f"{speaker}_ema_scaler.npz",
            mean=ema_mean,
            std=ema_std,
            channel_names=np.asarray(EMA_CHANNELS),
        )

        np.savez(
            scaler_dir / f"{speaker}_tv_scaler.npz",
            mean=tv_mean,
            std=tv_std,
            channel_names=np.asarray(TV_CHANNELS),
        )

        row = {"speaker": speaker, **metadata[speaker]}

        for index, channel in enumerate(EMA_CHANNELS):
            row[f"ema_{channel}_mean"] = float(ema_mean[index])
            row[f"ema_{channel}_std"] = float(ema_std[index])

        for index, channel in enumerate(TV_CHANNELS):
            row[f"tv_{channel}_mean"] = float(tv_mean[index])
            row[f"tv_{channel}_std"] = float(tv_std[index])

        rows.append(row)

    pd.DataFrame(rows).to_csv(output_fold_dir / "speaker_scalers.csv", index=False)

    metadata_output = {
        "condition": "speaker-wise EMA and TV normalization",
        "test_speaker_scaler": "oracle scalers calculated from the test split",
        "ema_channels": EMA_CHANNELS,
        "tv_channels": TV_CHANNELS,
        "speakers": metadata,
    }

    with open(output_fold_dir / "normalization_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata_output, f, ensure_ascii=False, indent=2)


# =====================================================================
# 正規化データ・manifest保存
# =====================================================================

def normalize_split(dataframe, split, ema_scalers, tv_scalers, output_fold_dir):
    ema_dir = output_fold_dir / "ema_normalized" / split
    tv_dir = output_fold_dir / "tv_normalized" / split

    ema_dir.mkdir(parents=True, exist_ok=True)
    tv_dir.mkdir(parents=True, exist_ok=True)

    records = []

    for _, row in dataframe.iterrows():
        speaker = row["_speaker"]
        utterance_id = str(row["utterance_id"]).zfill(3)

        feature = load_feature(row["_feature_resolved"])
        ema = load_ema(row["_ema_original_resolved"])
        tv = load_tv(row["_tv_original_resolved"])

        lengths = {"feature": len(feature), "EMA": len(ema), "TV": len(tv)}

        if len(set(lengths.values())) != 1:
            raise ValueError(f"フレーム数不一致: {speaker}/{utterance_id} {lengths}")

        ema_mean, ema_std = ema_scalers[speaker]
        tv_mean, tv_std = tv_scalers[speaker]

        ema_normalized = ((ema - ema_mean) / ema_std).astype(np.float32)
        tv_normalized = ((tv - tv_mean) / tv_std).astype(np.float32)

        if not np.isfinite(ema_normalized).all():
            raise ValueError(f"正規化EMAにNaNまたはInfがあります: {speaker}/{utterance_id}")

        if not np.isfinite(tv_normalized).all():
            raise ValueError(f"正規化TVにNaNまたはInfがあります: {speaker}/{utterance_id}")

        ema_speaker_dir = ema_dir / speaker
        tv_speaker_dir = tv_dir / speaker

        ema_speaker_dir.mkdir(parents=True, exist_ok=True)
        tv_speaker_dir.mkdir(parents=True, exist_ok=True)

        ema_output_path = ema_speaker_dir / f"{speaker}_{utterance_id}_ema_normalized.npy"
        tv_output_path = tv_speaker_dir / f"{speaker}_{utterance_id}_tv_normalized.npy"

        np.save(ema_output_path, ema_normalized)
        np.save(tv_output_path, tv_normalized)

        records.append({
            "split": split,
            "speaker": speaker,
            "utterance_id": utterance_id,
            "num_frames": len(feature),
            "feature_path": str(Path(row["_feature_resolved"]).resolve()),
            "ema_original_path": str(Path(row["_ema_original_resolved"]).resolve()),
            "tv_original_path": str(Path(row["_tv_original_resolved"]).resolve()),
            "ema_normalized_path": str(ema_output_path.resolve()),
            "tv_normalized_path": str(tv_output_path.resolve()),
        })

    pd.DataFrame(records).to_csv(
        output_fold_dir / f"{split}_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )


# =====================================================================
# 正規化確認
# =====================================================================

def check_normalization(dataframe, ema_scalers, tv_scalers, label):
    print(f"\nNormalization check: {label}")

    for speaker in sorted(dataframe["_speaker"].unique()):
        rows = dataframe[dataframe["_speaker"] == speaker]

        ema_mean, ema_std = ema_scalers[speaker]
        tv_mean, tv_std = tv_scalers[speaker]

        ema_values = [
            (load_ema(path) - ema_mean) / ema_std
            for path in rows["_ema_original_resolved"]
        ]

        tv_values = [
            (load_tv(path) - tv_mean) / tv_std
            for path in rows["_tv_original_resolved"]
        ]

        ema_values = np.concatenate(ema_values, axis=0)
        tv_values = np.concatenate(tv_values, axis=0)

        print(speaker)
        print("  EMA mean:", np.round(ema_values.mean(axis=0), 3))
        print("  EMA std :", np.round(ema_values.std(axis=0), 3))
        print("  TV mean :", np.round(tv_values.mean(axis=0), 3))
        print("  TV std  :", np.round(tv_values.std(axis=0), 3))


# =====================================================================
# 1 fold
# =====================================================================

def process_fold(args, test_speaker):
    print("\n" + "=" * 64)
    print(f"Speaker-wise normalization: test speaker = {test_speaker}")
    print("=" * 64)

    input_fold_dir = Path(args.input_root) / f"test_{test_speaker}"
    output_fold_dir = Path(args.output_root) / f"test_{test_speaker}"
    output_fold_dir.mkdir(parents=True, exist_ok=True)

    split_paths = {
        "train": input_fold_dir / "train.csv",
        "validation": input_fold_dir / "validation.csv",
        "test": input_fold_dir / "test.csv",
    }

    project_root = Path(args.project_root).resolve()

    dataframes = {
        split: load_split_csv(path, project_root)
        for split, path in split_paths.items()
    }

    ema_scalers, tv_scalers, metadata = create_fold_scalers(
        dataframes["train"], dataframes["test"], test_speaker
    )

    save_scalers(ema_scalers, tv_scalers, metadata, output_fold_dir)

    for speaker in sorted(ema_scalers):
        print(
            f"scaler: {speaker} | "
            f"source={metadata[speaker]['source_split']} | "
            f"utterances={metadata[speaker]['num_utterances']} | "
            f"frames={metadata[speaker]['num_frames']}"
        )

    if args.check_normalization:
        check_normalization(dataframes["train"], ema_scalers, tv_scalers, "train")
        check_normalization(dataframes["test"], ema_scalers, tv_scalers, "test oracle")

    for split in ["train", "validation", "test"]:
        normalize_split(
            dataframes[split],
            split,
            ema_scalers,
            tv_scalers,
            output_fold_dir,
        )

        print(f"{split}: saved {len(dataframes[split])} utterances")

    print(f"Saved fold: {output_fold_dir.resolve()}")


# =====================================================================
# main
# =====================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--speakers", nargs="+", default=SPEAKERS)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--input-root", default="Data/splits_loso")
    parser.add_argument("--output-root", default="Data/oracle_speaker_normalized")
    parser.add_argument("--check-normalization", action="store_true")

    args = parser.parse_args()

    for speaker in args.speakers:
        process_fold(args, speaker)

    print("\nAll speaker-wise EMA/TV normalized datasets were created.")


if __name__ == "__main__":
    main()
