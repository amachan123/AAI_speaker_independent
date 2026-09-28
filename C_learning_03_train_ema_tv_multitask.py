#EMAデータとTVsのマルチタスク学習
#lamdaはでTVsの重み

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
    pad_sequence,
)
from torch.utils.data import DataLoader, Dataset


SPEAKERS = ["M0102", "M0201", "W0401", "W0901"]

EMA_CHANNELS = [
    "UL_y", "UL_z", "LL_y", "LL_z", "LJ_y", "LJ_z",
    "T1_y", "T1_z", "T2_y", "T2_z", "T3_y", "T3_z",
]
TV_CHANNELS = [
    "LA", "LP", "JA", "TTCD", "TTCL",
    "TMCD", "TMCL", "TBCD", "TBCL",
]

FEATURE_DIM = 1024
EMA_DIM = 12
TV_DIM = 9


# =====================================================================
# 基本処理
# =====================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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

    return str(candidate.resolve())


def load_manifest(path, project_root):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"manifestが存在しません: {path}")

    dataframe = pd.read_csv(path)
    required = [
        "speaker",
        "utterance_id",
        "num_frames",
        "feature_path",
        "ema_original_path",
        "ema_normalized_path",
        "tv_original_path",
        "tv_normalized_path",
    ]
    missing = [column for column in required if column not in dataframe.columns]
    if missing:
        raise KeyError(f"{path}に必要な列がありません: {missing}")

    dataframe = dataframe.copy()
    dataframe["_speaker"] = dataframe["speaker"].astype(str)

    for column in [
        "feature_path",
        "ema_original_path",
        "ema_normalized_path",
        "tv_original_path",
        "tv_normalized_path",
    ]:
        dataframe[column] = dataframe[column].map(
            lambda value: resolve_path(
                value,
                project_root,
                path.parent.resolve(),
            )
        )

    return dataframe


# =====================================================================
# データ読み込み
# =====================================================================

def load_feature(path):
    feature = np.asarray(np.load(path), dtype=np.float32)

    if feature.ndim != 2 or feature.shape[1] != FEATURE_DIM:
        raise ValueError(
            f"音声特徴量shapeが不正です: {path}, {feature.shape} "
            f"(期待値: [T, {FEATURE_DIM}])"
        )
    if not np.isfinite(feature).all():
        raise ValueError(f"音声特徴量にNaNまたはInfがあります: {path}")

    return feature


def load_normalized(path, expected_dim, name):
    values = np.asarray(np.load(path), dtype=np.float32)

    if values.ndim != 2 or values.shape[1] != expected_dim:
        raise ValueError(
            f"{name} shapeが不正です: {path}, {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"{name}にNaNまたはInfがあります: {path}")

    return values


def load_speaker_scalers(fold_dir, target, expected_dim, expected_channels):
    scaler_dir = Path(fold_dir) / "speaker_scalers"
    pattern = f"*_{target}_scaler.npz"
    suffix = f"_{target}_scaler.npz"
    scalers = {}

    for path in sorted(scaler_dir.glob(pattern)):
        speaker = path.name.removesuffix(suffix)

        with np.load(path, allow_pickle=False) as values:
            required = {"mean", "std", "channel_names"}
            missing = required - set(values.files)
            if missing:
                raise KeyError(
                    f"{path}に必要なキーがありません: {sorted(missing)}"
                )

            mean = np.asarray(values["mean"], dtype=np.float32)
            std = np.asarray(values["std"], dtype=np.float32)
            channels = [
                str(value)
                for value in values["channel_names"].tolist()
            ]

        if mean.shape != (expected_dim,) or std.shape != (expected_dim,):
            raise ValueError(
                f"{target.upper()} scaler shapeが不正です: {path}, "
                f"mean={mean.shape}, std={std.shape}"
            )
        if channels != expected_channels:
            raise ValueError(
                f"チャネル順が一致しません: {path}\n"
                f"saved={channels}\nexpected={expected_channels}"
            )
        if (
            not np.isfinite(mean).all()
            or not np.isfinite(std).all()
            or np.any(std <= 0)
        ):
            raise ValueError(f"スケーラー値が不正です: {path}")

        scalers[speaker] = (mean, std)

    if not scalers:
        raise FileNotFoundError(
            f"{target.upper()} scalerが見つかりません: "
            f"{scaler_dir / pattern}"
        )

    return scalers


# =====================================================================
# Dataset
# =====================================================================

class SpeakerWiseMultiTaskDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        speaker = row["_speaker"]

        feature = load_feature(row["feature_path"])
        ema_normalized = load_normalized(
            row["ema_normalized_path"], EMA_DIM, "正規化EMA"
        )
        tv_normalized = load_normalized(
            row["tv_normalized_path"], TV_DIM, "正規化TV"
        )

        expected = int(row["num_frames"])
        lengths = [len(feature), len(ema_normalized), len(tv_normalized)]

        if len(set(lengths)) != 1 or lengths[0] != expected:
            raise ValueError(
                f"フレーム数不一致: speaker={speaker}, "
                f"utterance={row['utterance_id']}, "
                f"manifest={expected}, feature={lengths[0]}, "
                f"EMA={lengths[1]}, TV={lengths[2]}"
            )

        return {
            "feature": torch.from_numpy(feature),
            "ema": torch.from_numpy(ema_normalized),
            "tv": torch.from_numpy(tv_normalized),
            "length": len(feature),
            "speaker": speaker,
            "utterance_id": row["utterance_id"],
        }


def collate_fn(batch):
    return {
        "feature": pad_sequence(
            [item["feature"] for item in batch],
            batch_first=True,
        ),
        "ema": pad_sequence(
            [item["ema"] for item in batch],
            batch_first=True,
        ),
        "tv": pad_sequence(
            [item["tv"] for item in batch],
            batch_first=True,
        ),
        "lengths": torch.tensor(
            [item["length"] for item in batch],
            dtype=torch.long,
        ),
        "speakers": [item["speaker"] for item in batch],
        "utterance_ids": [item["utterance_id"] for item in batch],
    }


def make_loader(dataframe, args, shuffle, device, generator=None):
    return DataLoader(
        SpeakerWiseMultiTaskDataset(dataframe),
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        generator=generator,
    )


# =====================================================================
# モデル
# =====================================================================

class MultiTaskBiLSTM(nn.Module):
    def __init__(
        self,
        input_dim=FEATURE_DIM,
        hidden_size=256,
        num_layers=2,
        dropout=0.2,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        shared_dim = hidden_size * 2
        self.ema_head = nn.Linear(shared_dim, EMA_DIM)
        self.tv_head = nn.Linear(shared_dim, TV_DIM)

    def forward(self, feature, lengths):
        packed = pack_padded_sequence(
            feature,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.lstm(packed)
        shared, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
        )

        ema = self.ema_head(shared)
        tv = self.tv_head(shared)
        return ema, tv


# =====================================================================
# 損失・学習
# =====================================================================

def masked_squared_sum(prediction, target, lengths):
    time = torch.arange(
        prediction.shape[1],
        device=prediction.device,
    )[None, :]
    mask = time < lengths[:, None]

    difference = prediction[mask] - target[mask]
    return torch.square(difference).sum(), difference.numel()


def run_epoch(
    model,
    loader,
    device,
    lambda_tv,
    optimizer=None,
    clip_norm=5.0,
):
    training = optimizer is not None
    model.train(training)

    ema_square_sum = 0.0
    ema_values = 0
    tv_square_sum = 0.0
    tv_values = 0

    for batch in loader:
        feature = batch["feature"].to(device, non_blocking=True)
        ema_target = batch["ema"].to(device, non_blocking=True)
        tv_target = batch["tv"].to(device, non_blocking=True)
        lengths = batch["lengths"].to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            ema_prediction, tv_prediction = model(feature, lengths)

            ema_sq, ema_count = masked_squared_sum(
                ema_prediction,
                ema_target,
                lengths,
            )
            tv_sq, tv_count = masked_squared_sum(
                tv_prediction,
                tv_target,
                lengths,
            )

            ema_loss = ema_sq / ema_count
            tv_loss = tv_sq / tv_count
            total_loss = ema_loss + lambda_tv * tv_loss

            if training:
                total_loss.backward()
                clip_grad_norm_(model.parameters(), clip_norm)
                optimizer.step()

        ema_square_sum += ema_sq.detach().item()
        ema_values += ema_count
        tv_square_sum += tv_sq.detach().item()
        tv_values += tv_count

    ema_mse = ema_square_sum / ema_values
    tv_mse = tv_square_sum / tv_values

    return {
        "ema_mse": ema_mse,
        "tv_mse": tv_mse,
        "total_loss": ema_mse + lambda_tv * tv_mse,
    }


# =====================================================================
# 物理単位評価
# =====================================================================

@torch.no_grad()
def evaluate_test(
    model,
    loader,
    device,
    ema_scalers,
    tv_scalers,
):
    model.eval()

    ema_normalized_square = 0.0
    tv_normalized_square = 0.0
    ema_physical_square = torch.zeros(
        EMA_DIM,
        dtype=torch.float64,
        device=device,
    )
    tv_physical_square = torch.zeros(
        TV_DIM,
        dtype=torch.float64,
        device=device,
    )
    total_frames = 0
    utterance_rows = []

    for batch in loader:
        feature = batch["feature"].to(device, non_blocking=True)
        ema_target = batch["ema"].to(device, non_blocking=True)
        tv_target = batch["tv"].to(device, non_blocking=True)
        lengths = batch["lengths"].to(device, non_blocking=True)

        ema_prediction, tv_prediction = model(feature, lengths)

        for batch_index, speaker in enumerate(batch["speakers"]):
            length = int(lengths[batch_index].item())

            ema_pred_norm = ema_prediction[batch_index, :length]
            ema_true_norm = ema_target[batch_index, :length]
            tv_pred_norm = tv_prediction[batch_index, :length]
            tv_true_norm = tv_target[batch_index, :length]

            ema_mean, ema_std = ema_scalers[speaker]
            tv_mean, tv_std = tv_scalers[speaker]

            ema_mean = torch.as_tensor(
                ema_mean,
                dtype=torch.float32,
                device=device,
            )
            ema_std = torch.as_tensor(
                ema_std,
                dtype=torch.float32,
                device=device,
            )
            tv_mean = torch.as_tensor(
                tv_mean,
                dtype=torch.float32,
                device=device,
            )
            tv_std = torch.as_tensor(
                tv_std,
                dtype=torch.float32,
                device=device,
            )

            ema_pred_phys = ema_pred_norm * ema_std + ema_mean
            ema_true_phys = ema_true_norm * ema_std + ema_mean
            tv_pred_phys = tv_pred_norm * tv_std + tv_mean
            tv_true_phys = tv_true_norm * tv_std + tv_mean

            ema_norm_sq = torch.square(
                ema_pred_norm - ema_true_norm
            ).double()
            tv_norm_sq = torch.square(
                tv_pred_norm - tv_true_norm
            ).double()
            ema_phys_sq = torch.square(
                ema_pred_phys - ema_true_phys
            ).double()
            tv_phys_sq = torch.square(
                tv_pred_phys - tv_true_phys
            ).double()

            ema_normalized_square += ema_norm_sq.sum().item()
            tv_normalized_square += tv_norm_sq.sum().item()
            ema_physical_square += ema_phys_sq.sum(dim=0)
            tv_physical_square += tv_phys_sq.sum(dim=0)
            total_frames += length

            utterance_rows.append({
                "speaker": speaker,
                "utterance_id": batch["utterance_ids"][batch_index],
                "num_frames": length,
                "ema_normalized_mse": float(ema_norm_sq.mean().item()),
                "tv_normalized_mse": float(tv_norm_sq.mean().item()),
                "ema_mean_physical_rmse": float(
                    torch.sqrt(ema_phys_sq.mean(dim=0)).mean().item()
                ),
                "tv_mean_physical_rmse": float(
                    torch.sqrt(tv_phys_sq.mean(dim=0)).mean().item()
                ),
            })

    ema_channel_rmse = torch.sqrt(
        ema_physical_square / total_frames
    ).cpu().numpy()
    tv_channel_rmse = torch.sqrt(
        tv_physical_square / total_frames
    ).cpu().numpy()

    return {
        "ema_normalized_mse": (
            ema_normalized_square / (total_frames * EMA_DIM)
        ),
        "tv_normalized_mse": (
            tv_normalized_square / (total_frames * TV_DIM)
        ),
        "ema_mean_physical_rmse": float(ema_channel_rmse.mean()),
        "ema_global_physical_rmse": float(
            torch.sqrt(
                ema_physical_square.sum()
                / (total_frames * EMA_DIM)
            ).cpu()
        ),
        "tv_mean_physical_rmse": float(tv_channel_rmse.mean()),
        "tv_global_physical_rmse": float(
            torch.sqrt(
                tv_physical_square.sum()
                / (total_frames * TV_DIM)
            ).cpu()
        ),
        "ema_channel_rmse": ema_channel_rmse,
        "tv_channel_rmse": tv_channel_rmse,
        "utterance_rows": utterance_rows,
    }


# =====================================================================
# 1 fold
# =====================================================================

def run_fold(args, test_speaker, device):
    print(f"\n{'=' * 68}")
    print(f"Speaker-wise EMA+TV multitask: test speaker = {test_speaker}")
    print("=" * 68)

    fold_dir = Path(args.input_root) / f"test_{test_speaker}"
    project_root = Path(args.project_root).resolve()

    dataframes = {
        "train": load_manifest(
            fold_dir / "train_manifest.csv",
            project_root,
        ),
        "validation": load_manifest(
            fold_dir / "validation_manifest.csv",
            project_root,
        ),
        "test": load_manifest(
            fold_dir / "test_manifest.csv",
            project_root,
        ),
    }

    ema_scalers = load_speaker_scalers(
        fold_dir,
        target="ema",
        expected_dim=EMA_DIM,
        expected_channels=EMA_CHANNELS,
    )
    tv_scalers = load_speaker_scalers(
        fold_dir,
        target="tv",
        expected_dim=TV_DIM,
        expected_channels=TV_CHANNELS,
    )

    result_dir = (
        Path(args.result_root)
        / f"lambda_{args.lambda_tv:g}"
        / f"test_{test_speaker}"
    )
    result_dir.mkdir(parents=True, exist_ok=True)

    # DataLoaderのshuffle専用乱数生成器。
    # モデルにTV headが追加されても、乱数消費の影響で
    # trainのミニバッチ順序が変わらないように固定する。
    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)

    loaders = {
        "train": make_loader(
            dataframes["train"],
            args,
            True,
            device,
            generator=train_generator,
        ),
        "validation": make_loader(
            dataframes["validation"],
            args,
            False,
            device,
        ),
        "test": make_loader(
            dataframes["test"],
            args,
            False,
            device,
        ),
    }

    model = MultiTaskBiLSTM(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    # 前回設定を維持：Adam、learning rate=1e-3。
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    checkpoint_path = result_dir / "best_model.pt"
    best_val_ema = float("inf")
    best_val_tv = float("inf")
    best_epoch = 0
    wait = 0
    history = []

    for epoch in range(1, args.max_epochs + 1):
        train_metrics = run_epoch(
            model,
            loaders["train"],
            device,
            args.lambda_tv,
            optimizer=optimizer,
            clip_norm=args.clip_norm,
        )
        validation_metrics = run_epoch(
            model,
            loaders["validation"],
            device,
            args.lambda_tv,
        )

        history.append({
            "epoch": epoch,
            "train_ema_normalized_mse": train_metrics["ema_mse"],
            "train_tv_normalized_mse": train_metrics["tv_mse"],
            "train_total_loss": train_metrics["total_loss"],
            "val_ema_normalized_mse": validation_metrics["ema_mse"],
            "val_tv_normalized_mse": validation_metrics["tv_mse"],
            "val_total_loss": validation_metrics["total_loss"],
        })

        print(
            f"Epoch {epoch:4d} | "
            f"train EMA {train_metrics['ema_mse']:.6f} | "
            f"train TV {train_metrics['tv_mse']:.6f} | "
            f"val EMA {validation_metrics['ema_mse']:.6f} | "
            f"val TV {validation_metrics['tv_mse']:.6f}"
        )

        # 主目的がEMAなので、best checkpointはvalidation EMA MSEで選択。
        if validation_metrics["ema_mse"] < best_val_ema:
            best_val_ema = validation_metrics["ema_mse"]
            best_val_tv = validation_metrics["tv_mse"]
            best_epoch = epoch
            wait = 0

            torch.save({
                "model_state_dict": model.state_dict(),
                "best_epoch": best_epoch,
                "best_val_ema_normalized_mse": best_val_ema,
                "best_val_tv_normalized_mse": best_val_tv,
                "test_speaker": test_speaker,
                "lambda_tv": args.lambda_tv,
                "learning_rate": args.learning_rate,
                "seed": args.seed,
                "dataloader_generator_fixed": True,
            }, checkpoint_path)
        else:
            wait += 1

        if wait >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    pd.DataFrame(history).to_csv(
        result_dir / "training_history.csv",
        index=False,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate_test(
        model,
        loaders["test"],
        device,
        ema_scalers,
        tv_scalers,
    )

    result = {
        "condition": "pre-normalized speaker-wise EMA and TV with independent heads and fixed DataLoader generator",
        "test_speaker": test_speaker,
        "lambda_tv": args.lambda_tv,
        "best_epoch": best_epoch,
        "best_val_ema_normalized_mse": float(best_val_ema),
        "best_val_tv_normalized_mse": float(best_val_tv),
        "test_ema_normalized_mse": float(
            test_metrics["ema_normalized_mse"]
        ),
        "test_tv_normalized_mse": float(
            test_metrics["tv_normalized_mse"]
        ),
        "test_ema_mean_physical_rmse": float(
            test_metrics["ema_mean_physical_rmse"]
        ),
        "test_ema_global_physical_rmse": float(
            test_metrics["ema_global_physical_rmse"]
        ),
        "test_tv_mean_physical_rmse": float(
            test_metrics["tv_mean_physical_rmse"]
        ),
        "test_tv_global_physical_rmse": float(
            test_metrics["tv_global_physical_rmse"]
        ),
        "ema_physical_rmse_per_channel": {
            channel: float(value)
            for channel, value in zip(
                EMA_CHANNELS,
                test_metrics["ema_channel_rmse"],
            )
        },
        "tv_physical_rmse_per_channel": {
            channel: float(value)
            for channel, value in zip(
                TV_CHANNELS,
                test_metrics["tv_channel_rmse"],
            )
        },
    }

    with open(
        result_dir / "test_results.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    pd.DataFrame({
        "channel": EMA_CHANNELS,
        "physical_rmse": test_metrics["ema_channel_rmse"],
    }).to_csv(
        result_dir / "ema_channel_results.csv",
        index=False,
    )

    pd.DataFrame({
        "channel": TV_CHANNELS,
        "physical_rmse": test_metrics["tv_channel_rmse"],
    }).to_csv(
        result_dir / "tv_channel_results.csv",
        index=False,
    )

    pd.DataFrame(test_metrics["utterance_rows"]).to_csv(
        result_dir / "utterance_results.csv",
        index=False,
    )

    print(f"\nBest epoch                  : {best_epoch}")
    print(f"Best val EMA normalized MSE : {best_val_ema:.6f}")
    print(f"Best val TV normalized MSE  : {best_val_tv:.6f}")
    print(
        f"Test EMA normalized MSE     : "
        f"{test_metrics['ema_normalized_mse']:.6f}"
    )
    print(
        f"Test TV normalized MSE      : "
        f"{test_metrics['tv_normalized_mse']:.6f}"
    )
    print(
        f"Test EMA mean physical RMSE : "
        f"{test_metrics['ema_mean_physical_rmse']:.6f}"
    )
    print(
        f"Test TV mean physical RMSE  : "
        f"{test_metrics['tv_mean_physical_rmse']:.6f}"
    )

    return result


# =====================================================================
# main
# =====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speakers", nargs="+", default=SPEAKERS)
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--input-root",
        default="Data/oracle_speaker_normalized",
    )
    parser.add_argument(
        "--result-root",
        default="Results/speakerwise_ema_tv_multitask",
    )

    parser.add_argument("--lambda-tv", type=float, default=0.7)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max-epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--clip-norm", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"device        : {device}")
    print(f"input root    : {args.input_root}")
    print(f"optimizer     : Adam")
    print(f"learning rate : {args.learning_rate}")
    print(f"lambda TV     : {args.lambda_tv}")
    print(f"loader seed   : {args.seed} (dedicated generator)")

    results = []

    for speaker in args.speakers:
        set_seed(args.seed)
        results.append(
            run_fold(args, speaker, device)
        )

    summary_rows = []
    for result in results:
        summary_rows.append({
            "test_speaker": result["test_speaker"],
            "lambda_tv": result["lambda_tv"],
            "best_epoch": result["best_epoch"],
            "best_val_ema_normalized_mse": (
                result["best_val_ema_normalized_mse"]
            ),
            "best_val_tv_normalized_mse": (
                result["best_val_tv_normalized_mse"]
            ),
            "test_ema_normalized_mse": (
                result["test_ema_normalized_mse"]
            ),
            "test_tv_normalized_mse": (
                result["test_tv_normalized_mse"]
            ),
            "test_ema_mean_physical_rmse": (
                result["test_ema_mean_physical_rmse"]
            ),
            "test_ema_global_physical_rmse": (
                result["test_ema_global_physical_rmse"]
            ),
            "test_tv_mean_physical_rmse": (
                result["test_tv_mean_physical_rmse"]
            ),
            "test_tv_global_physical_rmse": (
                result["test_tv_global_physical_rmse"]
            ),
        })

    summary = pd.DataFrame(summary_rows)

    output_dir = (
        Path(args.result_root)
        / f"lambda_{args.lambda_tv:g}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(
        output_dir / "summary.csv",
        index=False,
    )

    print("\n===== Speaker-wise EMA+TV multitask summary =====")
    print(summary.to_string(index=False))

    print("\n===== Four-speaker mean =====")
    print(
        "EMA normalized MSE : "
        f"{summary['test_ema_normalized_mse'].mean():.6f}"
    )
    print(
        "TV normalized MSE  : "
        f"{summary['test_tv_normalized_mse'].mean():.6f}"
    )
    print(
        "EMA physical RMSE  : "
        f"{summary['test_ema_mean_physical_rmse'].mean():.6f}"
    )
    print(
        "EMA global RMSE    : "
        f"{summary['test_ema_global_physical_rmse'].mean():.6f}"
    )
    print(
        "TV physical RMSE   : "
        f"{summary['test_tv_mean_physical_rmse'].mean():.6f}"
    )


if __name__ == "__main__":
    main()
