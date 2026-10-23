#!/usr/bin/env python3
"""Analyze WeNet training logs (TRAIN batch metrics and epoch CV metrics).

Usage:
    python check_loss.py train.txt
    python check_loss.py train.txt --output-dir results

Training rows are periodic batch snapshots, NOT epoch-averaged train loss.
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
TRAIN_RE = re.compile(r"\bTRAIN\s*\|.*?\bBatch\s+(?P<epoch>\d+)\s*/\s*(?P<batch>\d+)\b")
CV_RE = re.compile(r"\bEpoch\s+(?P<epoch>\d+)\s+Step\s+(?P<step>\d+)\s+CV info\b")
EPOCH_START_RE = re.compile(r"\bEpoch\s+(?P<epoch>\d+)\s+Step\s+(?P<step>\d+)\s+TRAIN info\b")
METRIC_RE = re.compile(rf"\b(?P<key>loss_att|loss_ctc|loss|th_accuracy|cv_loss|grad_norm|lr|acc)\s+(?P<value>{NUMBER})\b")
TIME_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")


def metrics(line):
    return {m.group("key"): float(m.group("value")) for m in METRIC_RE.finditer(line)}


def parse_log(path):
    train, cv = [], []
    epoch_steps = {}
    with path.open(encoding="utf-8", errors="replace") as file:
        for line in file:
            start = EPOCH_START_RE.search(line)
            if start:
                epoch_steps[int(start["epoch"])] = int(start["step"])
            match = TRAIN_RE.search(line)
            if match:
                values = metrics(line)
                if "loss" not in values:
                    continue
                epoch = int(match["epoch"])
                batch = int(match["batch"])
                time = TIME_RE.search(line)
                train.append({"epoch": epoch, "batch": batch,
                              "step": epoch_steps.get(epoch, 0) + batch,
                              "timestamp": time.group(1) if time else None,
                              **values})
                continue
            match = CV_RE.search(line)
            if match:
                values = metrics(line)
                if "cv_loss" not in values:
                    continue
                time = TIME_RE.search(line)
                cv.append({"epoch": int(match["epoch"]), "step": int(match["step"]),
                           "timestamp": time.group(1) if time else None, **values})
    train_df = pd.DataFrame(train)
    cv_df = pd.DataFrame(cv)
    if not train_df.empty:
        train_df = train_df.sort_values(["epoch", "step"]).drop_duplicates(["epoch", "batch"], keep="last").reset_index(drop=True)
    if not cv_df.empty:
        cv_df = cv_df.sort_values("epoch").drop_duplicates("epoch", keep="last").reset_index(drop=True)
    return train_df, cv_df


def summarize(train_df, cv_df):
    if train_df.empty and cv_df.empty:
        raise ValueError("Không tìm thấy WeNet TRAIN hoặc CV entries trong log.")
    if train_df.empty:
        summary = pd.DataFrame(columns=["epoch"])
    else:
        # Mean of periodic LOGGED snapshots only, not a true full-epoch mean.
        summary = train_df.groupby("epoch", as_index=False).agg(
            train_loss_logged_mean=("loss", "mean"),
            train_loss_last=("loss", "last"),
            train_loss_att_logged_mean=("loss_att", "mean"),
            train_loss_ctc_logged_mean=("loss_ctc", "mean"),
            train_accuracy_logged_mean=("th_accuracy", "mean"),
            lr_last=("lr", "last"),
            train_logged_points=("loss", "count"),
        )
    if not cv_df.empty:
        cv_summary = cv_df[["epoch", "cv_loss", "acc", "lr", "step"]].rename(
            columns={"acc": "cv_accuracy", "lr": "cv_lr", "step": "cv_step"})
        summary = summary.merge(cv_summary, on="epoch", how="outer")
    return summary.sort_values("epoch").reset_index(drop=True)


def make_plots(train_df, cv_df, output_dir):
    def save(name):
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / name, dpi=160)
        plt.close()

    fig, ax = plt.subplots(figsize=(10, 5))
    if not train_df.empty:
        grouped = train_df.groupby("epoch", as_index=False)["loss"].mean()
        ax.plot(grouped.epoch, grouped.loss, "o-", label="Train loss (mean of logged snapshots)")
    if not cv_df.empty:
        ax.plot(cv_df.epoch, cv_df.cv_loss, "o-", label="CV loss")
    ax.set(xlabel="Epoch (0-based)", ylabel="Loss", title="WeNet: train snapshots vs CV loss")
    save("loss_by_epoch.png")

    if not train_df.empty:
        fig, ax = plt.subplots(figsize=(11, 5))
        for key, label in [("loss", "Total"), ("loss_att", "Attention"), ("loss_ctc", "CTC")]:
            if key in train_df:
                ax.plot(train_df.step, train_df[key], marker=".", label=label)
        ax.set(xlabel="Approx. global step (epoch start + batch counter)", ylabel="Loss",
               title="WeNet: logged TRAIN batch losses")
        save("training_loss.png")

    if not cv_df.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(cv_df.epoch, cv_df.cv_loss, "o-", label="CV loss")
        ax.set(xlabel="Epoch (0-based)", ylabel="Loss", title="WeNet: CV loss")
        save("validation_loss.png")

        if "acc" in cv_df:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(cv_df.epoch, cv_df.acc, "o-", label="CV accuracy")
            ax.set(xlabel="Epoch (0-based)", ylabel="Accuracy (fraction)", title="WeNet: CV accuracy")
            save("validation_accuracy.png")


def main():
    parser = argparse.ArgumentParser(description="Check WeNet-only TRAIN / CV losses")
    parser.add_argument("log_file", type=Path, help="WeNet train.txt")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    if not args.log_file.is_file():
        parser.error(f"Log không tồn tại: {args.log_file}")
    output = args.output_dir or args.log_file.parent / "loss_analysis"
    output.mkdir(parents=True, exist_ok=True)
    train, cv = parse_log(args.log_file)
    summary = summarize(train, cv)
    train.to_csv(output / "train_points.csv", index=False)
    cv.to_csv(output / "cv_epochs.csv", index=False)
    summary.to_csv(output / "loss_summary.csv", index=False)
    make_plots(train, cv, output)
    print(f"TRAIN snapshots: {len(train)} | CV epochs: {len(cv)}")
    print("\n" + summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    if not cv.empty:
        best = cv.loc[cv.cv_loss.idxmin()]
        print(f"\nBest CV: epoch {int(best.epoch)} (epoch thứ {int(best.epoch)+1}), loss={best.cv_loss:.6f}, accuracy={best.get('acc', float('nan')):.4%}")
        first, last = cv.iloc[0].cv_loss, cv.iloc[-1].cv_loss
        if first:
            print(f"CV loss change first -> last: {(last / first - 1)*100:+.2f}%")
    print("\nChú ý: train loss là snapshot từng batch được log; trung bình chỉ trên các snapshot, không phải epoch mean thực.")
    print(f"Saved results: {output}")


if __name__ == "__main__":
    main()
