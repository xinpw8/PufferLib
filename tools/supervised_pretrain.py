"""Supervised pre-training of ChessSeven policy on DeepMind's behavioral cloning data.

Usage:
    # Convert .bag to numpy (if not done):
    python tools/fen_converter.py data/searchless_chess/test/behavioral_cloning_data.bag data/searchless_chess/converted_test

    # Pre-train on converted data:
    python tools/supervised_pretrain.py --data-dir data/searchless_chess/converted_test --epochs 10

    # Pre-train on training data (streaming from .bag, no pre-conversion needed):
    python tools/supervised_pretrain.py --bag-file data/searchless_chess/train/behavioral_cloning_data.bag --epochs 3
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pufferlib.ocean.torch import ChessSeven
from pufferlib.pytorch import layer_init
from tools.fen_converter import BagReader, convert_record, OBS_SIZE


class NumpyChessDataset(Dataset):
    """Load pre-converted numpy chunks."""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        # Find all chunk files
        self.obs0_files = sorted(
            [f for f in os.listdir(data_dir) if f.startswith("obs_phase0_")]
        )
        # Load all into memory
        obs0_list, obs1_list, act_list = [], [], []
        for f in self.obs0_files:
            idx = f.replace("obs_phase0_", "").replace(".npy", "")
            obs0_list.append(np.load(os.path.join(data_dir, f)))
            obs1_list.append(np.load(os.path.join(data_dir, f"obs_phase1_{idx}.npy")))
            act_list.append(np.load(os.path.join(data_dir, f"actions_{idx}.npy")))

        self.obs_phase0 = np.concatenate(obs0_list)
        self.obs_phase1 = np.concatenate(obs1_list)
        self.actions = np.concatenate(act_list)
        print(f"Loaded {len(self.obs_phase0):,} samples from {len(self.obs0_files)} chunks")

    def __len__(self):
        return len(self.obs_phase0) * 2  # Each record yields 2 training examples

    def __getitem__(self, idx):
        # Alternate between phase0 and phase1 examples
        record_idx = idx // 2
        phase = idx % 2
        if phase == 0:
            obs = self.obs_phase0[record_idx]
            action = self.actions[record_idx, 0]
        else:
            obs = self.obs_phase1[record_idx]
            action = self.actions[record_idx, 1]
        return (
            torch.from_numpy(obs).float(),
            torch.tensor(action, dtype=torch.long),
        )


class StreamingBagDataset(IterableDataset):
    """Stream directly from .bag file, converting on the fly."""

    def __init__(self, bag_path: str, max_records: int = None, shuffle: bool = True):
        self.bag_path = bag_path
        self.max_records = max_records
        self.shuffle = shuffle

    def __iter__(self):
        reader = BagReader(self.bag_path)
        total = min(len(reader), self.max_records) if self.max_records else len(reader)

        indices = np.arange(total)
        if self.shuffle:
            np.random.shuffle(indices)

        for idx in indices:
            fen, move = reader.decode_behavioral_cloning(int(idx))
            result = convert_record(fen, move)
            if result is None:
                continue

            obs_p0, obs_p1, a0, a1 = result

            # Yield phase 0 example
            yield (
                torch.from_numpy(obs_p0).float(),
                torch.tensor(a0, dtype=torch.long),
            )
            # Yield phase 1 example
            yield (
                torch.from_numpy(obs_p1).float(),
                torch.tensor(a1, dtype=torch.long),
            )


def make_dummy_env():
    """Create a minimal env-like object for ChessSeven init."""
    from types import SimpleNamespace
    from gymnasium import spaces
    env = SimpleNamespace()
    env.single_action_space = spaces.Discrete(97)
    env.single_observation_space = spaces.Box(0, 255, shape=(OBS_SIZE,), dtype=np.uint8)
    return env


def train(args):
    device = torch.device(args.device)

    # Create model
    env = make_dummy_env()
    model = ChessSeven(
        env,
        square_dim=args.square_dim,
        proj_dim=args.proj_dim,
        hidden_size=args.hidden_size,
        embed_dim=args.embed_dim,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"Resumed from {args.resume}, starting at epoch {start_epoch}")

    # Create dataset
    if args.data_dir:
        dataset = NumpyChessDataset(args.data_dir)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
    elif args.bag_file:
        dataset = StreamingBagDataset(
            args.bag_file,
            max_records=args.max_records,
            shuffle=True,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=0,  # Can't multiprocess with mmap
            pin_memory=True,
            drop_last=True,
        )
    else:
        raise ValueError("Must specify either --data-dir or --bag-file")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1
    )

    # Training loop
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        phase0_correct = 0
        phase1_correct = 0
        phase0_total = 0
        phase1_total = 0
        epoch_start = time.time()

        for batch_idx, (obs, target_action) in enumerate(dataloader):
            obs = obs.to(device)
            target_action = target_action.to(device)

            # Forward pass
            hidden = model.encode_observations(obs)
            logits, _ = model.decode_actions(hidden)

            # Cross-entropy loss (no action masking during supervised training)
            loss = F.cross_entropy(logits, target_action)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            # Stats
            total_loss += loss.item() * obs.shape[0]
            pred = logits.argmax(dim=1)
            total_correct += (pred == target_action).sum().item()
            total_samples += obs.shape[0]

            # Track per-phase accuracy
            is_phase1 = obs[:, 852] > 0  # O_PICK_PHASE+1
            phase0_mask = ~is_phase1
            phase1_mask = is_phase1

            if phase0_mask.any():
                phase0_correct += (pred[phase0_mask] == target_action[phase0_mask]).sum().item()
                phase0_total += phase0_mask.sum().item()
            if phase1_mask.any():
                phase1_correct += (pred[phase1_mask] == target_action[phase1_mask]).sum().item()
                phase1_total += phase1_mask.sum().item()

            if (batch_idx + 1) % args.log_interval == 0:
                avg_loss = total_loss / total_samples
                accuracy = total_correct / total_samples * 100
                p0_acc = (phase0_correct / phase0_total * 100) if phase0_total > 0 else 0
                p1_acc = (phase1_correct / phase1_total * 100) if phase1_total > 0 else 0
                elapsed = time.time() - epoch_start
                sps = total_samples / elapsed
                print(f"  Epoch {epoch} | Batch {batch_idx+1} | "
                      f"Loss: {avg_loss:.4f} | Acc: {accuracy:.1f}% | "
                      f"P0: {p0_acc:.1f}% P1: {p1_acc:.1f}% | "
                      f"{sps:.0f} samples/sec")

        scheduler.step()
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples * 100
        p0_acc = (phase0_correct / phase0_total * 100) if phase0_total > 0 else 0
        p1_acc = (phase1_correct / phase1_total * 100) if phase1_total > 0 else 0
        elapsed = time.time() - epoch_start
        print(f"Epoch {epoch} done | Loss: {avg_loss:.4f} | Acc: {accuracy:.1f}% | "
              f"P0: {p0_acc:.1f}% P1: {p1_acc:.1f}% | "
              f"LR: {scheduler.get_last_lr()[0]:.6f} | {elapsed:.1f}s")

        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0 or epoch == args.epochs - 1:
            ckpt_path = os.path.join(args.checkpoint_dir, f"supervised_epoch_{epoch:04d}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'accuracy': accuracy,
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

    print("Training complete.")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Supervised pre-training for chess")
    parser.add_argument("--data-dir", type=str, help="Directory with pre-converted numpy data")
    parser.add_argument("--bag-file", type=str, help="Path to .bag file for streaming")
    parser.add_argument("--max-records", type=int, default=None, help="Max records from bag file")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/supervised_chess")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")

    # Model
    parser.add_argument("--square-dim", type=int, default=64)
    parser.add_argument("--proj-dim", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--embed-dim", type=int, default=32)

    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Logging
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=1)

    args = parser.parse_args()
    train(args)
