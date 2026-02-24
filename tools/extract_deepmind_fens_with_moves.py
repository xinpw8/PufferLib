#!/usr/bin/env python3
"""Extract (FEN, UCI_move) pairs from DeepMind's searchless_chess .bag file.

Parallelized across all available CPU cores. Each worker processes a chunk
of records, writes (FEN, move) pairs to a temp file, then a final pass
uniformly subsamples to the desired count.

Usage:
    python tools/extract_deepmind_fens_with_moves.py [bag_file] [output_file] [--max-records N] [--sample N] [--workers N]
"""

import argparse
import multiprocessing as mp
import os
import struct
import sys
import tempfile
import time
import random


def _read_varint(buf, pos):
    result = 0
    shift = 0
    while pos < len(buf):
        b = buf[pos]
        result |= (b & 0x7f) << shift
        pos += 1
        if (b & 0x80) == 0:
            return result, pos
        shift += 7
    return result, pos


def _worker(args):
    """Process a chunk of records from the .bag file."""
    bag_path, start_idx, end_idx, worker_id, index_start = args
    import mmap

    fh = open(bag_path, 'rb')
    mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)

    def get_limit(idx):
        offset = index_start + idx * 8
        return struct.unpack('<Q', mm[offset:offset + 8])[0]

    pairs = []
    errors = 0
    count = 0

    for i in range(start_idx, end_idx):
        rec_start = 0 if i == 0 else get_limit(i - 1)
        rec_end = get_limit(i)
        data = mm[rec_start:rec_end]

        try:
            fen_len, pos = _read_varint(data, 0)
            fen_bytes = data[pos:pos + fen_len]
            move_bytes = data[pos + fen_len:]

            # Strip halfmove clock + fullmove number (keep first 4 fields)
            spaces = 0
            cut = len(fen_bytes)
            for j in range(len(fen_bytes)):
                if fen_bytes[j] == 0x20:  # space
                    spaces += 1
                    if spaces == 4:
                        cut = j
                        break
            fen = fen_bytes[:cut].decode('ascii', errors='replace')

            # Move is the remaining bytes after the FEN
            move = move_bytes.decode('ascii', errors='replace').strip()
            if len(move) >= 4:  # Valid UCI move is at least 4 chars (e.g. e2e4)
                pairs.append((fen, move))
        except Exception:
            errors += 1

        count += 1
        if count % 5_000_000 == 0:
            print(f"  Worker {worker_id}: {count:,}/{end_idx - start_idx:,} records, {len(pairs):,} pairs", flush=True)

    mm.close()
    fh.close()

    # Write to temp file to avoid pickling large lists
    tmp = tempfile.NamedTemporaryFile(mode='w', prefix=f'fens_moves_w{worker_id}_', suffix='.txt',
                                       delete=False)
    for fen, move in pairs:
        tmp.write(f"{fen}\t{move}\n")
    tmp.close()

    print(f"  Worker {worker_id}: done — {len(pairs):,} pairs from {count:,} records ({errors} errors)", flush=True)
    return tmp.name, len(pairs), errors


def main():
    parser = argparse.ArgumentParser(description='Extract (FEN, move) pairs from DeepMind .bag file')
    parser.add_argument('bag_file', nargs='?',
                        default='data/searchless_chess/train/behavioral_cloning_data.bag',
                        help='Path to the .bag file')
    parser.add_argument('output_file', nargs='?',
                        default='pufferlib/ocean/chess/fens_moves_deepmind.txt',
                        help='Path to write FEN+move pairs')
    parser.add_argument('--max-records', type=int, default=0,
                        help='Max records to process (0 = all)')
    parser.add_argument('--sample', type=int, default=2_000_000,
                        help='Number of pairs to uniformly subsample (0 = all)')
    parser.add_argument('--workers', type=int, default=0,
                        help='Number of workers (0 = all cores)')
    args = parser.parse_args()

    t0 = time.time()
    num_workers = args.workers if args.workers > 0 else mp.cpu_count()

    # Read index metadata
    with open(args.bag_file, 'rb') as f:
        f.seek(-8, 2)
        file_size = f.tell() + 8
        index_start = struct.unpack('<Q', f.read(8))[0]

    index_size = file_size - 8 - index_start
    num_records = index_size // 8

    limit = args.max_records if args.max_records > 0 else num_records
    limit = min(limit, num_records)

    print(f"Records: {num_records:,}, processing: {limit:,}, workers: {num_workers}", flush=True)

    # Split work across workers
    chunk_size = (limit + num_workers - 1) // num_workers
    chunks = []
    for w in range(num_workers):
        s = w * chunk_size
        e = min(s + chunk_size, limit)
        if s >= limit:
            break
        chunks.append((args.bag_file, s, e, w, index_start))

    print(f"Launching {len(chunks)} workers...", flush=True)

    with mp.Pool(len(chunks)) as pool:
        results = pool.map(_worker, chunks)

    t1 = time.time()
    print(f"Workers done in {t1 - t0:.1f}s. Merging...", flush=True)

    # Merge all temp files
    all_lines = []
    total_errors = 0
    for tmp_path, count, errors in results:
        total_errors += errors
        with open(tmp_path, 'r') as f:
            all_lines.extend(f.readlines())
        os.unlink(tmp_path)

    t2 = time.time()
    print(f"Merge done in {t2 - t1:.1f}s. {len(all_lines):,} total pairs, {total_errors} errors", flush=True)

    # Subsample if requested
    if args.sample > 0 and len(all_lines) > args.sample:
        print(f"Subsampling {args.sample:,} from {len(all_lines):,}...", flush=True)
        random.seed(42)
        all_lines = random.sample(all_lines, args.sample)

    # Write final output
    with open(args.output_file, 'w') as f:
        for line in all_lines:
            f.write(line if line.endswith('\n') else line + '\n')

    t3 = time.time()
    print(f"Wrote {len(all_lines):,} (FEN, move) pairs to {args.output_file} in {t3 - t2:.1f}s (total: {t3 - t0:.1f}s)", flush=True)


if __name__ == '__main__':
    main()
