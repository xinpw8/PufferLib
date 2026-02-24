#!/usr/bin/env python3
"""Extract unique FENs from DeepMind's searchless_chess .bag file.

Parallelized across all available CPU cores. Each worker processes a chunk
of records, writes unique FENs to a temp file, then a final pass deduplicates.

Usage:
    python tools/extract_deepmind_fens.py <bag_file> <output_file> [--max-records N] [--workers N]
"""

import argparse
import multiprocessing as mp
import os
import struct
import sys
import tempfile
import time


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

    seen = set()
    errors = 0
    count = 0

    for i in range(start_idx, end_idx):
        rec_start = 0 if i == 0 else get_limit(i - 1)
        rec_end = get_limit(i)
        data = mm[rec_start:rec_end]

        try:
            fen_len, pos = _read_varint(data, 0)
            fen_bytes = data[pos:pos + fen_len]
            # Find the 4th space to strip halfmove clock + fullmove number
            spaces = 0
            cut = len(fen_bytes)
            for j in range(len(fen_bytes)):
                if fen_bytes[j] == 0x20:  # space
                    spaces += 1
                    if spaces == 4:
                        cut = j
                        break
            seen.add(fen_bytes[:cut])
        except Exception:
            errors += 1

        count += 1
        if count % 5_000_000 == 0:
            print(f"  Worker {worker_id}: {count:,}/{end_idx - start_idx:,} records, {len(seen):,} unique", flush=True)

    mm.close()
    fh.close()

    # Write to temp file to avoid pickling large sets
    tmp = tempfile.NamedTemporaryFile(mode='wb', prefix=f'fens_w{worker_id}_', suffix='.bin',
                                       delete=False)
    for fen in seen:
        tmp.write(fen)
        tmp.write(b'\n')
    tmp.close()

    print(f"  Worker {worker_id}: done — {len(seen):,} unique from {count:,} records ({errors} errors)", flush=True)
    return tmp.name, len(seen), errors


def main():
    parser = argparse.ArgumentParser(description='Extract unique FENs from DeepMind .bag file')
    parser.add_argument('bag_file', help='Path to the .bag file')
    parser.add_argument('output_file', help='Path to write unique FENs')
    parser.add_argument('--max-records', type=int, default=0,
                        help='Max records to process (0 = all)')
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

    # Merge: read all temp files, deduplicate
    seen = set()
    total_errors = 0
    for tmp_path, count, errors in results:
        total_errors += errors
        with open(tmp_path, 'rb') as f:
            for line in f:
                seen.add(line.rstrip(b'\n'))
        os.unlink(tmp_path)

    t2 = time.time()
    print(f"Merge done in {t2 - t1:.1f}s. {len(seen):,} unique FENs, {total_errors} total errors", flush=True)

    # Write final output
    with open(args.output_file, 'wb') as f:
        for fen in seen:
            f.write(fen)
            f.write(b'\n')

    t3 = time.time()
    print(f"Wrote {len(seen):,} FENs to {args.output_file} in {t3 - t2:.1f}s (total: {t3 - t0:.1f}s)", flush=True)


if __name__ == '__main__':
    main()
