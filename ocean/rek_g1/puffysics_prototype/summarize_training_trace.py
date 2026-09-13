"""Summarize a node-level Nsight trace captured during real PPO training."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3


def summarize(path):
    path = Path(path).resolve(strict=True)
    connection = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)
    connection.row_factory = sqlite3.Row
    kernels = list(connection.execute('''
        SELECT k.start, k.end, k.deviceId, s.value AS name,
               k.registersPerThread, k.gridX, k.gridY, k.gridZ,
               k.blockX, k.blockY, k.blockZ, k.localMemoryPerThread
        FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s
          ON s.id=k.demangledName ORDER BY k.deviceId, k.start, k.end
    '''))
    if not kernels:
        raise ValueError('No kernel nodes; capture with --cuda-graph-trace=node')
    totals = defaultdict(lambda: {'calls': 0, 'sum_kernel_ms': 0., 'launch_configurations': set()})
    unions = defaultdict(float)
    intervals = {}
    for row in kernels:
        device = row['deviceId']
        start, end = row['start'], row['end']
        current = intervals.get(device)
        if current is None:
            intervals[device] = (start, end)
        elif start > current[1]:
            unions[device] += current[1] - current[0]
            intervals[device] = (start, end)
        else:
            intervals[device] = (current[0], max(current[1], end))
        value = totals[row['name']]
        value['calls'] += 1
        value['sum_kernel_ms'] += (end - start) / 1e6
        value['launch_configurations'].add(tuple(row[name] for name in (
            'registersPerThread', 'gridX', 'gridY', 'gridZ', 'blockX', 'blockY', 'blockZ', 'localMemoryPerThread')))
    for device, (start, end) in intervals.items():
        unions[device] += end-start
    total_ms = sum(value['sum_kernel_ms'] for value in totals.values())
    ranked = []
    for name, value in sorted(totals.items(), key=lambda item: -item[1]['sum_kernel_ms']):
        ranked.append({'name': name, 'calls': value['calls'],
                       'sum_kernel_ms': value['sum_kernel_ms'],
                       'percent_sum_kernel_time': 100 * value['sum_kernel_ms'] / total_ms,
                       'launch_configurations': [dict(zip(
                           ('registers_per_thread', 'grid_x', 'grid_y', 'grid_z',
                            'block_x', 'block_y', 'block_z', 'local_bytes_per_thread'), values))
                           for values in sorted(value['launch_configurations'])]})
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    copies = []
    if 'CUPTI_ACTIVITY_KIND_MEMCPY' in tables:
        copies = [dict(row) for row in connection.execute('''
            SELECT copyKind, COUNT(*) AS calls, SUM(bytes) AS bytes,
                   SUM(end-start)/1e6 AS sum_copy_ms
            FROM CUPTI_ACTIVITY_KIND_MEMCPY GROUP BY copyKind ORDER BY copyKind
        ''')]
    api = [dict(row) for row in connection.execute('''
        SELECT s.value AS name, COUNT(*) AS calls, SUM(r.end-r.start)/1e6 AS sum_api_ms
        FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId
        GROUP BY r.nameId ORDER BY sum_api_ms DESC LIMIT 20
    ''')]
    connection.close()
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024*1024), b''):
            digest.update(block)
    return {
        'schema': 'rek.native_training_nsight_kernel_summary.v1',
        'sqlite': {'path': str(path), 'sha256': digest.hexdigest()},
        'kernel_calls': len(kernels), 'sum_kernel_ms': total_ms,
        'kernel_busy_union_ms_by_device': {str(key): value/1e6 for key, value in unions.items()},
        'kernel_timeline_span_ms': (max(row['end'] for row in kernels)-min(row['start'] for row in kernels))/1e6,
        'kernels': ranked, 'memcpy_by_cuda_copy_kind': copies, 'largest_runtime_api_totals': api,
        'interpretation': [
            'Captured range includes actual training and update-boundary diagnostics.',
            'Kernel sums count overlapping kernels separately; union time counts overlap once per device.',
            'CUDA API duration overlaps GPU execution and must not be added to GPU duration.',
            'Node-level tracing perturbs throughput; use unprofiled training runs for the SPS comparison.',
        ],
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sqlite', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.sqlite)
    with args.out.open('x', encoding='utf-8') as destination:
        json.dump(report, destination, indent=2, sort_keys=True, allow_nan=False)
        destination.write('\n')
    print(json.dumps({key: report[key] for key in ('kernel_calls', 'sum_kernel_ms', 'kernel_busy_union_ms_by_device')}))
    print(json.dumps(report['kernels'][:8], indent=2))
