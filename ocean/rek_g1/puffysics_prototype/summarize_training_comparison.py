"""Validate matched real-training inputs and aggregate measured timings."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def record(path):
    path = Path(path)
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def comparable(report):
    return {
        'host': report['host'], 'gpu': report['runtime']['gpu'],
        'duel': report['effective_duel_config'],
        'learner': report['effective_native_config'],
        'native_extension_sha256': report['native_extension']['sha256'],
        'initial_weights_sha256': report['checkpoints']['initial']['sha256'],
        'controller_sha256': {key: value['sha256'] for key, value in report['controller_models'].items()},
        'shared_inputs_sha256': {key: value['sha256'] for key, value in report['inputs'].items()
                                if key not in ('model_export', 'puffysics_library')},
        'benchmark_sha256': report['sources']['benchmark']['sha256'],
    }


def aggregate(paths):
    reports = [read(path) for path in paths]
    if not reports or any(report['status'] != 'passed' for report in reports):
        raise ValueError('Throughput aggregation requires successful runs')
    transitions = sum(report['measured']['learner_transitions'] for report in reports)
    seconds = sum(report['measured']['timing']['wall_seconds'] for report in reports)
    phases = {name: sum(report['measured']['timing'][name] for report in reports)
              for name in ('rollout_cuda_stream_ms', 'ppo_update_cuda_stream_ms', 'cuda_stream_envelope_ms')}
    if not all(report['checkpoints']['measured_weights_changed'] and
               report['measured']['verified_updates'] == report['measured']['completed_updates']
               for report in reports):
        raise ValueError('Run lacks verified updates or changed policy weights')
    return {
        'runs': [record(path) for path in paths],
        'per_run_training_sps': [report['measured']['training_learner_transitions_per_second'] for report in reports],
        'training_sps': transitions / seconds, 'learner_transitions': transitions,
        'training_wall_seconds': seconds, 'all_weights_changed': True,
        'rollout_percent_cuda_stream': 100*phases['rollout_cuda_stream_ms']/phases['cuda_stream_envelope_ms'],
        'ppo_percent_cuda_stream': 100*phases['ppo_update_cuda_stream_ms']/phases['cuda_stream_envelope_ms'],
        'arenas': reports[0]['training']['arenas'], 'horizon': reports[0]['training']['horizon'],
    }


def summarize(root):
    root = Path(root)
    paths = {backend: sorted(root.glob(backend+'-final-h16-r*/report.json')) for backend in ('mujoco', 'puffysics')}
    if any(len(selected) != 3 for selected in paths.values()):
        raise ValueError('Expected three completed trials per primary backend')
    reference = comparable(read(paths['mujoco'][0]))
    for selected in paths.values():
        for path in selected:
            if comparable(read(path)) != reference:
                raise ValueError('Shared benchmark inputs differ: '+str(path))
    paired = {backend: aggregate(selected) for backend, selected in paths.items()}
    production_paths = sorted((root/'current-production').glob('report-r*.json'))
    production = [read(path) for path in production_paths]
    failure_path = root/'puffysics-sustained-h256-v1/report.json'
    failure = read(failure_path)
    stats = failure.get('physics_failure_evidence', {})
    evidence = {key: value for key, value in stats.items() if key != 'per_arena'}
    evidence['arenas_with_any_solver_or_nonfinite_failure'] = sum(bool(row[1] or row[2]) for row in stats.get('per_arena', []))
    result = {
        'schema': 'rek.matched_native_ppo_training_summary.v1', 'host': reference['host'],
        'paired_inputs_equal': True, 'paired_inputs': reference,
        'matched_512_arena_h16': paired,
        'mujoco_over_puffysics_training_sps_ratio': paired['mujoco']['training_sps']/paired['puffysics']['training_sps'],
        'current_production_h256': {
            'runs': [record(path) for path in production_paths],
            'per_run_training_sps': [r['training']['agent_steps_per_second'] for r in production],
            'training_sps': sum(r['training']['agent_steps'] for r in production)/sum(r['training']['wall_seconds'] for r in production),
            'timing_scope': 'existing production loop including periodic reporting; setup and final checkpoint excluded',
            'difference_from_matched_pair': 'horizon256 and conditional reset forward enabled; separate baseline',
        },
        'sustained_puffysics': {'report': record(failure_path), 'status': failure['status'],
                               'valid_training_sps': failure['measured']['training_learner_transitions_per_second'],
                               'failure_stage': failure['errors'][0]['stage'], 'physics': evidence,
                               'native_counters': failure.get('native_counters_at_failure')},
        'limits': ['Matched short runs span 0.32 simulated seconds of warmup plus 0.64 measured seconds per arena.',
                   'Physics trajectories differ, so this measures the same task implementation with different solvers, not parity.',
                   'No learning quality or superhuman conclusion follows from short training throughput.',
                   'Current production and short matched comparisons have different horizons; do not call their ratio a physics-only effect.'],
    }
    for backend in ('mujoco', 'puffysics'):
        path = root/(backend+'-4096-h16-v1/report.json')
        if path.exists():
            run = read(path)
            result.setdefault('scaling_4096', {})[backend] = aggregate([path]) if run['status']=='passed' else {
                'status': run['status'], 'report': record(path),
                'failure_stages': [error['stage'] for error in run['errors']],
                'physics': {key: value for key, value in run.get('physics_failure_evidence', {}).items() if key != 'per_arena'}}
        path = root/(backend+'-nsys-summary.json')
        if path.exists():
            trace = read(path)
            result.setdefault('nsight', {})[backend] = {
                'report': record(path), 'top_kernels': trace['kernels'][:5],
                'kernel_calls': trace['kernel_calls'], 'sum_kernel_ms': trace['sum_kernel_ms'],
                'adapter_percent_sum_kernel_time': sum(row['percent_sum_kernel_time'] for row in trace['kernels'] if row['name'].startswith('rps_')),
            }
        path = root/(backend+'-events-h16-v1/report.json')
        if path.exists():
            run = read(path)
            totals = defaultdict(float)
            for row in run['measured']['records']:
                for name, part in row.get('rollout_subphases', {}).items():
                    totals[name] += part['cuda_stream_elapsed_ms']
            envelope = run['measured']['timing']['cuda_stream_envelope_ms']
            result.setdefault('instrumented_training_breakdown', {})[backend] = {
                'report': record(path), 'status': run['status'],
                'percent_total_cuda_stream': {name:100*ms/envelope for name,ms in totals.items()},
                'ppo_percent_total_cuda_stream': 100*run['measured']['timing']['ppo_update_cuda_stream_ms']/envelope,
            }
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.root)
    with args.out.open('x', encoding='utf-8') as destination:
        json.dump(report, destination, indent=2, sort_keys=True, allow_nan=False)
        destination.write('\n')
    print(json.dumps({key: report[key] for key in ('paired_inputs_equal','mujoco_over_puffysics_training_sps_ratio')},indent=2))
