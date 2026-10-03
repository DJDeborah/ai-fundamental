"""Single-GPU non-preemptive queue replay; times are seconds, not speedups."""
import argparse
import json
from pathlib import Path


def replay(jobs, policy):
    pending = [dict(j) for j in jobs]
    if len({j['id'] for j in pending}) != len(pending):
        raise ValueError('Job IDs must be unique')
    for j in pending:
        if j['duration_s'] <= 0 or j.get('arrival_s', 0) < 0:
            raise ValueError('Invalid duration/arrival')
        if policy == 'predicted_spt' and j.get('predicted_s', 0) <= 0:
            raise ValueError('predicted_s must be positive')
    clock, rows = 0.0, []
    while pending:
        ready = [j for j in pending if j.get('arrival_s', 0) <= clock]
        if not ready:
            clock = min(j.get('arrival_s', 0) for j in pending)
            continue
        if policy == 'fcfs':
            key = lambda j: (j.get('arrival_s', 0), jobs.index(next(x for x in jobs if x['id'] == j['id'])))
        else:
            key = lambda j: (j['duration_s'] if policy == 'oracle_spt' else j['predicted_s'], j['id'])
        job = min(ready, key=key)
        arrival = job.get('arrival_s', 0)
        finish = clock + job['duration_s']
        rows.append({'id': job['id'], 'start_s': clock, 'finish_s': finish,
                     'wait_s': clock - arrival, 'flow_s': finish - arrival})
        clock = finish
        pending.remove(job)
    n = len(rows)
    return {'policy': policy, 'jobs': rows,
            'mean_wait_s': sum(j['wait_s'] for j in rows) / n if n else 0,
            'mean_flow_s': sum(j['flow_s'] for j in rows) / n if n else 0,
            'makespan_s': clock}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--jobs', type=Path)
    p.add_argument('--out', type=Path, default=Path('runs/queue_replay.json'))
    args = p.parse_args()
    jobs = json.loads(args.jobs.read_text()) if args.jobs else [
        {'id': 'A', 'duration_s': 3600, 'predicted_s': 3500},
        {'id': 'B', 'duration_s': 600, 'predicted_s': 650},
        {'id': 'C', 'duration_s': 300, 'predicted_s': 320},
    ]
    results = [replay(jobs, p) for p in ['fcfs', 'oracle_spt', 'predicted_spt']]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
