"""
Evaluate MMDuet predictions against ROMA alert ground truth.

Usage:
    python eval_roma.py \
        --gt_file /path/to/alert.jsonl \
        --pred_file /path/to/mmduet_output.jsonl \
        --tolerance 1.0

MMDuet output format (one JSON per line):
    {"question_id": "908", "model_response_list": [
        {"time": 0.0, "content": "...", "role": "user"},
        {"time": 4.5, "content": "The person fell", "role": "assistant"},
    ], "video_duration": 12.0}

ROMA ground truth (one JSON per line):
    {"id": "908", "ans": [{"text": "alert", "time": 4.0}, {"time": 5.0}, {"time": 6.0}], ...}

The gt window is [min(ans.time), max(ans.time)].
A model fire (role=assistant) is a HIT if it falls within [window_start - tol, window_end + tol].
"""

import argparse
import json
from collections import defaultdict


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_gt_windows(gt_data):
    windows = {}
    for sample in gt_data:
        times = [a['time'] for a in sample['ans']]
        windows[str(sample['id'])] = (min(times), max(times))
    return windows


def extract_pred_fires(pred_data):
    fires = {}
    for sample in pred_data:
        qid = str(sample['question_id'])
        fire_times = [
            r['time'] for r in sample['model_response_list']
            if r['role'] == 'assistant'
        ]
        fires[qid] = fire_times
    return fires


def evaluate(gt_windows, pred_fires, tolerance):
    results = []
    for qid, (gt_start, gt_end) in gt_windows.items():
        fires = pred_fires.get(qid, [])
        hits = [t for t in fires if gt_start - tolerance <= t <= gt_end + tolerance]
        false_alarms = [t for t in fires if t < gt_start - tolerance or t > gt_end + tolerance]

        detected = len(hits) > 0
        first_hit = min(hits) if hits else None
        latency = first_hit - gt_start if first_hit is not None else None

        results.append({
            'id': qid,
            'gt_window': (gt_start, gt_end),
            'n_fires': len(fires),
            'n_hits': len(hits),
            'n_false_alarms': len(false_alarms),
            'detected': detected,
            'latency': latency,
        })

    n = len(results)
    if n == 0:
        return results, {}

    total_fires = sum(r['n_fires'] for r in results)
    total_hits = sum(r['n_hits'] for r in results)
    total_false_alarms = sum(r['n_false_alarms'] for r in results)
    total_detected = sum(r['detected'] for r in results)
    latencies = [r['latency'] for r in results if r['latency'] is not None]

    recall = total_detected / n
    precision = total_hits / total_fires if total_fires > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    avg_latency = sum(latencies) / len(latencies) if latencies else None

    missed_ids = [r['id'] for r in results if not r['detected']]

    summary = {
        'n_samples': n,
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'avg_latency_s': avg_latency,
        'total_fires': total_fires,
        'total_hits': total_hits,
        'total_false_alarms': total_false_alarms,
        'n_missed': len(missed_ids),
        'tolerance': tolerance,
    }

    return results, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_file', required=True, help='Path to ROMA alert.jsonl')
    parser.add_argument('--pred_file', required=True, help='Path to MMDuet output JSONL')
    parser.add_argument('--tolerance', type=float, default=1.0,
                        help='Seconds of slack around the GT window edges (default: 1.0)')
    parser.add_argument('--out_file', default=None, help='Write per-sample results to this JSONL')
    args = parser.parse_args()

    gt_data = load_jsonl(args.gt_file)
    pred_data = load_jsonl(args.pred_file)

    gt_windows = extract_gt_windows(gt_data)
    pred_fires = extract_pred_fires(pred_data)

    missing = set(gt_windows.keys()) - set(pred_fires.keys())
    if missing:
        print(f'WARNING: {len(missing)} GT samples have no predictions (will count as missed)')

    results, summary = evaluate(gt_windows, pred_fires, args.tolerance)

    print(f"\n{'='*50}")
    print(f"ROMA Alert Evaluation  (tol={summary['tolerance']}s)")
    print(f"{'='*50}")
    print(f"Samples:        {summary['n_samples']}")
    print(f"Recall:         {summary['recall']:.3f}  ({summary['n_samples'] - summary['n_missed']}/{summary['n_samples']} events detected)")
    print(f"Precision:      {summary['precision']:.3f}  ({summary['total_hits']}/{summary['total_fires']} fires in-window)")
    print(f"F1:             {summary['f1']:.3f}")
    if summary['avg_latency_s'] is not None:
        print(f"Avg latency:    {summary['avg_latency_s']:.2f}s  (first hit - window start)")
    print(f"False alarms:   {summary['total_false_alarms']}")
    print(f"{'='*50}\n")

    if args.out_file:
        with open(args.out_file, 'w') as f:
            f.write(json.dumps(summary) + '\n')
            for r in results:
                f.write(json.dumps(r) + '\n')
        print(f"Per-sample results written to {args.out_file}")


if __name__ == '__main__':
    main()
