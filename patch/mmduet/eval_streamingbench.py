"""
Evaluate MMDuet predictions against StreamingBench Proactive_Output ground truth.

Usage:
    python eval_streamingbench.py \
        --pred_file outputs/mmduet/eval/streamingbench_proactive-pred.jsonl \
        --tolerance 5.0

StreamingBench ground truth (loaded via HF datasets):
    ground_truth_time_stamp: "00:01:01"
    ground_truth_output: "20"

MMDuet output (one JSON per line):
    {"question_id": "...", "model_response_list": [
        {"time": 61.0, "content": "20", "role": "assistant"}, ...
    ]}
"""

import argparse
import json
from collections import defaultdict


def ts_to_sec(ts):
    parts = ts.split(':')
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])


def load_streamingbench_gt():
    from datasets import load_dataset
    ds = load_dataset("mjuicem/StreamingBench", "Proactive_Output",
                      split="Proactive_Output", verification_mode='no_checks')
    gt = {}
    for row in ds:
        gt[row['question_id']] = {
            'gt_time': ts_to_sec(row['ground_truth_time_stamp']),
            'gt_output': row['ground_truth_output'],
            'query_time': ts_to_sec(row['time_stamp']),
            'temporal_clue_type': row.get('temporal_clue_type'),
        }
    return gt


def load_pred(path):
    preds = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            qid = d['question_id']
            fires = [r for r in d['model_response_list'] if r['role'] == 'assistant']
            preds[qid] = fires
    return preds


def evaluate(gt, preds, tolerance):
    results = []
    for qid, info in gt.items():
        fires = preds.get(qid, [])
        fire_times = [f['time'] for f in fires]
        gt_time = info['gt_time']

        hits = [t for t in fire_times if abs(t - gt_time) <= tolerance]
        detected = len(hits) > 0
        first_hit = min(hits) if hits else None
        latency = first_hit - gt_time if first_hit is not None else None
        false_alarms = [t for t in fire_times if abs(t - gt_time) > tolerance]

        results.append({
            'id': qid,
            'gt_time': gt_time,
            'detected': detected,
            'latency': latency,
            'n_fires': len(fires),
            'n_false_alarms': len(false_alarms),
        })

    if not results:
        return results, {}

    n = len(results)
    total_detected = sum(r['detected'] for r in results)
    total_fires = sum(r['n_fires'] for r in results)
    total_hits = sum(r['n_fires'] - r['n_false_alarms'] for r in results)
    total_false = sum(r['n_false_alarms'] for r in results)
    latencies = [r['latency'] for r in results if r['latency'] is not None]

    recall = total_detected / n
    precision = total_hits / total_fires if total_fires > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    avg_latency = sum(latencies) / len(latencies) if latencies else None

    # Firing accuracy by GT time bucket (decay curve)
    buckets = [(0, 60), (60, 120), (120, 180), (180, 300), (300, 600), (600, 900)]
    by_gt_time = {}
    for lo, hi in buckets:
        bucket_results = [r for r in results if lo <= r['gt_time'] < hi]
        if bucket_results:
            bd = sum(r['detected'] for r in bucket_results)
            by_gt_time[f'{lo}-{hi}s'] = {
                'n': len(bucket_results),
                'recall': bd / len(bucket_results),
            }

    summary = {
        'n_samples': n,
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'avg_latency_s': avg_latency,
        'total_fires': total_fires,
        'total_false_alarms': total_false,
        'tolerance': tolerance,
        'by_gt_time': by_gt_time,
    }
    return results, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_file', required=True)
    parser.add_argument('--tolerance', type=float, default=5.0)
    parser.add_argument('--out_file', default=None)
    args = parser.parse_args()

    gt = load_streamingbench_gt()
    preds = load_pred(args.pred_file)

    results, summary = evaluate(gt, preds, args.tolerance, )

    print(f"\n{'='*60}")
    print(f"StreamingBench Proactive Evaluation  (tol={summary['tolerance']}s)")
    print(f"{'='*60}")
    print(f"Samples:        {summary['n_samples']}")
    print(f"Recall:         {summary['recall']:.3f}")
    print(f"Precision:      {summary['precision']:.3f}")
    print(f"F1:             {summary['f1']:.3f}")
    if summary['avg_latency_s'] is not None:
        print(f"Avg latency:    {summary['avg_latency_s']:.2f}s")
    print(f"False alarms:   {summary['total_false_alarms']}")

    print(f"\nFiring accuracy by trigger time (decay curve):")
    for bucket, v in summary['by_gt_time'].items():
        print(f"  {bucket:10s}  n={v['n']:3d}  recall={v['recall']:.3f}")
    print(f"{'='*60}\n")

    if args.out_file:
        with open(args.out_file, 'w') as f:
            f.write(json.dumps(summary) + '\n')
            for r in results:
                f.write(json.dumps(r) + '\n')


if __name__ == '__main__':
    main()
