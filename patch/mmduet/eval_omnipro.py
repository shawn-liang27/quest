"""
Evaluate MMDuet predictions against OmniPro ground truth.

Usage:
    python eval_omnipro.py \
        --pred_file outputs/mmduet/eval/omnipro-pred.jsonl \
        --tolerance 5.0 \
        --visual_only

OmniPro ground_truth field (JSON string per sample):
    [{"trigger_time_sec": 164, "response": "...", ...}, ...]

MMDuet output (one JSON per line):
    {"question_id": "...", "model_response_list": [
        {"time": 4.5, "content": "...", "role": "assistant"}, ...
    ]}
"""

import argparse
import json
from collections import defaultdict


def load_omnipro_gt():
    from datasets import load_dataset
    ds = load_dataset("RuixiangZhao/OmniPro", split="test")
    gt = {}
    for row in ds:
        triggers = json.loads(row['ground_truth'])
        gt[row['id']] = {
            'triggers': [{'time': t['trigger_time_sec'], 'response': t['response']} for t in triggers],
            'task': row['task'],
            'audio_dependency': row.get('audio_dependency'),
            'duration': row['duration'],
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


def evaluate(gt, preds, tolerance, visual_only=False, tasks=None):
    results = []
    for qid, info in gt.items():
        if visual_only and info['audio_dependency'] != 'none':
            continue
        if tasks and info['task'] not in tasks:
            continue

        fires = preds.get(qid, [])
        fire_times = [f['time'] for f in fires]

        per_trigger = []
        for trig in info['triggers']:
            gt_time = trig['time']
            hits = [t for t in fire_times if abs(t - gt_time) <= tolerance]
            per_trigger.append({
                'gt_time': gt_time,
                'detected': len(hits) > 0,
                'closest': min((abs(t - gt_time) for t in fire_times), default=None),
            })

        n_triggers = len(info['triggers'])
        n_detected = sum(1 for t in per_trigger if t['detected'])
        gt_times_set = set()
        for trig in info['triggers']:
            for t in range(int(trig['time'] - tolerance), int(trig['time'] + tolerance) + 1):
                gt_times_set.add(t)
        n_false = sum(1 for t in fire_times if int(t) not in gt_times_set)

        results.append({
            'id': qid,
            'task': info['task'],
            'duration': info['duration'],
            'n_triggers': n_triggers,
            'n_detected': n_detected,
            'n_fires': len(fires),
            'n_false_alarms': n_false,
            'trigger_recall': n_detected / n_triggers if n_triggers > 0 else 0,
            'per_trigger': per_trigger,
        })

    if not results:
        return results, {}

    n = len(results)
    total_triggers = sum(r['n_triggers'] for r in results)
    total_detected = sum(r['n_detected'] for r in results)
    total_fires = sum(r['n_fires'] for r in results)
    total_false = sum(r['n_false_alarms'] for r in results)

    recall = total_detected / total_triggers if total_triggers > 0 else 0
    precision = (total_fires - total_false) / total_fires if total_fires > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    by_task = defaultdict(lambda: {'n': 0, 'triggers': 0, 'detected': 0})
    for r in results:
        by_task[r['task']]['n'] += 1
        by_task[r['task']]['triggers'] += r['n_triggers']
        by_task[r['task']]['detected'] += r['n_detected']

    # Firing accuracy by duration bucket (for the decay curve)
    buckets = [(0, 60), (60, 120), (120, 180), (180, 300), (300, 600)]
    by_duration = {}
    for lo, hi in buckets:
        bucket_results = [r for r in results if lo <= r['duration'] < hi]
        if bucket_results:
            bt = sum(r['n_triggers'] for r in bucket_results)
            bd = sum(r['n_detected'] for r in bucket_results)
            by_duration[f'{lo}-{hi}s'] = {'n': len(bucket_results), 'recall': bd / bt if bt > 0 else 0}

    summary = {
        'n_samples': n,
        'total_triggers': total_triggers,
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'total_fires': total_fires,
        'total_false_alarms': total_false,
        'tolerance': tolerance,
        'by_task': dict(by_task),
        'by_duration': by_duration,
    }
    return results, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_file', required=True)
    parser.add_argument('--tolerance', type=float, default=5.0,
                        help='Seconds of slack around each trigger time (default: 5.0)')
    parser.add_argument('--visual_only', action='store_true',
                        help='Only evaluate samples with audio_dependency=none')
    parser.add_argument('--tasks', nargs='+', default=None,
                        help='Filter to specific task types')
    parser.add_argument('--out_file', default=None)
    args = parser.parse_args()

    gt = load_omnipro_gt()
    preds = load_pred(args.pred_file)

    results, summary = evaluate(gt, preds, args.tolerance, args.visual_only, args.tasks)

    print(f"\n{'='*60}")
    print(f"OmniPro Evaluation  (tol={summary['tolerance']}s, visual_only={args.visual_only})")
    print(f"{'='*60}")
    print(f"Samples:          {summary['n_samples']}")
    print(f"Trigger Recall:   {summary['recall']:.3f}  ({sum(r['n_detected'] for r in results)}/{summary['total_triggers']})")
    print(f"Precision:        {summary['precision']:.3f}")
    print(f"F1:               {summary['f1']:.3f}")
    print(f"False alarms:     {summary['total_false_alarms']}")

    print(f"\nBy task:")
    for task, v in summary['by_task'].items():
        r = v['detected'] / v['triggers'] if v['triggers'] > 0 else 0
        print(f"  {task:35s}  n={v['n']:3d}  recall={r:.3f}")

    print(f"\nFiring accuracy by video duration (decay curve):")
    for bucket, v in summary['by_duration'].items():
        print(f"  {bucket:10s}  n={v['n']:3d}  recall={v['recall']:.3f}")
    print(f"{'='*60}\n")

    if args.out_file:
        with open(args.out_file, 'w') as f:
            f.write(json.dumps(summary) + '\n')
            for r in results:
                f.write(json.dumps(r) + '\n')
        print(f"Per-sample results written to {args.out_file}")


if __name__ == '__main__':
    main()
