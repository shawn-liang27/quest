"""
Evaluate predictions against OmniPro ground truth (timing + language).

Usage:
    python eval_omnipro.py \
        --pred_file outputs/mmduet/eval/omnipro-pred.jsonl \
        --tolerance 5.0 \
        --visual_only \
        --eval_language

OmniPro ground_truth field (JSON string per sample):
    [{"trigger_time_sec": 164, "response": "...", ...}, ...]

Prediction output (one JSON per line):
    {"question_id": "...", "model_response_list": [
        {"time": 4.5, "content": "...", "role": "assistant"}, ...
    ]}
"""

import argparse
import json
from collections import defaultdict


# ---------------------------------------------------------------------------
# Language evaluation helpers
# ---------------------------------------------------------------------------

def _lcs_length(a, b):
    """Longest common subsequence length for two token lists."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev = curr
    return prev[n]


def rouge_l(hypothesis, reference):
    """Compute ROUGE-L F1 between two strings (word-level)."""
    hyp_tokens = hypothesis.lower().split()
    ref_tokens = reference.lower().split()
    if not hyp_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(hyp_tokens, ref_tokens)
    prec = lcs / len(hyp_tokens) if hyp_tokens else 0.0
    rec = lcs / len(ref_tokens) if ref_tokens else 0.0
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


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


def _match_fire_to_trigger(fires, gt_time, tolerance):
    """Find the closest fire within tolerance of a ground-truth trigger time.
    Returns the matched fire dict or None."""
    best, best_dist = None, float('inf')
    for f in fires:
        d = abs(f['time'] - gt_time)
        if d <= tolerance and d < best_dist:
            best = f
            best_dist = d
    return best


def evaluate(gt, preds, tolerance, visual_only=False, tasks=None,
             eval_language=False):
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
            trig_result = {
                'gt_time': gt_time,
                'detected': len(hits) > 0,
                'closest': min((abs(t - gt_time) for t in fire_times), default=None),
            }

            if eval_language and trig_result['detected']:
                matched = _match_fire_to_trigger(fires, gt_time, tolerance)
                if matched and matched.get('content') and trig.get('response'):
                    trig_result['rouge_l'] = rouge_l(
                        matched['content'], trig['response']
                    )
                    trig_result['pred_text'] = matched['content']
                    trig_result['gt_text'] = trig['response']

            per_trigger.append(trig_result)

        n_triggers = len(info['triggers'])
        n_detected = sum(1 for t in per_trigger if t['detected'])
        gt_times_set = set()
        for trig in info['triggers']:
            for t in range(int(trig['time'] - tolerance), int(trig['time'] + tolerance) + 1):
                gt_times_set.add(t)
        n_false = sum(1 for t in fire_times if int(t) not in gt_times_set)

        entry = {
            'id': qid,
            'task': info['task'],
            'duration': info['duration'],
            'n_triggers': n_triggers,
            'n_detected': n_detected,
            'n_fires': len(fires),
            'n_false_alarms': n_false,
            'trigger_recall': n_detected / n_triggers if n_triggers > 0 else 0,
            'per_trigger': per_trigger,
        }

        if eval_language:
            scores = [t['rouge_l'] for t in per_trigger if 'rouge_l' in t]
            entry['mean_rouge_l'] = sum(scores) / len(scores) if scores else None

        results.append(entry)

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

    by_task = defaultdict(lambda: {'n': 0, 'triggers': 0, 'detected': 0,
                                   'rouge_l_scores': []})
    for r in results:
        by_task[r['task']]['n'] += 1
        by_task[r['task']]['triggers'] += r['n_triggers']
        by_task[r['task']]['detected'] += r['n_detected']
        if eval_language:
            for t in r['per_trigger']:
                if 'rouge_l' in t:
                    by_task[r['task']]['rouge_l_scores'].append(t['rouge_l'])

    buckets = [(0, 60), (60, 120), (120, 180), (180, 300), (300, 600)]
    by_duration = {}
    for lo, hi in buckets:
        bucket_results = [r for r in results if lo <= r['duration'] < hi]
        if bucket_results:
            bt = sum(r['n_triggers'] for r in bucket_results)
            bd = sum(r['n_detected'] for r in bucket_results)
            bucket = {'n': len(bucket_results),
                      'recall': bd / bt if bt > 0 else 0}
            if eval_language:
                rl = [t['rouge_l'] for r in bucket_results
                      for t in r['per_trigger'] if 'rouge_l' in t]
                bucket['rouge_l'] = sum(rl) / len(rl) if rl else None
            by_duration[f'{lo}-{hi}s'] = bucket

    summary = {
        'n_samples': n,
        'total_triggers': total_triggers,
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'total_fires': total_fires,
        'total_false_alarms': total_false,
        'tolerance': tolerance,
        'by_task': {k: {kk: vv for kk, vv in v.items() if kk != 'rouge_l_scores'}
                    for k, v in by_task.items()},
        'by_duration': by_duration,
    }

    if eval_language:
        all_rl = [t['rouge_l'] for r in results
                  for t in r['per_trigger'] if 'rouge_l' in t]
        summary['language'] = {
            'mean_rouge_l': sum(all_rl) / len(all_rl) if all_rl else None,
            'n_scored': len(all_rl),
        }
        for k, v in by_task.items():
            rl = v['rouge_l_scores']
            summary['by_task'][k]['rouge_l'] = (
                sum(rl) / len(rl) if rl else None
            )

    return results, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_file', required=True)
    parser.add_argument('--tolerance', type=float, default=5.0,
                        help='Seconds of slack around each trigger time (default: 5.0)')
    parser.add_argument('--visual_only', action='store_true',
                        help='Only evaluate samples with audio_dependency=none')
    parser.add_argument('--eval_language', action='store_true',
                        help='Score response text against ground truth (ROUGE-L)')
    parser.add_argument('--tasks', nargs='+', default=None,
                        help='Filter to specific task types')
    parser.add_argument('--out_file', default=None)
    args = parser.parse_args()

    gt = load_omnipro_gt()
    preds = load_pred(args.pred_file)

    results, summary = evaluate(gt, preds, args.tolerance, args.visual_only,
                                args.tasks, eval_language=args.eval_language)

    print(f"\n{'='*60}")
    print(f"OmniPro Evaluation  (tol={summary['tolerance']}s, visual_only={args.visual_only})")
    print(f"{'='*60}")
    print(f"Samples:          {summary['n_samples']}")
    print(f"Trigger Recall:   {summary['recall']:.3f}  ({sum(r['n_detected'] for r in results)}/{summary['total_triggers']})")
    print(f"Precision:        {summary['precision']:.3f}")
    print(f"F1:               {summary['f1']:.3f}")
    print(f"False alarms:     {summary['total_false_alarms']}")

    if args.eval_language and 'language' in summary:
        lang = summary['language']
        print(f"\nLanguage quality (matched triggers only):")
        print(f"  ROUGE-L:        {lang['mean_rouge_l']:.3f}  (n={lang['n_scored']})")

    print(f"\nBy task:")
    for task, v in summary['by_task'].items():
        r = v['detected'] / v['triggers'] if v['triggers'] > 0 else 0
        line = f"  {task:35s}  n={v['n']:3d}  recall={r:.3f}"
        if args.eval_language and v.get('rouge_l') is not None:
            line += f"  rouge_l={v['rouge_l']:.3f}"
        print(line)

    print(f"\nFiring accuracy by video duration (decay curve):")
    for bucket, v in summary['by_duration'].items():
        line = f"  {bucket:10s}  n={v['n']:3d}  recall={v['recall']:.3f}"
        if args.eval_language and v.get('rouge_l') is not None:
            line += f"  rouge_l={v['rouge_l']:.3f}"
        print(line)
    print(f"{'='*60}\n")

    if args.out_file:
        with open(args.out_file, 'w') as f:
            f.write(json.dumps(summary) + '\n')
            for r in results:
                f.write(json.dumps(r) + '\n')
        print(f"Per-sample results written to {args.out_file}")


if __name__ == '__main__':
    main()
