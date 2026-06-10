# # import json
# # with open('/mnt/data0/sgl57/roma_proactive/alert.jsonl') as f:
# #     durations = []
# #     for line in f:
# #         d = json.loads(line)
# #         times = [a['time'] for a in d['ans']]
# #         durations.append(max(times))
# #     durations.sort()
# #     print(f'Count: {len(durations)}')
# #     print(f'Median last event time: {durations[len(durations)//2]}s')
# #     print(f'90th percentile: {durations[int(len(durations)*0.9)]}s')
# #     print(f'Max: {durations[-1]}s')

# # with open('/mnt/data0/sgl57/roma_proactive/narration.jsonl') as f:
# #     for i, line in enumerate(f):
# #         d = json.loads(line)
# #         times = [a['time'] for a in d['ans']]
# #         if i == 0: print(d.keys(), d['task'])
# #         if i < 3: print(f"id={d['id']}, last_event={max(times)}s")
# #     print(f'Total: {i+1}')

# from datasets import load_dataset

# # OmniPro: which tasks have no audio dependency?
# ds = load_dataset('RuixiangZhao/OmniPro', split='test')
# from collections import Counter
# visual_tasks = Counter()
# visual_durations = []
# for d in ds:
#     if d['audio_dependency'] == 'none':
#         visual_tasks[d['task']] += 1
#         visual_durations.append(d['duration'])
# visual_durations.sort()
# print('=== OmniPro (audio_dependency=none) ===')
# print(f'Total: {len(visual_durations)}')
# print(f'By task: {dict(visual_tasks)}')
# if visual_durations:
#     print(f'Median duration: {visual_durations[len(visual_durations)//2]:.0f}s')
#     print(f'90th pct: {visual_durations[int(len(visual_durations)*0.9)]:.0f}s')
#     print(f'Max: {visual_durations[-1]:.0f}s')

# from datasets import load_dataset
# ds = load_dataset('mjuicem/StreamingBench', 'Proactive_Output', split='Proactive_Output', verification_mode='no_checks')
# print(f'Samples: {len(ds)}')

# # Check the 250 split too
# ds2 = load_dataset('mjuicem/StreamingBench', 'Proactive_Output', split='Proactive_Output_250', verification_mode='no_checks')
# print(f'Samples (250): {len(ds2)}')
# print(f'Columns: {ds2.column_names}')
# print(ds2[0])

# # Duration range from timestamps
# def ts_to_sec(ts):
#     parts = ts.split(':')
#     return int(parts[0])*3600 + int(parts[1])*60 + int(parts[2])

# times = sorted([ts_to_sec(d['ground_truth_time_stamp']) for d in ds2])
# print(f'Median GT time: {times[len(times)//2]}s')
# print(f'90th pct: {times[int(len(times)*0.9)]}s')
# print(f'Max: {times[-1]}s')

# from datasets import load_dataset
# ds = load_dataset('RuixiangZhao/OmniPro', split='test')
# row = ds[0]
# print(type(row['file_name']), row['file_name'])
# # Check if there's a video column we missed
# print([c for c in ds.column_names if 'video' in c.lower() or 'file' in c.lower()])

# print(ds.features)
# # See if file_name resolves to something in cache
# import os
# print(os.path.exists(row['file_name']))

from datasets import load_dataset
ds = load_dataset('mjuicem/StreamingBench', 'Proactive_Output', split='Proactive_Output', verification_mode='no_checks')
print(ds.features)
print(ds.column_names)
# Check all configs for video columns
for cfg in ['Real_Time_Visual_Understanding', 'Contextual_Understanding', 'Omni_Source_Understanding']:
    try:
        ds2 = load_dataset('mjuicem/StreamingBench', cfg, split=cfg, verification_mode='no_checks')
        print(f'\n{cfg}: {ds2.features}')
        video_cols = [c for c in ds2.column_names if 'video' in c.lower() or 'file' in c.lower() or 'path' in c.lower()]
        print(f'  video-related cols: {video_cols}')
    except: pass