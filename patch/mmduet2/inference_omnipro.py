"""
MMDuet2 (Qwen2.5-VL) OmniPro proactive monitoring inference.

Loads OmniPro metadata from local directory, extracts frames with the same
cv2 backend as MMDuet1, builds Qwen2.5-VL multimodal conversation, and
runs ProactiveInferenceClient per sample.

Output: JSONL compatible with eval_omnipro.py

Usage:
    python patch/mmduet2/inference_omnipro.py \
        --llm_pretrained wangyueqian/MMDuet2 \
        --video_base /path/to/omnipro \
        --output_fname outputs/mmduet2/eval/omnipro_visual-pred.jsonl \
        --frame_interval 2.0 --max_num_frames 200 --visual_only
"""

import json, math, os
import cv2
import numpy as np
from dataclasses import dataclass
from PIL import Image
from tqdm import tqdm
from transformers import HfArgumentParser

from inference import (
    ProactiveInferenceClient,
    ProactiveTestArguments,
    post_process_conversation_for_print,
)


@dataclass
class OmniProTestArguments(ProactiveTestArguments):
    llm_pretrained: str = 'wangyueqian/MMDuet2'
    video_base: str = ''
    frame_interval: float = 2.0
    max_num_frames: int = 200
    output_resolution: int = 384
    visual_only: bool = True
    device_map: str = None


def load_omnipro_samples(video_base, visual_only=True, start_idx=0, end_idx=None):
    """Load OmniPro metadata from local dataset directory."""
    from datasets import load_dataset
    ds = load_dataset(video_base, split="test")
    data = [ds[i] for i in range(len(ds))]

    if visual_only:
        data = [d for d in data if d.get("audio_dependency") == "none"]

    data = data[start_idx:end_idx]

    samples = []
    for d in data:
        samples.append({
            "id": d["id"],
            "question": d["question"],
            "file_name": d["file_name"],
            "task": d["task"],
        })
    print(f"Loaded {len(samples)} OmniPro samples (visual_only={visual_only})")
    return samples


def load_video(video_path, output_fps, max_num_frames, output_resolution=384):
    """Extract frames using the same cv2 method as MMDuet1.

    Matches patch/mmduet/datasets.py FastAndAccurateStreamingVideoQADataset.load_video
    but returns PIL images instead of torch tensors (for Qwen2.5-VL).
    """
    cap = cv2.VideoCapture(video_path)
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    video_duration = frame_count / input_fps
    input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    num_frames_total = math.ceil(video_duration * output_fps)
    num_frames_total = min(num_frames_total, max_num_frames)
    frame_sec = [i / output_fps for i in range(num_frames_total)]

    frame_list = []
    cur_time = 0
    frame_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_index < len(frame_sec) and cur_time >= frame_sec[frame_index]:
            if input_width > input_height:
                new_width = output_resolution
                new_height = int((input_height / input_width) * output_resolution)
            else:
                new_height = output_resolution
                new_width = int((input_width / input_height) * output_resolution)
            resized = cv2.resize(frame, (new_width, new_height))
            canvas = cv2.copyMakeBorder(
                resized,
                top=(output_resolution - new_height) // 2,
                bottom=(output_resolution - new_height + 1) // 2,
                left=(output_resolution - new_width) // 2,
                right=(output_resolution - new_width + 1) // 2,
                borderType=cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )
            rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            frame_list.append(Image.fromarray(rgb))
            frame_index += 1
        if len(frame_list) >= max_num_frames:
            break
        cur_time += 1.0 / input_fps
    cap.release()
    return frame_list, output_fps, video_duration


def build_conversation(question, frames, system_prompt):
    """Build Qwen2.5-VL multimodal conversation for proactive monitoring."""
    conversation = [{"role": "system", "content": system_prompt}]

    first_content = [
        {"type": "image", "image": frames[0]},
        {"type": "text", "text": question},
    ]
    conversation.append({"role": "user", "content": first_content})

    for frame in frames[1:]:
        conversation.append({
            "role": "user",
            "content": [{"type": "image", "image": frame}],
        })

    return conversation


def main():
    args, = HfArgumentParser(OmniProTestArguments).parse_args_into_dataclasses()
    print(args)

    output_fps = 1.0 / args.frame_interval

    samples = load_omnipro_samples(
        args.video_base,
        visual_only=args.visual_only,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
    )

    existing_ids = set()
    if os.path.exists(args.output_fname):
        for line in open(args.output_fname):
            if line.strip():
                existing_ids.add(json.loads(line)["question_id"])
        print(f"Found {len(existing_ids)} existing predictions, will skip those")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_fname)), exist_ok=True)
    f_out = open(args.output_fname, "a")

    wrapper = ProactiveInferenceClient(args)
    wrapper.set_fps(frame_interval=args.frame_interval)

    for sample_i, sample in enumerate(tqdm(samples, desc="MMDuet2 OmniPro")):
        qid = sample["id"]
        if qid in existing_ids:
            continue

        video_path = os.path.join(args.video_base, sample["file_name"])
        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}, skipping")
            f_out.write(json.dumps({"question_id": qid, "model_response_list": []}) + "\n")
            f_out.flush()
            continue

        try:
            frames, fps, duration = load_video(
                video_path, output_fps, args.max_num_frames,
                output_resolution=args.output_resolution,
            )
            if not frames:
                print(f"No frames extracted for {qid}, skipping")
                f_out.write(json.dumps({"question_id": qid, "model_response_list": []}) + "\n")
                f_out.flush()
                continue

            conversation = build_conversation(
                sample["question"], frames, args.system_prompt,
            )

            wrapper.reset()
            wrapper.set_fps(frame_interval=args.frame_interval)
            wrapper.input_query_stream(conversation)
            model_outputs = wrapper.inference(
                debug_print=(sample_i < 2),
            )

            processed = post_process_conversation_for_print(model_outputs["conversation"])

            res = {
                "question_id": qid,
                "model_response_list": processed,
            }
            f_out.write(json.dumps(res) + "\n")
            f_out.flush()

        except Exception as e:
            print(f"Error processing {qid}: {e}")
            import traceback
            traceback.print_exc()
            f_out.write(json.dumps({"question_id": qid, "model_response_list": []}) + "\n")
            f_out.flush()

    f_out.close()
    print(f"Results saved to {args.output_fname}")


if __name__ == "__main__":
    main()
