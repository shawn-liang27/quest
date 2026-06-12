"""
DiSPider OmniPro proactive monitoring inference.

Two-pass approach:
  1. Streaming model (forward_inference) -> fire positions + timestamps
  2. Long model (generate) -> text at each fire position (truncated to fire clip)

Output: JSONL compatible with eval_omnipro.py

Usage:
    python model_omnipro.py \
        --model-path Mar2Ding/Dispider \
        --video-base /path/to/omnipro/videos \
        --output-file outputs/dispider/omnipro-pred.jsonl \
        --visual_only
"""

import argparse
import json
import os
import sys
import math
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from decord import VideoReader

from transformers import StoppingCriteria, StoppingCriteriaList

from dispider.constants import (
    IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN, DEFAULT_ANS_TOKEN, DEFAULT_TODO_TOKEN,
)
from dispider.conversation import conv_templates
from dispider.model.builder import load_pretrained_model
from dispider.utils import disable_torch_init
from dispider.mm_utils import tokenizer_image_token, get_model_name_from_path


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False


# ---------------------------------------------------------------------------
# Video loading utilities (adapted from baseline/Dispider/inference.py)
# ---------------------------------------------------------------------------

def get_seq_frames(total_num_frames, desired_num_frames):
    seg_size = float(total_num_frames - 1) / desired_num_frames
    seq = []
    for i in range(desired_num_frames):
        start = int(np.round(seg_size * i))
        end = int(np.round(seg_size * (i + 1)))
        seq.append((start + end) // 2)
    return seq


def get_seq_time(vr, frame_idx, num_clip):
    frm_per_clip = len(frame_idx) // num_clip
    key_frame = [
        [frame_idx[i * frm_per_clip], frame_idx[i * frm_per_clip + frm_per_clip - 1]]
        for i in range(num_clip)
    ]
    time = vr.get_frame_timestamp(key_frame)
    return np.hstack([time[:, 0, 0], time[:, 1, 1]])


def load_video(vis_path, num_frm=16, max_clip=32):
    vr = VideoReader(vis_path, num_threads=1)
    total_frame_num = len(vr)
    fps = vr.get_avg_fps()
    total_time = total_frame_num / fps

    num_clip = total_time / num_frm
    num_clip = int(np.round(num_clip)) if num_clip > 1 else 1
    num_clip = max(num_clip, 1)
    num_clip = min(num_clip, max_clip)
    total_num_frm = num_frm * num_clip
    frame_idx = get_seq_frames(total_frame_num, total_num_frm)

    time_idx = get_seq_time(vr, frame_idx, num_clip)
    img_array = vr.get_batch(frame_idx).asnumpy()

    a, H, W, _ = img_array.shape
    if H != W:
        img_array = torch.from_numpy(img_array).permute(0, 3, 1, 2).float()
        img_array = torch.nn.functional.interpolate(
            img_array, size=(min(H, W), min(H, W))
        )
        img_array = img_array.permute(0, 2, 3, 1).to(torch.uint8).numpy()

    img_array = img_array.reshape(
        (1, total_num_frm, img_array.shape[-3], img_array.shape[-2], img_array.shape[-1])
    )

    clip_imgs = [Image.fromarray(img_array[0, j]) for j in range(total_num_frm)]
    return clip_imgs, time_idx, num_clip


def preprocess_time(time, num_clip, tokenizer):
    time = time.reshape(2, num_clip)
    seq = []
    for i in range(num_clip):
        start, end = time[:, i]
        start = int(np.round(start))
        end = int(np.round(end))
        sentence = (
            "This contains a clip sampled in %d to %d seconds" % (start, end)
            + DEFAULT_IMAGE_TOKEN
        )
        sentence = tokenizer_image_token(sentence, tokenizer, return_tensors="pt")
        seq.append(sentence)
    return seq


def preprocess_question(questions, tokenizer):
    seq = []
    for q in questions:
        sentence = tokenizer_image_token(
            q + DEFAULT_TODO_TOKEN, tokenizer, return_tensors="pt"
        )
        seq.append(sentence)
    return seq


# ---------------------------------------------------------------------------
# Data processing
# ---------------------------------------------------------------------------

def process_data(
    video_path, question, model_config, tokenizer,
    processor, time_tokenizer, conv_mode="qwen", max_clip=32,
):
    num_frames = 16

    frames, time_idx, num_clips = load_video(video_path, num_frames, max_clip)

    video = processor.preprocess(frames, return_tensors="pt")["pixel_values"]
    video = video.view(num_clips, num_frames, *video.shape[1:])

    video_large = processor.preprocess(frames, return_tensors="pt")["pixel_values"]
    video_large = video_large.view(num_clips, num_frames, *video_large.shape[1:])[:, :1].contiguous()

    seqs = preprocess_time(time_idx, num_clips, time_tokenizer)
    seqs = torch.nn.utils.rnn.pad_sequence(
        seqs, batch_first=True, padding_value=time_tokenizer.pad_token_id
    )
    compress_mask = seqs.ne(time_tokenizer.pad_token_id)

    qs = preprocess_question([question], time_tokenizer)
    qs = torch.nn.utils.rnn.pad_sequence(
        qs, batch_first=True, padding_value=time_tokenizer.pad_token_id
    )
    qs_mask = qs.ne(time_tokenizer.pad_token_id)

    if model_config.mm_use_im_start_end:
        prompt_qs = (
            DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            + "\n" + question
        )
    else:
        prompt_qs = DEFAULT_IMAGE_TOKEN + "\n" + question

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], prompt_qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    )

    return input_ids, video, video_large, seqs, compress_mask, qs, qs_mask, time_idx, num_clips


def clip_index_to_time(clip_idx, time_idx, num_clips):
    """Convert 1-indexed clip index to timestamp (end-time of clip in seconds)."""
    time = time_idx.reshape(2, num_clips)
    if clip_idx <= 0:
        return 0.0
    clip_idx_0 = min(clip_idx, num_clips) - 1
    return float(time[1, clip_idx_0])


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_omnipro_dataset(video_base, visual_only=True, tasks=None,
                         start_idx=0, end_idx=None):
    from datasets import load_dataset

    ds = load_dataset("RuixiangZhao/OmniPro", split="test")
    data = [ds[i] for i in range(len(ds))]

    if visual_only:
        data = [d for d in data if d.get("audio_dependency") == "none"]
    if tasks:
        data = [d for d in data if d["task"] in tasks]

    data = data[start_idx:end_idx]

    samples = []
    for d in data:
        video_path = os.path.join(video_base, d["file_name"])
        samples.append({
            "id": d["id"],
            "question": d["question"],
            "video_path": video_path,
            "task": d["task"],
        })

    print(f"Loaded {len(samples)} OmniPro samples (visual_only={visual_only})")
    return samples


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def eval_omnipro(args):
    disable_torch_init()

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name
    )
    model.eval()

    image_processor, time_tokenizer = image_processor
    if time_tokenizer.pad_token is None:
        time_tokenizer.pad_token = "<pad>"

    compressor = model.get_compressor()
    streaming_model = compressor.compressor

    stop_words_ids = [torch.tensor(tokenizer("<|im_end|>").input_ids).cuda()]
    stopping_criteria = StoppingCriteriaList(
        [StoppingCriteriaSub(stops=stop_words_ids)]
    )

    ans_token_ids = time_tokenizer(
        DEFAULT_ANS_TOKEN, return_tensors="pt"
    ).input_ids.cuda()
    todo_token_ids = time_tokenizer(
        DEFAULT_TODO_TOKEN, return_tensors="pt"
    ).input_ids.cuda()

    samples = load_omnipro_dataset(
        args.video_base,
        visual_only=args.visual_only,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

    with open(args.output_file, "w") as fout:
        for sample in tqdm(samples, desc="DiSPider OmniPro"):
            qid = sample["id"]
            video_path = sample["video_path"]
            question = sample["question"]

            if not os.path.exists(video_path):
                print(f"Video not found: {video_path}, skipping")
                fout.write(
                    json.dumps({"question_id": qid, "model_response_list": []}) + "\n"
                )
                fout.flush()
                continue

            try:
                (
                    input_ids, video, video_large,
                    seqs, compress_mask, qs, qs_mask,
                    time_idx, num_clips,
                ) = process_data(
                    video_path, question, model.config, tokenizer,
                    image_processor, time_tokenizer,
                    conv_mode=args.conv_mode, max_clip=args.max_clip,
                )

                # ---- Pass 1: fire positions from streaming model ----
                with torch.inference_mode():
                    ans_position, silent_list = streaming_model.forward_inference(
                        input_ids=seqs.cuda(),
                        attention_mask=compress_mask.cuda(),
                        qs_ids=qs.cuda(),
                        qs_mask=qs_mask.cuda(),
                        images=video.to(dtype=torch.float16, device="cuda"),
                        ans_token=ans_token_ids,
                        todo_token=todo_token_ids,
                    )

                model_responses = []

                if ans_position:
                    # ---- Pass 2: generate text at each fire position ----
                    for fp in ans_position:
                        fire_time = clip_index_to_time(fp, time_idx, num_clips)

                        # Truncate to clips [0, fp) for faithful streaming eval
                        t_video = video[:fp].contiguous()
                        t_video_large = video_large[:fp].contiguous()
                        t_seqs = seqs[:fp].contiguous()
                        t_compress_mask = compress_mask[:fp].contiguous()

                        with torch.inference_mode():
                            output_ids = model.generate(
                                input_ids.unsqueeze(0).cuda(),
                                images=t_video.to(
                                    dtype=torch.float16, device="cuda"
                                ),
                                images_large=t_video_large.to(
                                    dtype=torch.float16, device="cuda"
                                ),
                                seqs=t_seqs.cuda(),
                                compress_mask=t_compress_mask.cuda(),
                                qs=qs.cuda(),
                                qs_mask=qs_mask.cuda(),
                                ans_token=ans_token_ids,
                                todo_token=todo_token_ids,
                                insert_position=0,
                                ans_position=[],
                                do_sample=False,
                                max_new_tokens=args.max_new_tokens,
                                pad_token_id=tokenizer.eos_token_id,
                                stopping_criteria=stopping_criteria,
                                use_cache=True,
                            )

                        response = tokenizer.batch_decode(
                            output_ids, skip_special_tokens=True
                        )[0].strip()

                        model_responses.append({
                            "time": fire_time,
                            "content": response,
                            "role": "assistant",
                        })

                result = {
                    "question_id": qid,
                    "model_response_list": model_responses,
                    "ans_position": ans_position,
                    "silent_list": silent_list,
                }
                fout.write(json.dumps(result) + "\n")
                fout.flush()

            except Exception as e:
                print(f"Error processing {qid}: {e}")
                import traceback
                traceback.print_exc()
                fout.write(
                    json.dumps({"question_id": qid, "model_response_list": []}) + "\n"
                )
                fout.flush()

    print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DiSPider OmniPro inference")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--video-base", type=str, required=True,
                        help="Base directory with OmniPro videos")
    parser.add_argument("--output-file", type=str,
                        default="outputs/dispider/omnipro-pred.jsonl")
    parser.add_argument("--conv-mode", type=str, default="qwen")
    parser.add_argument("--max-clip", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--visual_only", action="store_true", default=True)
    parser.add_argument("--no_visual_only", dest="visual_only",
                        action="store_false")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    args = parser.parse_args()

    eval_omnipro(args)
