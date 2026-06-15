#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
echo "[gpu] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

llm=${LLM_PRETRAINED:-wangyueqian/MMDuet2}
omnipro_dir=${OMNIPRO_DIR:-/mnt/data0/sgl57/data/omnipro}
output_dir=outputs/mmduet2

mkdir -vp ${output_dir}/eval

pred_file=${output_dir}/eval/omnipro_visual-pred.jsonl
eval_file=${output_dir}/eval/omnipro_visual-eval.jsonl

# --------------------
# run inference
# --------------------
python -u patch/mmduet2/inference_omnipro.py \
    --llm_pretrained ${llm} \
    --video_base ${omnipro_dir} \
    --frame_interval 2.0 --max_num_frames 200 \
    --visual_only \
    --output_fname ${pred_file} \
    > ${output_dir}/eval/omnipro_visual-pred.log 2>&1

# --------------------
# evaluate — timing + language
# --------------------
python -u patch/mmduet/eval_omnipro.py \
    --pred_file ${pred_file} \
    --visual_only --tolerance 5.0 --eval_language \
    --out_file ${eval_file} \
    > ${output_dir}/eval/omnipro_visual-eval.log 2>&1
