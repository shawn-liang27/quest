#!/usr/bin/env bash
set -euo pipefail

model_path=${MODEL_PATH:-Mar2Ding/Dispider}
omnipro_dir=${OMNIPRO_DIR:-/mnt/data0/sgl57/data/omnipro}
dispider_root=${DISPIDER_ROOT:-baseline/Dispider}
output_dir=outputs/dispider
max_clip=32

mkdir -vp ${output_dir}/eval

# --------------------
# apply patches
# --------------------
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
echo "[gpu] Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

echo "[patch] Copying patched files into ${dispider_root} ..."
cp -v patch/dispider/builder.py       ${dispider_root}/dispider/model/builder.py
cp -v patch/dispider/clip_encoder.py  ${dispider_root}/dispider/model/multimodal_encoder/clip_encoder.py

pred_file=${output_dir}/eval/omnipro_visual-pred.jsonl
eval_file=${output_dir}/eval/omnipro_visual-eval.jsonl

# --------------------
# run inference
# --------------------
python -u patch/dispider/model_omnipro.py \
    --model-path ${model_path} \
    --video-base ${omnipro_dir} \
    --output-file ${pred_file} \
    --max-clip ${max_clip} \
    --visual_only \
    > ${output_dir}/eval/omnipro_visual-pred.log 2>&1

# --------------------
# evaluate — timing + language
# --------------------
python -u patch/mmduet/eval_omnipro.py \
    --pred_file ${pred_file} \
    --visual_only --tolerance 5.0 --eval_language \
    --out_file ${eval_file} \
    > ${output_dir}/eval/omnipro_visual-eval.log 2>&1
