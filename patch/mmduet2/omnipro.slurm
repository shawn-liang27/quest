#!/bin/bash
#SBATCH -p gpu
#SBATCH -q normal
#SBATCH --mem-per-cpu=16gb
#SBATCH --time=0-8:00:00
#SBATCH -c 8
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:2
#SBATCH -J mmduet2-omnipro
#SBATCH -o out/%x-%J.out
#SBATCH -e out/%x-%J.err

conda activate mmduet2-infer

cd $SLURM_SUBMIT_DIR

llm=${LLM_PRETRAINED:-wangyueqian/MMDuet2}
omnipro_dir=${OMNIPRO_DIR:-/mnt/data0/sgl57/data/omnipro}
output_dir=outputs/mmduet2

mkdir -vp ${output_dir}/eval

pred_file=${output_dir}/eval/omnipro_visual-pred.jsonl
eval_file=${output_dir}/eval/omnipro_visual-eval.jsonl

# --------------------
# run inference (2 GPUs via device_map=auto)
# --------------------
python -u patch/mmduet2/inference_omnipro.py \
    --llm_pretrained ${llm} \
    --video_base ${omnipro_dir} \
    --frame_interval 2.0 --max_num_frames 200 \
    --visual_only \
    --device_map auto \
    --output_fname ${pred_file}

# --------------------
# evaluate — timing + language
# --------------------
python -u patch/mmduet/eval_omnipro.py \
    --pred_file ${pred_file} \
    --visual_only --tolerance 5.0 --eval_language \
    --out_file ${eval_file}
