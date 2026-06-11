output_dir=outputs/mmduet
mkdir -vp ${output_dir}/eval

thres_sum=2
omnipro_dir=/mnt/data0/sgl57/data/omnipro

# --------------------
# run inference (visual-only, all proactive tasks)
# --------------------
python -u -m test.inference \
    --llm_pretrained lmms-lab/llava-onevision-qwen2-7b-ov --bf16 true \
    --lora_pretrained ${output_dir} \
    --input_dir ${omnipro_dir} --frame_fps 2 --max_num_frames 200 \
    --dataset omnipro \
    --stream_end_score_sum_threshold ${thres_sum} --remove_assistant_turns true \
    --output_fname ${output_dir}/eval/omnipro_visual-thres_sum_${thres_sum}-rm_ass_turns-pred.jsonl \
    > ${output_dir}/eval/omnipro_visual-thres_sum_${thres_sum}-rm_ass_turns-pred.log 2>&1 &
wait

# --------------------
# evaluate — includes per-task and decay curve breakdown
# --------------------
python -m test.eval_omnipro \
    --pred_file ${output_dir}/eval/omnipro_visual-thres_sum_${thres_sum}-rm_ass_turns-pred.jsonl \
    --visual_only --tolerance 5.0 \
    --out_file ${output_dir}/eval/omnipro_visual-thres_sum_${thres_sum}-rm_ass_turns-eval.jsonl \
    > ${output_dir}/eval/omnipro_visual-thres_sum_${thres_sum}-rm_ass_turns-eval.log 2>&1 &
