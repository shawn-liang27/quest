output_dir=outputs/mmduet
mkdir -vp  ${output_dir}/eval

thres_sum=2
roma_dir=/mnt/data0/sgl57/roma_proactive

# --------------------
# run inference
# --------------------
python -u -m test.inference \
    --llm_pretrained lmms-lab/llava-onevision-qwen2-7b-ov --bf16 true \
    --lora_pretrained ${output_dir} \
    --input_dir ${roma_dir} --frame_fps 2 --max_num_frames 100 \
    --test_fname ${roma_dir}/alert.jsonl \
    --dataset roma \
    --stream_end_score_sum_threshold ${thres_sum} --remove_assistant_turns true \
    --output_fname ${output_dir}/eval/roma_alert-thres_sum_${thres_sum}-rm_ass_turns-pred.jsonl \
    > ${output_dir}/eval/roma_alert-thres_sum_${thres_sum}-rm_ass_turns-pred.log 2>&1 &
wait

# --------------------
# evaluate against ROMA ground truth
# --------------------
python -m test.eval_roma \
    --gt_file ${roma_dir}/alert.jsonl \
    --pred_file ${output_dir}/eval/roma_alert-thres_sum_${thres_sum}-rm_ass_turns-pred.jsonl \
    --tolerance 1.0 \
    --out_file ${output_dir}/eval/roma_alert-thres_sum_${thres_sum}-rm_ass_turns-eval.jsonl \
    > ${output_dir}/eval/roma_alert-thres_sum_${thres_sum}-rm_ass_turns-eval.log 2>&1 &
