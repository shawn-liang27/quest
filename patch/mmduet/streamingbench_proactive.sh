output_dir=outputs/mmduet
mkdir -vp ${output_dir}/eval

thres_sum=2
sb_dir=/mnt/data0/sgl57/streamingbench

# NOTE: verify video path mapping before running at scale.
# StreamingBench metadata has no video path column — the video_map_fn
# in StreamingBenchProactiveDataset needs to match your local file layout.
# Run with --end_idx 5 first to confirm videos load correctly.

# --------------------
# run inference
# --------------------
python -u -m test.inference \
    --llm_pretrained lmms-lab/llava-onevision-qwen2-7b-ov --bf16 true \
    --lora_pretrained ${output_dir} \
    --input_dir ${sb_dir} --frame_fps 2 --max_num_frames 200 \
    --dataset streamingbench \
    --stream_end_score_sum_threshold ${thres_sum} --remove_assistant_turns true \
    --output_fname ${output_dir}/eval/streamingbench_proactive-thres_sum_${thres_sum}-rm_ass_turns-pred.jsonl \
    > ${output_dir}/eval/streamingbench_proactive-thres_sum_${thres_sum}-rm_ass_turns-pred.log 2>&1 &
wait

# --------------------
# evaluate — includes decay curve by trigger time
# --------------------
python -m test.eval_streamingbench \
    --pred_file ${output_dir}/eval/streamingbench_proactive-thres_sum_${thres_sum}-rm_ass_turns-pred.jsonl \
    --tolerance 5.0 \
    --out_file ${output_dir}/eval/streamingbench_proactive-thres_sum_${thres_sum}-rm_ass_turns-eval.jsonl \
    > ${output_dir}/eval/streamingbench_proactive-thres_sum_${thres_sum}-rm_ass_turns-eval.log 2>&1 &
