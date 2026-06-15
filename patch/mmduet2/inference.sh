# ORIGNAL inference.sh

exp_name=mmduet2

dataset=ego
output_dir=outputs/${exp_name}/${dataset}
mkdir -p $output_dir

python -u inference.py \
    --start_idx $((interval*i)) --end_idx $((interval*i+interval)) \
    --llm_pretrained MMDUET2_CKPT \
    --test_fname ./data/annotations/${dataset}-frame_input_format.json \
    --output_fname ${output_dir}/pred.jsonl \
    >  ${output_dir}/pred.log 2>&1 &