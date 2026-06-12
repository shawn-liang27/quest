#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# --------------------------------------------------------------------------
# PATCHED: resolves stale mm_compressor path from the checkpoint config.
#
# Replace: dispider/model/builder.py
# --------------------------------------------------------------------------

import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from dispider.model import *
from dispider.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


def _resolve_compressor_path(config, model_path):
    """Fix mm_compressor when it points to the original author's absolute path."""
    compressor = getattr(config, 'mm_compressor', None)
    if compressor is None or os.path.exists(compressor):
        return

    compressor_name = os.path.basename(compressor)

    if os.path.isdir(model_path):
        local_dir = model_path
    else:
        try:
            from huggingface_hub import snapshot_download
            local_dir = snapshot_download(model_path)
        except Exception:
            return

    resolved = os.path.join(local_dir, compressor_name)
    if os.path.exists(resolved):
        print(f"[builder] Resolved compressor path: {resolved}")
        config.mm_compressor = resolved


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", **kwargs):
    kwargs = {"device_map": device_map, **kwargs}
    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)

    # --- PATCHED: load config first, fix compressor path, then pass to from_pretrained ---
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    _resolve_compressor_path(config, model_path)
    model = LongQwen2ForCausalLM.from_pretrained(model_path, config=config, low_cpu_mem_usage=True, **kwargs)
    # --- END PATCH ---

    image_processor = None

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    compressor = model.get_compressor()
    image_processor = compressor.compressor.get_vision_tower().image_processor
    time_tokenizer = compressor.tokenizer
    image_processor = (image_processor, time_tokenizer)

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
