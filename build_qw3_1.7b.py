from tyllm.build_util import build_and_compile_llm
from tyllm import torch_edgex
from vllm.config import ModelConfig
ModelConfig.verify_with_parallel_config = lambda a, b: True

quant_path = "./quantized_models/Qwen3-1.7B-AWQ"
aot_path = f"./compiled_models/Qwen3-1.7B-AWQ-AOT"

# 预填充序列长度
prefill_seq_len = 96
# 最大KV键值对数，控制模型推理期间上下文长度
max_kv_cache_size = 8192
# 指定多die编译，多die并行计算
die_num = 1
# 是否将embedding操作作为输入，默认False；如果True，embedding计算将被offload到cpu
embedding_as_input = False

torch_edgex.set_device_trace_only("edgex", True)
torch_edgex.set_device_mode("LM_die_remap", [0,1,2,3,0,0,0,0,0,0,0,0,0,0,0,0]) #1230

build_and_compile_llm(
    model_path=quant_path,
    artifacts_path=f"{aot_path}_{prefill_seq_len}_{max_kv_cache_size}/{die_num}die",
    max_kv_cache_size=max_kv_cache_size,
    seq_len_list=[1, prefill_seq_len],
    dev_count=die_num,
    embedding_as_input=embedding_as_input,
)
