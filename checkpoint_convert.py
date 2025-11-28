      
#!/usr/bin/env python3
"""
rename_quanted_keys.py

Usage:
    python rename_quanted_keys.py /path/to/original_folder

Behavior:
  - 复制原文件夹到 <original_folder>_new（若存在则 _new_1/_new_2... 保证唯一）
  - 在新目录中对所有 .safetensors 文件的 header keys 做后缀替换（不载入 tensor 数据）：
      ".quanted_layer.quanted_weight" -> ".qweight"
      ".quanted_layer.zero" -> ".qzeros"
      ".quanted_layer.weight_scale" -> ".scales"
  - 若有 model.safetensors.index.json 或其他 *.index.json，会尝试替换其中 "weight_map" 的 keys
  - 如果某个文件在替换后会产生重复 key，该文件会被跳过并打印警告（保持原样）
"""
from pathlib import Path
import argparse
import shutil
import sys
import struct
import json
import os
import tempfile

# 后缀替换规则（按长度降序，避免部分覆盖）
AWQ_REPLACEMENTS = [
    (".quanted_layer.quanted_weight", ".qweight"),
    (".quanted_layer.weight_scale", ".scales"),
    (".quanted_layer.zero", ".qzeros"),
    (".quanted_layer.bias", ".bias"),
]

VLLM_QWEN2_5_VL_REPLACEMENTS = [
    ("model.layers", ".quanted_layer.quanted_weight", ".qweight"),
    ("model.layers", ".quanted_layer.weight_scale", ".scales"),
    ("model.layers", ".quanted_layer.bias", ".bias"),
    ("visual", ".quanted_layer.quanted_weight", ".weight"),
    ("visual", ".quanted_layer.weight_scale", ".weight_scale"),
    ("visual", ".quanted_layer.static_act_scale", ".input_scale"),
    ("visual", ".quanted_layer.bias", ".bias"),
]

VLLM_QWEN3_VL_REPLACEMENTS = [
    ("model.language_model", ".quanted_layer.quanted_weight", ".qweight"),
    ("model.language_model", ".quanted_layer.weight_scale", ".scales"),
    ("model.language_model", ".quanted_layer.bias", ".bias"),
    ("model.visual", ".quanted_layer.quanted_weight", ".weight"),
    ("model.visual", ".quanted_layer.weight_scale", ".weight_scale"),
    ("model.visual", ".quanted_layer.static_act_scale", ".input_scale"),
    ("model.visual", ".quanted_layer.bias", ".bias"),
]

QUANT_METHOD = {
    "awq": AWQ_REPLACEMENTS,
    "awq_triton_w4a8": AWQ_REPLACEMENTS,
    "vllm_qwen2_5_vl": VLLM_QWEN2_5_VL_REPLACEMENTS,
    "vllm_qwen3_vl": VLLM_QWEN3_VL_REPLACEMENTS
}

AWQ_QUANT_CONFIG = {
  "quantization_config": {
    "quant_method": "awq",
    "zero_point": True,
    "group_size": 128,
    "bits": 4,
    "version": "gemm"
  }
}

AWQ_W4A8_QUANT_CONFIG = {
  "quantization_config": {
    "quant_method": "awq_triton_w4a8",
    "zero_point": False,
    "group_size": -1,
    "bits": 4,
    "version": "gemm"
  }
}

QUANT_CONFIG = {
    "awq": AWQ_QUANT_CONFIG,
    "awq_triton_w4a8": AWQ_W4A8_QUANT_CONFIG
}

def make_unique_copydir(src: Path, dst: Path = None) -> Path:
    """将 src 复制为 src_sglang（若存在则尝试 _sglang_1/_sglang_2 ...），返回新目录 Path"""
    parent = src.parent
    base = src.name
    if dst:
        candidate = dst
    else:
        # candidate = parent / f"{base}_sglang"
        candidate = src.with_name(src.name + f"_1")
    i = 1
    while candidate.exists():
        # candidate = parent / f"{base}_sglang_{i}"
        candidate = candidate.with_name(candidate.name + f"_{i}") 
        i += 1
    shutil.copytree(src, candidate)
    print(f"已复制目录: {src} -> {candidate}")
    return candidate

def parse_safetensors_header(path: Path) -> dict:
    """读取 safetensors 文件 header（不载入数据区），返回 dict"""
    with open(path, "rb") as f:
        first8 = f.read(8)
        if len(first8) != 8:
            raise ValueError("文件太短或不是 safetensors")
        header_len = struct.unpack("<Q", first8)[0]
        header_bytes = f.read(header_len)
        if len(header_bytes) != header_len:
            raise ValueError("无法读取完整 header")
        header = json.loads(header_bytes.decode("utf-8"))
    return header

def read_safetensors_databuffer(path: Path):
    """返回 header_len, header_bytes, data_buffer(bytes)"""
    with open(path, "rb") as f:
        first8 = f.read(8)
        if len(first8) != 8:
            raise ValueError("文件太短或不是 safetensors")
        header_len = struct.unpack("<Q", first8)[0]
        header_bytes = f.read(header_len)
        data_buffer = f.read()  # rest of file
    return header_len, header_bytes, data_buffer

def rename_key(key: str, replacements: list) -> str:
    """按规则对单个 key 做后缀替换（若没有匹配返回原 key）"""
    if len(replacements[0]) == 2:
        for old, new in replacements:
            if key.endswith(old):
                return key[: -len(old)] + new
    elif len(replacements[0]) == 3:
        for head, old, new in replacements:
            if key.startswith(head) and key.endswith(old):
                return key[: -len(old)] + new
    return key

def process_safetensors_file(path: Path, quant_type: str) -> bool:
    """
    在 place 替换 header keys 并写回文件（原地覆盖）。
    返回 True 表示成功修改，False 表示跳过或失败（文件保留原样）。
    """
    try:
        header_len, header_bytes, data_buffer = read_safetensors_databuffer(path)
        header = json.loads(header_bytes.decode("utf-8"))
    except Exception as e:
        print(f"[ERROR] 解析 header 失败: {path} : {e}")
        return False

    # 构建新的 header dict（保留 __metadata__ 不变）
    new_header = {}
    collision = False
    for k, v in header.items():
        if k == "__metadata__":
            new_header[k] = v
            continue
        new_k = rename_key(k, QUANT_METHOD[quant_type])
        if new_k in new_header:
            # 如果替换后产生重复 key（且不是同一 key），视为冲突
            print(f"[WARN] 替换导致重复 key，跳过文件: {path}")
            print(f"       原 key: {k} -> {new_k}，已有同名条目")
            collision = True
            break
        new_header[new_k] = v

    if collision:
        return False

    # 如果没有变化，就不写文件
    if new_header.keys() == header.keys():
        # 仍可能 keys 顺序不同，但如果确实没有任何 key 被替换则跳过
        changed = any(rename_key(k, QUANT_METHOD[quant_type]) != k for k in header.keys() if k != "__metadata__")
        if not changed:
            print(f"[SKIP] 文件中没有需要替换的 key: {path.name}")
            return False
        
    # 清理所有仍以 .quanted_layer 开头的 key（替换后剩余的）
    keys_to_remove = [k for k in new_header.keys() 
                    if k != "__metadata__" and "quanted_layer" in k]

    if not keys_to_remove:
        # 序列化新的 header 并写回文件（使用临时文件，成功后替换）
        try:
            new_header_bytes = json.dumps(new_header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            new_header_len = len(new_header_bytes)
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            with open(tmp_path, "wb") as out:
                out.write(struct.pack("<Q", new_header_len))
                out.write(new_header_bytes)
                out.write(data_buffer)  # data buffer 原样写入
            # 原子替换
            os.replace(tmp_path, path)
            print(f"[OK] 已修改 safetensors: {path.name}")
            return True
        except Exception as e:
            print(f"[ERROR] 写入新 safetensors 失败: {path} : {e}")
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass
            return False
    else:
        print(f"[INFO] 检测到 {len(keys_to_remove)} 个需要安全删除的 key: {keys_to_remove}")
        try:
            # 尝试导入 safetensors，如果失败则提示安装
            try:
                from safetensors import safe_open
                from safetensors.torch import save_file
            except ImportError:
                print("[ERROR] 需要安装 safetensors 库才能安全删除 key: pip install safetensors")
                return False
                
            # 1. 读取所有 tensor
            tensors = {}
            with safe_open(path, framework="pt") as f:
                for k in f.keys():
                    if k not in keys_to_remove:
                        tensors[k] = f.get_tensor(k)
                        
            # 2. 重新应用替换规则到所有 tensor keys
            renamed_tensors = {}
            for k, v in tensors.items():
                if k == "__metadata__":
                    renamed_tensors[k] = v
                    continue
                new_k = rename_key(k, QUANT_METHOD[quant_type])
                # 检查替换后是否有冲突
                if new_k in renamed_tensors and new_k != k:
                    print(f"[WARN] 替换导致重复 key，跳过文件: {path}")
                    print(f"原key: {k} -> {new_k}，已有同名条目")
                    return False
                renamed_tensors[new_k] = v
            
            # 3. 删除需要删除的key
            keys_to_remove_after_rename = [k for k in renamed_tensors.keys() 
                                         if k != "__metadata__" and ".quanted_layer" in k]
            for k in keys_to_remove_after_rename:
                print(f"[INFO] 删除多余 key: {k}")
                del renamed_tensors[k]
            
            # 保存新文件（自动处理偏移量和数据布局）
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            save_file(renamed_tensors, tmp_path)
            
            # 原子替换
            os.replace(tmp_path, path)
            
            print(f"[OK] 已安全删除 {len(keys_to_remove)} 个 key 并重新构建文件: {path.name}")
            return True
            
        except Exception as e:
            print(f"[ERROR] 安全删除 key 失败: {e}")
            # 清理临时文件
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass
            return False

def process_index_json(path: Path, quant_type: str) -> bool:
    """
    尝试处理 index.json（例如 model.safetensors.index.json），在里面替换 weight_map 的 keys。
    返回 True 表示有修改并已写入，False 表示未修改或失败。
    """
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[WARN] 读取 index.json 失败（跳过）: {path} : {e}")
        return False

    # 常见字段是 "weight_map"
    modified = False
    if isinstance(obj, dict) and "weight_map" in obj and isinstance(obj["weight_map"], dict):
        old_map = obj["weight_map"]
        new_map = {}
        for k, v in old_map.items():
            new_k = rename_key(k, QUANT_METHOD[quant_type])
            if new_k in new_map and new_k != k:
                print(f"[WARN] index.json 替换导致重复 key，跳过 index 文件: {path}")
                return False
            new_map[new_k] = v
            if new_k != k:
                modified = True
        # 清理 weight_map 中所有仍以 .quanted_layer 开头的 key
        keys_to_remove = [k for k in new_map.keys() 
                        if "quanted_layer" in k]
        for k in keys_to_remove:
            print(f"[INFO] 删除多余 weight_map key: {k}")
            del new_map[k]
            modified = True  # 确保标记为有修改
        if modified:
            obj["weight_map"] = new_map
            # 写回文件（备份原文件）
            bak = path.with_suffix(path.suffix + ".bak")
            try:
                path.rename(bak)
                path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"[OK] 已修改 index.json: {path.name}（备份: {bak.name}）")
                return True
            except Exception as e:
                print(f"[ERROR] 写入 index.json 失败: {path} : {e}")
                # 尝试还原备份
                if bak.exists():
                    bak.rename(path)
                return False
        else:
            # 没有修改
            return False
    else:
        # 也许存在其它 index json 风格，尝试寻找 "weight_map" 之外可能包含 tensor key 的字段会很不确定，
        # 因此这里只处理标准的 weight_map 风格。
        return False
    
def add_quant_config(cfg_path: str, quant_type: str, overwrite: bool = False, backup: bool = True):
    # 支持传模型目录或直接传config.json
    if os.path.isdir(cfg_path):
        cfg_path = os.path.join(cfg_path, "config.json")

    dirpath = os.path.dirname(cfg_path) or "."

    # 读取现有 config（若不存在则创建空 dict）
    data = {}
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"无法解析 JSON 文件 {cfg_path}: {e}")

    # 决策：是否写入/覆盖
    if "quantization_config" in data and not overwrite:
        print(f"{cfg_path} 已包含 'quantization_config'，且 overwrite=False，跳过写入。")
        return

    # 插入/覆盖
    if quant_type in QUANT_CONFIG:
        data["quantization_config"] = QUANT_CONFIG[quant_type]["quantization_config"]
        if "vision_config" in data and quant_type == "awq":
            data["quantization_config"]["modules_to_not_convert"] = ["visual"]
            
    # NOTE: 临时方案
    if "vllm_qwen" in quant_type:
        group_0 = {}
        group_0["format"] = "int-quantized"
        group_0["input_activations"] = {
            "actorder": None,
            "block_structure": None,
            "dynamic": False,
            "group_size": None,
            "num_bits": 8,
            "observer": "minmax",
            "observer_kwargs": {},
            "scale_dtype": None,
            "strategy": "tensor",
            "symmetric": True,
            "type": "int",
            "zp_dtype": None
        }
        group_0["output_activations"] = None
        group_0["weights"] = {
            "actorder": None,
            "block_structure": None,
            "dynamic": False,
            "group_size": None,
            "num_bits": 8,
            "observer": "minmax",
            "observer_kwargs": {},
            "scale_dtype": None,
            "strategy": "channel",
            "symmetric": True,
            "type": "int",
            "zp_dtype": None
        }
        group_0["targets"] = [
            "re:model.visual.blocks.*attn.qkv.*",
            "re:model.visual.blocks.*attn.proj.*",
            "re:model.visual.blocks.*mlp.linear_fc1.*",
            "re:model.visual.blocks.*mlp.linear_fc2.*",
            "re:visual.blocks.*attn.qkv.*",
            "re:visual.blocks.*attn.proj.*",
            "re:visual.blocks.*mlp.linear_fc1.*",
            "re:visual.blocks.*mlp.linear_fc2.*"
        ]
        
        vision_quant_config = {
            "config_groups": {
                "group_0": group_0
            },
            "format": "int-quantized",
            "global_compression_ratio": None,
            "ignore": [
                "model.visual.merger.linear_fc1",
                "model.visual.merger.linear_fc2",
                "model.visual.deepstack_merger_list.0.linear_fc1",
                "model.visual.deepstack_merger_list.0.linear_fc2",
                "model.visual.deepstack_merger_list.1.linear_fc1",
                "model.visual.deepstack_merger_list.1.linear_fc2",
                "model.visual.deepstack_merger_list.2.linear_fc1",
                "model.visual.deepstack_merger_list.2.linear_fc2",
                "lm_head"
            ],
            "kv_cache_scheme": None,
            "quant_method": "compressed-tensors",
            "quantization_status": "compressed",
            "sparsity_config": {},
            "transform_config": {},
            "version": "0.12.3.a20251114"
        }
        
        data["vision_config"]["quantization_config"] = vision_quant_config
        data["quantization_config"] = QUANT_CONFIG["awq_triton_w4a8"]["quantization_config"]
        
    # 备份原文件
    if backup and os.path.exists(cfg_path):
        bak_path = cfg_path + ".bak"
        shutil.copy2(cfg_path, bak_path)
        print(f"已备份原文件到: {bak_path}")

    # 原子写入：先写临时文件，再替换
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_config_", dir=dirpath, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmpf:
            json.dump(data, tmpf, ensure_ascii=False, indent=2)
            tmpf.write("\n")
        os.replace(tmp_path, cfg_path)
    finally:
        # 若出错确保临时文件不留
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    print(f"已将 'quantization_config' 写入: {cfg_path}")

def main():
    parser = argparse.ArgumentParser(description="Copy folder and rename quanted safetensors keys.")
    # parser.add_argument("folder", help="原始文件夹路径（必传）")
    parser.add_argument("--src", required=True, help="原始文件夹路径（必传）")
    parser.add_argument("--dst", help="目标文件夹路径（可选）")
    parser.add_argument("--quant_type", required=True, help="量化格式")
    args = parser.parse_args()
    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve() if args.dst else None
    quant_type = args.quant_type
    if quant_type not in QUANT_METHOD:
        print(f"不支持的量化类型: {quant_type}")
        sys.exit(1)
    if not src.exists() or not src.is_dir():
        print(f"路径不存在或不是目录: {src}")
        sys.exit(1)

    dest = make_unique_copydir(src, dst)

    # 遍历新目录，处理 .safetensors 和 *.index.json
    safetensors_files = list(dest.rglob("*.safetensors"))
    index_json_files = list(dest.rglob("*.index.json")) + list(dest.rglob("*.safetensors.index.json"))

    print(f"发现 {len(safetensors_files)} 个 .safetensors 文件，{len(index_json_files)} 个 index.json 文件，开始处理...")

    changed_count = 0
    for f in safetensors_files:
        try:
            if process_safetensors_file(f, quant_type):
                changed_count += 1
        except Exception as e:
            print(f"[ERROR] 处理文件异常: {f} : {e}")

    index_changed = 0
    for j in index_json_files:
        try:
            if process_index_json(j, quant_type):
                index_changed += 1
        except Exception as e:
            print(f"[WARN] 处理 index.json 出错: {j} : {e}")
            
    try:
        add_quant_config(str(dst), quant_type)
    except Exception as e:
        print(f"[ERROR] 处理量化config失败: {e}")

    print(f"处理完成。safetensors 修改数量: {changed_count}, index.json 修改数量: {index_changed}")
    print(f"新目录在: {dest}")

if __name__ == "__main__":
    main()

    