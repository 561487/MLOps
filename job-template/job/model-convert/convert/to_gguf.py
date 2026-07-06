"""HuggingFace → GGUF (llama.cpp / ollama 格式)"""
import os, glob, json


def convert_to_gguf(model_path: str, output_dir: str,
                    out_type: str = "f16", model_name: str = None):
    """
    HuggingFace 模型 → GGUF
    model_path: 本地 HF 模型目录
    output_dir: 输出目录
    out_type: f16 / q8_0 / q4_k_m / q4_0
    """
    os.makedirs(output_dir, exist_ok=True)

    if model_name is None:
        model_name = os.path.basename(model_path.rstrip('/')) or "model"

    print(f"[INFO] 加载模型: {model_path}")
    print(f"[INFO] 量化类型: {out_type}")

    # 检查是否有 safetensors 文件
    safetensor_files = glob.glob(os.path.join(model_path, "*.safetensors"))
    if safetensor_files:
        print(f"[INFO] 找到 {len(safetensor_files)} 个 safetensors 文件")

    # 方式1: 用 llama.cpp 的 convert_hf_to_gguf.py（最可靠）
    try:
        import subprocess, sys
        # 查找 convert 脚本
        convert_script = None
        search_paths = [
            "/usr/local/bin/convert_hf_to_gguf.py",
            "/usr/local/lib/python3.10/site-packages/llama_cpp/convert.py",
            os.path.expanduser("~/.local/bin/convert_hf_to_gguf.py"),
        ]
        for sp in search_paths:
            if os.path.exists(sp):
                convert_script = sp
                break

        if convert_script:
            out_file = os.path.join(output_dir, f"{model_name}-{out_type}.gguf")
            cmd = [
                sys.executable, convert_script,
                model_path,
                "--outfile", out_file,
                "--outtype", out_type,
            ]
            print(f"[INFO] 执行: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(result.stdout)
                print(f"[OK] GGUF: {out_file}")
                return out_file
            else:
                print(f"[WARN] convert_hf_to_gguf 失败: {result.stderr[:500]}")
        else:
            print("[WARN] 未找到 convert_hf_to_gguf.py, 使用 Python API")
    except Exception as e:
        print(f"[WARN] 外部脚本方式失败: {e}")

    # 方式2: 用 gguf Python 库直接写
    print("[INFO] 使用 gguf Python 库转换...")
    from gguf import GGUFWriter, GGMLQuantizationType, TensorInfo
    import torch

    # 读取 HF config
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {}

    # 量化类型映射
    quant_map = {
        "f16": GGMLQuantizationType.F16,
        "f32": GGMLQuantizationType.F32,
        "q8_0": GGMLQuantizationType.Q8_0,
        "q4_0": GGMLQuantizationType.Q4_0,
        "q4_k_m": GGMLQuantizationType.Q4_K_M,
    }
    quant_type = quant_map.get(out_type, GGMLQuantizationType.F16)

    out_file = os.path.join(output_dir, f"{model_name}-{out_type}.gguf")

    # 用 safetensors 加载权重
    from safetensors.torch import load_file as load_safetensors

    # 提取 vocab_size 等元信息
    vocab_size = config.get("vocab_size", 32000)
    hidden_size = config.get("hidden_size", config.get("d_model", 4096))
    num_layers = config.get("num_hidden_layers", config.get("n_layer", 32))
    num_heads = config.get("num_attention_heads", config.get("n_head", 32))

    writer = GGUFWriter(out_file, model_name)
    writer.add_uint32("vocab_size", vocab_size)
    writer.add_uint32("hidden_size", hidden_size)
    writer.add_uint32("num_layers", num_layers)
    writer.add_uint32("num_heads", num_heads)

    # 写每层权重
    weight_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not weight_files:
        weight_files = sorted(glob.glob(os.path.join(model_path, "pytorch_model*.bin")))

    for wf in weight_files:
        print(f"  处理: {os.path.basename(wf)}")
        try:
            tensors = load_safetensors(wf)
        except Exception:
            tensors = torch.load(wf, map_location="cpu")

        for name, tensor in tensors.items():
            info = TensorInfo(
                name=name,
                tensor=tensor,
                quantization_type=quant_type,
            )
            writer.add_tensor_info(info)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()

    size_mb = os.path.getsize(out_file) / (1024 * 1024)
    print(f"[OK] GGUF: {out_file} ({size_mb:.1f}MB)")
    return out_file
