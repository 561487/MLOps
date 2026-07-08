"""HuggingFace → GGUF (llama.cpp / ollama 格式)"""
import os, glob, sys


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

    out_file = os.path.join(output_dir, f"{model_name}-{out_type}.gguf")

    # 方式1: 用 llama.cpp 的 convert_hf_to_gguf.py（最可靠）
    import subprocess
    convert_script = None
    search_paths = [
        "/usr/local/bin/convert_hf_to_gguf.py",
        os.path.expanduser("~/.local/bin/convert_hf_to_gguf.py"),
    ]
    for sp in search_paths:
        if os.path.exists(sp):
            convert_script = sp
            break

    if convert_script:
        cmd = [sys.executable, convert_script, model_path,
               "--outfile", out_file, "--outtype", out_type]
        print(f"[INFO] 执行: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout[-2000:])
            print(f"[OK] GGUF: {out_file}")
            return out_file
        else:
            print(f"[WARN] convert 脚本失败: {result.stderr[:1000]}")

    # 方式2: 用 gguf Python 库
    print("[INFO] 使用 gguf Python 库...")
    try:
        from gguf.gguf_writer import GGUFWriter
        from gguf.constants import GGMLQuantizationType, Keys
        import torch

        writer = GGUFWriter(out_file, model_name)

        # 写 safetensors 权重
        from safetensors.torch import load_file as load_safetensors

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
                writer.add_tensor(name, tensor.numpy())

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

        size_mb = os.path.getsize(out_file) / (1024 * 1024)
        print(f"[OK] GGUF: {out_file} ({size_mb:.1f}MB)")
        return out_file

    except ImportError as e:
        print(f"[ERROR] gguf 库不可用: {e}")
        print("[HINT] pip install gguf 或安装 llama.cpp 的 convert_hf_to_gguf.py")
        sys.exit(1)
