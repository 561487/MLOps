"""PyTorch / HuggingFace → ONNX（使用 optimum-cli，比 torch.onnx.export 快 3-5x）"""
import os, sys, subprocess


def convert_to_onnx(model_path: str, output_dir: str, input_shape: dict,
                    opset: int = 17, fp16: bool = False,
                    dynamic_axes: dict = None):
    """
    HuggingFace 模型 → ONNX
    model_path: 本地路径或 HF model id
    output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    model_name = os.path.basename(model_path.rstrip('/')) or "model"
    onnx_path = os.path.join(output_dir, f"{model_name}.onnx")

    print(f"[INFO] 模型: {model_path}")
    print(f"[INFO] 输出: {onnx_path}")
    print(f"[INFO] FP16: {fp16}  Opset: {opset}")

    # optimum-cli 自动处理 DynamicCache、external data、dummy inputs
    cmd = [
        sys.executable, "-m", "optimum.exporters.onnx",
        "--model", model_path,
        output_dir,
        "--task", "text-generation",
        "--opset", str(opset),
    ]
    if fp16:
        cmd.append("--fp16")

    print(f"[INFO] 执行: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

    if result.returncode == 0:
        print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)

        # optimum 输出文件名是固定的 model.onnx，重命名为模型名
        default_onnx = os.path.join(output_dir, "model.onnx")
        if os.path.exists(default_onnx) and default_onnx != onnx_path:
            os.rename(default_onnx, onnx_path)
            # 同时重命名外部数据文件
            default_weight = os.path.join(output_dir, "model.onnx_data")
            if os.path.exists(default_weight):
                os.rename(default_weight, os.path.join(output_dir, f"{model_name}.onnx_data"))

        # 打印大小
        total = os.path.getsize(onnx_path) if os.path.exists(onnx_path) else 0
        for f in sorted(os.listdir(output_dir)):
            fp = os.path.join(output_dir, f)
            if os.path.isfile(fp):
                sz = os.path.getsize(fp)
                total += sz
                print(f"  {f} ({sz/(1024*1024):.1f}MB)" if sz > 1024*1024 else f"  {f} ({sz} bytes)")
        print(f"[OK] ONNX 模型: {onnx_path}")
        return onnx_path
    else:
        stderr = result.stderr
        print(f"[ERROR] optimum-cli 失败:\n{stderr[-2000:]}")
        sys.exit(1)
