"""PyTorch → TorchScript"""
import os, torch


def convert_to_torchscript(model_path: str, output_dir: str, input_shape: dict = None):
    """
    PyTorch 模型 → TorchScript (.pt)
    model_path: 本地 HF 模型路径
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] 加载模型: {model_path}")
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32, trust_remote_code=True)
    model.eval()
    if hasattr(model, 'config') and hasattr(model.config, 'use_cache'):
        model.config.use_cache = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"[INFO] 设备: {device}")

    model_name = os.path.basename(model_path.rstrip('/')) or "model"
    out_path = os.path.join(output_dir, f"{model_name}.pt")

    print(f"[INFO] 开始 TorchScript trace → {out_path}")
    try:
        shape = input_shape.get("input_ids", [1, 32]) if input_shape else [1, 32]
        dummy = torch.randint(0, 1000, shape, dtype=torch.long).to(device)
        traced = torch.jit.trace(model, dummy, strict=False)
        torch.jit.save(traced, out_path)
        print(f"[OK] TorchScript: {out_path}")
    except Exception as e1:
        print(f"[WARN] trace 失败: {e1}")
        try:
            scripted = torch.jit.script(model)
            torch.jit.save(scripted, out_path)
            print(f"[OK] TorchScript(script): {out_path}")
        except Exception as e2:
            print(f"[ERROR] TorchScript 不兼容此模型: {e2}")
            print("[HINT] 新模型(Qwen3/Llama3+)不支持 TorchScript，建议用 ONNX")
            raise

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"[INFO] TorchScript 大小: {size_mb:.1f}MB")
    return out_path
