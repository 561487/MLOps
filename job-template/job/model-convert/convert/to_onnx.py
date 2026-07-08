"""PyTorch / HuggingFace → ONNX"""
import os
import torch


def convert_to_onnx(model_path: str, output_dir: str, input_shape: dict,
                    opset: int = 17, fp16: bool = False,
                    dynamic_axes: dict = None):
    """HuggingFace 模型 → ONNX"""
    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] 加载模型: {model_path}")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    if hasattr(model, 'config') and hasattr(model.config, 'use_cache'):
        model.config.use_cache = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"[INFO] 设备: {device}")

    if fp16:
        model = model.half()
        print("[INFO] FP16")

    model_name = os.path.basename(model_path.rstrip('/')) or "model"
    onnx_path = os.path.join(output_dir, f"{model_name}.onnx")

    # dummy inputs
    if input_shape:
        dummy = {}
        for name, shape in input_shape.items():
            is_int = 'input_ids' in name or 'mask' in name or 'token' in name
            dtype = torch.int64 if is_int else (torch.float16 if fp16 else torch.float32)
            dummy[name] = torch.randint(0, 1000, shape, dtype=dtype).to(device) if is_int \
                else torch.randn(shape, dtype=dtype).to(device)
        input_names = list(dummy.keys())
        inputs = tuple(dummy.values())
    else:
        tok = tokenizer("Hello, this is a test", return_tensors="pt")
        inputs = tuple(v.to(device) for v in tok.values())
        input_names = list(tok.keys())

    # JIT trace + 外部数据（do_constant_folding=False 大幅提速）
    import tempfile, onnx
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".onnx")
    os.close(tmp_fd)
    try:
        print(f"[INFO] ONNX 导出中...")
        torch.onnx.export(
            model, inputs, tmp_path,
            input_names=input_names, output_names=["logits"],
            dynamic_axes={n: {0: "batch_size"} for n in input_names},
            opset_version=opset, do_constant_folding=False,
        )
        m = onnx.load(tmp_path)
        onnx.save(m, onnx_path, save_as_external_data=True,
                  all_tensors_to_one_file=True, location=f"{model_name}.weight")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    print(f"[OK] ONNX: {onnx_path} ({os.path.getsize(onnx_path)/(1024*1024):.1f}MB)")
    return onnx_path
