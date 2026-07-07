"""PyTorch / HuggingFace → ONNX"""
import os
import tempfile
import torch
import onnx


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
    if fp16:
        model = model.half()
        print("[INFO] 模型转 FP16")

    # dummy inputs
    if input_shape:
        dummy_inputs = {}
        for name, shape in input_shape.items():
            is_int = 'input_ids' in name or 'mask' in name or 'token' in name
            dtype = torch.int64 if is_int else (torch.float16 if fp16 else torch.float32)
            dummy_inputs[name] = (torch.randint(0, 1000, shape, dtype=dtype) if is_int
                                  else torch.randn(shape, dtype=dtype))
    else:
        dummy_inputs = tokenizer("Hello, this is a test", return_tensors="pt")

    print(f"[INFO] Dummy inputs: {list(dummy_inputs.keys())}")

    if dynamic_axes is None:
        dynamic_axes = {name: {0: "batch_size"} for name in dummy_inputs}

    model_name = os.path.basename(model_path.rstrip('/')) or "model"
    onnx_path = os.path.join(output_dir, f"{model_name}.onnx")

    # 导出到临时文件 → 外部数据格式
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".onnx")
    os.close(tmp_fd)
    try:
        print(f"[INFO] 转换中 → {onnx_path}")
        torch.onnx.export(
            model,
            tuple(dummy_inputs.values()) if len(dummy_inputs) > 1
            else list(dummy_inputs.values())[0],
            tmp_path,
            input_names=list(dummy_inputs.keys()),
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=False,  # 关掉大幅提速，大模型减少 50%+ 时间
        )
        m = onnx.load(tmp_path)
        onnx.save(m, onnx_path, save_as_external_data=True,
                  all_tensors_to_one_file=True,
                  location=f"{model_name}.weight")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    total = os.path.getsize(onnx_path)
    for f in os.listdir(output_dir):
        total += os.path.getsize(os.path.join(output_dir, f))
    print(f"[OK] ONNX: {onnx_path} (总计 ~{total/(1024*1024):.1f}MB)")
    return onnx_path
