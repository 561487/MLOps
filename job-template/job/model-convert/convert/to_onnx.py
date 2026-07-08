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
    # GPU 可用则移到 GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"[INFO] 设备: {device}")
    if fp16:
        model = model.half()
        print("[INFO] 模型转 FP16")

    model_name = os.path.basename(model_path.rstrip('/')) or "model"
    onnx_path = os.path.join(output_dir, f"{model_name}.onnx")

    # dummy inputs — 统一转成 dict 格式
    if input_shape:
        dummy = {}
        for name, shape in input_shape.items():
            is_int = 'input_ids' in name or 'mask' in name or 'token' in name
            dtype = torch.int64 if is_int else (torch.float16 if fp16 else torch.float32)
            t = (torch.randint(0, 1000, shape, dtype=dtype) if is_int
                 else torch.randn(shape, dtype=dtype))
            dummy[name] = t.to(device)
    else:
        tok_out = tokenizer("Hello, this is a test", return_tensors="pt")
        dummy = {k: v.to(device) for k, v in tok_out.items()}

    # 优先用 dynamo_export（torch 2.x, 比 JIT trace 快 3-10x）
    try:
        print(f"[INFO] 使用 dynamo_export...")
        from torch.export import Dim
        if isinstance(dummy, dict):
            batch_dim = {name: Dim("batch", min=1, max=32) for name in dummy}
            exported = torch.onnx.dynamo_export(
                model, **dummy, export_options=torch.onnx.ExportOptions(
                    dynamic_shapes=batch_dim))
        else:
            exported = torch.onnx.dynamo_export(model, dummy)
        exported.save(onnx_path)
        print(f"[OK] dynamo_export 完成: {onnx_path}")
    except Exception as e:
        print(f"[WARN] dynamo_export 失败: {e}, 回退 JIT trace")
        export_jit(model, dummy, onnx_path, opset, model_name, output_dir)

    size = os.path.getsize(onnx_path)
    print(f"[OK] ONNX: {onnx_path} ({size/(1024*1024):.1f}MB)")
    return onnx_path


def export_jit(model, dummy, onnx_path, opset, model_name, output_dir):
    """回退方案: JIT trace + 外部数据格式"""
    import tempfile, onnx

    if isinstance(dummy, dict):
        inputs = tuple(dummy.values())
        input_names = list(dummy.keys())
    else:
        inputs = dummy
        input_names = ["input_ids", "attention_mask"]

    dynamic_axes = {name: {0: "batch_size"} for name in input_names}

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".onnx")
    os.close(tmp_fd)
    try:
        torch.onnx.export(
            model, inputs, tmp_path,
            input_names=input_names,
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=False,
        )
        m = onnx.load(tmp_path)
        onnx.save(m, onnx_path, save_as_external_data=True,
                  all_tensors_to_one_file=True,
                  location=f"{model_name}.weight")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
