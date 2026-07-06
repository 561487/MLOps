"""PyTorch / HuggingFace → ONNX"""
import os, json
import torch
import onnx


def _detect_inputs(model, sample_inputs):
    """如果用户没给 input_shape，从模型自动推断一批 sample inputs."""
    if sample_inputs:
        return sample_inputs
    # HF 模型自动生成 dummy input
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model.config._name_or_path if hasattr(model.config, '_name_or_path') else '.')
        # 尝试用 optimum 生成
        from optimum.exporters.onnx import OnnxConfig
        from optimum.utils import DummyInputGenerator
    except Exception:
        pass
    # 最后兜底：用 model 的 forward 参数签名推断
    return None


def convert_to_onnx(model_path: str, output_dir: str, input_shape: dict,
                    opset: int = 17, fp16: bool = False,
                    dynamic_axes: dict = None):
    """
    PyTorch / HuggingFace 模型 → ONNX
    model_path: 本地路径或 HF model id
    output_dir: 输出目录
    input_shape: {"input_ids": [1,512], "attention_mask": [1,512]}
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] 加载模型: {model_path}")
    from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    # 先试 HF 加载
    try:
        if os.path.isdir(model_path) or model_path.startswith('/'):
            model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.float32, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.float32, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print(f"[INFO] 检测到 HuggingFace CausalLM 模型")
    except Exception as e:
        print(f"[WARN] HF CausalLM 加载失败: {e}")
        raise

    model.eval()
    # 关掉 KV cache，否则 trace 时返回 DynamicCache 对象导致 JIT 报错
    if hasattr(model, 'config') and hasattr(model.config, 'use_cache'):
        model.config.use_cache = False

    # 生成 dummy inputs
    if input_shape:
        dummy_inputs = {}
        for name, shape in input_shape.items():
            dtype = torch.int64 if 'input_ids' in name or 'mask' in name or 'token' in name else torch.float32
            dummy_inputs[name] = torch.randint(0, 1000, shape, dtype=dtype) if dtype == torch.int64 else torch.randn(shape)
    else:
        # 用 tokenizer 生成 dummy
        dummy_inputs = tokenizer("Hello, this is a test", return_tensors="pt")

    print(f"[INFO] Dummy inputs: {list(dummy_inputs.keys())}")
    for k, v in dummy_inputs.items():
        print(f"  {k}: shape={list(v.shape)} dtype={v.dtype}")

    # 设置 dynamic_axes
    if dynamic_axes is None:
        dynamic_axes = {}
        for name in dummy_inputs:
            dynamic_axes[name] = {0: "batch_size"}
        if 'logits' in dir(model):
            dynamic_axes["logits"] = {0: "batch_size"}

    # 模型文件名
    model_name = os.path.basename(model_path.rstrip('/')) or "model"
    onnx_path = os.path.join(output_dir, f"{model_name}.onnx")

    print(f"[INFO] 开始转换 → {onnx_path}")
    torch.onnx.export(
        model,
        tuple(dummy_inputs.values()) if len(dummy_inputs) > 1 else list(dummy_inputs.values())[0],
        onnx_path,
        input_names=list(dummy_inputs.keys()),
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
    )

    # 验证
    _verify_onnx(onnx_path)
    print(f"[OK] ONNX 模型: {onnx_path}")
    return onnx_path


def _verify_onnx(onnx_path: str):
    """验证 ONNX 模型可读且合法."""
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"[INFO] ONNX 验证通过, 大小: {size_mb:.1f}MB")
