#!/usr/bin/env python3
"""Cube Studio Job Template: model-convert
模型格式转换入口：PyTorch/HF → ONNX → TensorRT / TorchScript
"""
import argparse, os, sys, json

# 平台环境变量
KFJ_CREATOR = os.getenv("KFJ_CREATOR", "admin")
KFJ_RUN_ID = os.getenv("KFJ_RUN_ID", "")
KFJ_PIPELINE_ID = os.getenv("KFJ_PIPELINE_ID", "0")


def detect_src_format(model_path: str) -> str:
    """从文件后缀自动检测源格式."""
    if not os.path.exists(model_path) and not model_path.startswith('/'):
        return "huggingface"  # HF model id

    if os.path.isdir(model_path):
        if os.path.exists(os.path.join(model_path, "config.json")):
            return "huggingface"
        if os.path.exists(os.path.join(model_path, "saved_model.pb")):
            return "tensorflow"
        return "huggingface"  # 含权重的目录，当作 HF

    # 单文件 → 找同目录的 config.json
    parent_dir = os.path.dirname(model_path)
    if parent_dir and os.path.exists(os.path.join(parent_dir, "config.json")):
        return "huggingface"

    ext = os.path.splitext(model_path)[1].lower()
    return {"safetensors": "pytorch", ".bin": "pytorch",
            ".pt": "pytorch", ".pth": "pytorch", ".onnx": "onnx"}.get(ext, "pytorch")


def main():
    parser = argparse.ArgumentParser(description="模型格式转换")
    parser.add_argument("--model_path", type=str, required=True,
                        help="输入模型路径 (本地目录/文件 或 HF model id)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="输出目录")
    parser.add_argument("--dst_format", type=str, default="onnx",
                        choices=["onnx", "tensorrt", "torchscript", "gguf"],
                        help="目标格式")
    parser.add_argument("--out_type", type=str, default="f16",
                        choices=["f16", "f32", "q8_0", "q4_0", "q4_k_m"],
                        help="GGUF 量化类型 (仅 gguf)")
    parser.add_argument("--input_shape", type=str, default="",
                        help='输入形状 JSON, 如 {"input_ids":[1,512]}')
    parser.add_argument("--opset", type=int, default=17,
                        help="ONNX opset 版本")
    parser.add_argument("--fp16", type=str, default="true",
                        help="是否使用 FP16")
    parser.add_argument("--dynamic_axes", type=str, default="",
                        help='动态轴 JSON, 如 {"input_ids":{"0":"batch"}}')

    args = parser.parse_args()
    fp16 = args.fp16.lower() in ("true", "1", "yes")
    input_shape = json.loads(args.input_shape) if args.input_shape else None
    dynamic_axes = json.loads(args.dynamic_axes) if args.dynamic_axes else None

    print("=" * 60)
    print("模型格式转换")
    print("=" * 60)
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print(f"  KFJ_CREATOR: {KFJ_CREATOR}")
    print(f"  KFJ_RUN_ID: {KFJ_RUN_ID}")
    print("=" * 60)

    os.makedirs(args.output_path, exist_ok=True)
    src = detect_src_format(args.model_path)
    print(f"\n[INFO] 检测到源格式: {src}")
    print(f"[INFO] 目标格式: {args.dst_format}")

    # 路由
    if args.dst_format == "onnx":
        assert src in ("pytorch", "huggingface"), \
            f"不支持 {src} → onnx 转换"
        from convert.to_onnx import convert_to_onnx
        convert_to_onnx(args.model_path, args.output_path,
                        input_shape, args.opset, fp16, dynamic_axes)

    elif args.dst_format == "torchscript":
        assert src in ("pytorch", "huggingface"), \
            f"不支持 {src} → torchscript 转换"
        from convert.to_torchscript import convert_to_torchscript
        convert_to_torchscript(args.model_path, args.output_path, input_shape)

    elif args.dst_format == "tensorrt":
        # 需要 ONNX 中间产物
        if src in ("pytorch", "huggingface"):
            print("[INFO] 先转换 PyTorch → ONNX ...")
            from convert.to_onnx import convert_to_onnx
            onnx_path = convert_to_onnx(args.model_path, args.output_path,
                                        input_shape, args.opset, fp16, dynamic_axes)
        elif src == "onnx":
            onnx_path = args.model_path
        else:
            raise ValueError(f"不支持 {src} → tensorrt 转换")
        from convert.to_tensorrt import convert_to_tensorrt
        convert_to_tensorrt(onnx_path, args.output_path, fp16)

    elif args.dst_format == "gguf":
        assert src in ("pytorch", "huggingface"), \
            f"GGUF 转换需要 PyTorch/HF 源模型, 当前: {src}"
        from convert.to_gguf import convert_to_gguf
        convert_to_gguf(args.model_path, args.output_path, args.out_type)

    print(f"\n[OK] 转换完成! 输出目录: {args.output_path}")
    for f in sorted(os.listdir(args.output_path)):
        fp = os.path.join(args.output_path, f)
        print(f"  {f} ({os.path.getsize(fp)} bytes)")


if __name__ == "__main__":
    main()
