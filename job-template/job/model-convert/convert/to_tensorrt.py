"""ONNX → TensorRT (.plan)"""
import os


def convert_to_tensorrt(onnx_path: str, output_dir: str,
                        fp16: bool = True, workspace: int = 4096):
    """
    ONNX 模型 → TensorRT engine (.plan)
    使用 Python tensorrt API
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] ONNX 输入: {onnx_path}")
    print(f"[INFO] 配置: fp16={fp16}, workspace={workspace}MB")

    try:
        import tensorrt as trt
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

        builder = trt.Builder(TRT_LOGGER)
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)
        parser = trt.OnnxParser(network, TRT_LOGGER)

        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX 文件不存在: {onnx_path}")

        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(f"[ERROR] ONNX Parse: {parser.get_error(i)}")
                raise RuntimeError("ONNX 解析失败")

        print(f"[INFO] ONNX 解析成功, 网络层数: {network.num_layers}")

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace * 1024 * 1024)

        if fp16:
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                print("[INFO] 启用 FP16")
            else:
                print("[WARN] 平台不支持 FP16, 回退到 FP32")

        print("[INFO] 构建 TensorRT engine (可能需要几分钟)...")
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError("TensorRT engine 构建失败")

        base = os.path.splitext(os.path.basename(onnx_path))[0]
        plan_path = os.path.join(output_dir, f"{base}.plan")
        with open(plan_path, "wb") as f:
            f.write(serialized_engine)

        size_mb = os.path.getsize(plan_path) / (1024 * 1024)
        print(f"[OK] TensorRT engine: {plan_path} ({size_mb:.1f}MB)")
        return plan_path

    except ImportError:
        print("[ERROR] tensorrt Python 包未安装, 请 pip install tensorrt")
        raise
