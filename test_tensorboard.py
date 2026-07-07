#!/usr/bin/env python3
"""TensorBoard 验证（无需 PyTorch/TensorFlow，用内置 tf.compat 生成 event 文件）"""
import os, sys

print("=" * 60)
print("TensorBoard 验证")
print("=" * 60)

print("\n[1/3] 检测环境...")
try:
    import tensorboard
    print(f"  ✅ tensorboard {tensorboard.__version__}")
except ImportError:
    print("  ❌ tensorboard 未安装")
    sys.exit(1)

logdir = "/tmp/tb_test"
os.makedirs(logdir, exist_ok=True)

print(f"\n[2/3] 写入测试数据到 {logdir} ...")

# 直接用 tensorboard 内置的 tf.compat 写 event 文件（不依赖 tensorflow）
from tensorboard.compat.proto import event_pb2, summary_pb2
from tensorboard.compat.tensorflow_stub.pywrap_tensorflow import masked_crc32c
import time, struct

path = os.path.join(logdir, "events.out.tfevents.test")
with open(path, "wb") as f:
    for i in range(20):
        s = summary_pb2.Summary(value=[
            summary_pb2.Summary.Value(tag="train/loss", simple_value=2.0 / (i + 1) + 0.1),
            summary_pb2.Summary.Value(tag="train/accuracy", simple_value=i / 25.0),
        ])
        ev = event_pb2.Event(wall_time=time.time(), step=i, summary=s)
        body = ev.SerializeToString()
        hdr = struct.pack("Q", len(body))
        crc = struct.pack("I", masked_crc32c(hdr + body))
        f.write(hdr + crc + body)

print(f"  ✅ {path} ({os.path.getsize(path)} bytes)")

print(f"\n[3/3] 启动:")
print(f"  tensorboard --logdir={logdir} --port=6006 --host 0.0.0.0")
print("=" * 60)
