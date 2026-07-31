#!/usr/bin/env python3
"""
litgpt-pretrain 任务模板启动器
================================

Workflow:
  1. 读取平台环境变量（KFJ_*）
  2. 解析 KFJ_TASK_ENV 中的自定义环境变量
  3. 收集所有 CLI args，原样传递给 start.py
  4. 执行 start.py 并实时转发输出
"""

import os
import sys
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    workdir = os.getenv("KFJ_TASK_WORKDIR", SCRIPT_DIR)

    # 切换到工作目录
    if not os.path.exists(workdir):
        print(f"[launcher.py] workdir {workdir} does not exist, creating...")
        os.makedirs(workdir, exist_ok=True)
    os.chdir(workdir)
    print(f"[launcher.py] workdir: {os.getcwd()}")

    # 平台自定义环境变量（KFJ_TASK_ENV 格式 KEY=VALUE，多个用换行分隔）
    env_vars = os.getenv("KFJ_TASK_ENV", "")
    if env_vars:
        for line in env_vars.split("\n"):
            line = line.strip()
            if line and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip()

    # 打印环境信息
    for key in ["KFJ_CREATOR", "KFJ_PIPELINE_NAME", "KFJ_TASK_NAME",
                "HF_ENDPOINT", "CUDA_VISIBLE_DEVICES"]:
        print(f"[launcher.py] {key}: {os.environ.get(key, '(not set)')}")

    # 收集所有 CLI 参数
    args = sys.argv[1:] if len(sys.argv) > 1 else []
    print(f"[launcher.py] received args: {' '.join(args)}")

    # 调用 start.py 并传递参数
    start_py = os.path.join(SCRIPT_DIR, "start.py")
    cmd = ["python3", start_py] + args
    print(f"[launcher.py] executing: {' '.join(cmd)}")

    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    for line in process.stdout:
        print(line, end="", flush=True)
    exit_code = process.wait()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
