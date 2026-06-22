import os
import sys
import subprocess
import json

# ============================================================
# llama_factory 任务模板启动器
# 功能：
#   1. 接收流水线表单传入的参数（CLI args）
#   2. 设置环境变量
#   3. 调用 start.py 并传递所有参数
# ============================================================

# launcher.py 所在目录（Docker 中为 /app/），start.py 也在此目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

KFJ_CREATOR = os.getenv('KFJ_CREATOR', 'admin')
KFJ_PIPELINE_NAME = os.getenv('KFJ_PIPELINE_NAME', '')
KFJ_TASK_NAME = os.getenv('KFJ_TASK_NAME', '')


def main():
    workdir = os.getenv('KFJ_TASK_WORKDIR', SCRIPT_DIR)

    # 切换到工作目录
    if not os.path.exists(workdir):
        print(f'[launcher.py] 工作目录 {workdir} 不存在，创建中...')
        os.makedirs(workdir, exist_ok=True)
    os.chdir(workdir)
    print(f'[launcher.py] 工作目录: {os.getcwd()}')

    # 解析并设置环境变量（KFJ_TASK_ENV 格式 KEY=VALUE，多个用换行分隔）
    env_vars = os.getenv('KFJ_TASK_ENV', '')
    if env_vars:
        for line in env_vars.replace(',', '\n').split('\n'):
            line = line.strip()
            if line and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()

    # 收集所有 CLI 参数（表单参数和未知参数）
    # sys.argv[0] 是 launcher.py，sys.argv[1:] 是平台传进来的表单参数
    args = sys.argv[1:] if len(sys.argv) > 1 else []
    print(f'[launcher.py] 接收到的表单参数: {" ".join(args)}')

    # 调用 start.py（使用绝对路径，不受 workdir 影响）并传递所有参数
    start_py = os.path.join(SCRIPT_DIR, 'start.py')
    cmd = ['python', start_py] + args
    print(f'[launcher.py] 执行: {" ".join(cmd)}')

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in process.stdout:
        print(line, end='', flush=True)
    exit_code = process.wait()
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
