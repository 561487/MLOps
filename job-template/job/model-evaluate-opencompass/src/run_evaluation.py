#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cube Studio Job Template: model-evaluate-opencompass

基于 OpenCompass 的 LLM 模型评测入口脚本。
被容器 ENTRYPOINT 调用，接收用户在 pipeline 编辑器中配置的参数。

参数通过 argparse 从命令行传入（Cube Studio 平台传递机制）。
"""

import argparse
import json
import os
import subprocess
import sys

# ---- 平台注入的环境变量 ----
KFJ_CREATOR = os.getenv('KFJ_CREATOR', 'admin')
KFJ_RUN_ID = os.getenv('KFJ_RUN_ID', '')
KFJ_PIPELINE_ID = os.getenv('KFJ_PIPELINE_ID', '0')
KFJ_TASK_PROJECT_NAME = os.getenv('KFJ_TASK_PROJECT_NAME', 'public')


def check_opencompass():
    """检查 OpenCompass CLI 是否可用。"""
    try:
        result = subprocess.run(
            ['opencompass', '--help'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            print(f"[ERROR] opencompass CLI 不可用: {result.stderr}")
            sys.exit(1)
        print(f"[INFO] opencompass CLI 可用")
    except FileNotFoundError:
        print("[ERROR] opencompass 命令未找到，请确认镜像中已安装 opencompass")
        sys.exit(1)


def build_opencompass_cmd(args: argparse.Namespace) -> list:
    """根据用户参数构建 OpenCompass 命令行。
    
    OpenCompass CLI 用法（pip 安装后）:
        opencompass --models <model_cfg> --datasets <dataset_cfg>
    
    或直接使用 run.py（可用 python -c 找到安装路径）:
        python -c "import opencompass; import os; print(os.path.dirname(opencompass.__file__))"
    
    这里优先使用 opencompass CLI，若缺失则尝试 python run.py。
    """
    work_dir = os.path.join(args.output_path, 'opencompass_results')
    os.makedirs(work_dir, exist_ok=True)

    cmd = [
        'opencompass',
        '--datasets', args.datasets,
        '--hf-path', args.model_path,
        '--hf-num-gpus', str(args.num_gpus),
        '--batch-size', str(args.batch_size),
        '--max-seq-len', str(args.max_seq_len),
        '--max-out-len', str(args.max_out_len),
        '--work-dir', work_dir,
        '--debug',
    ]

    # 模型类型
    if args.model_type == 'hf_chat':
        cmd.extend(['--hf-type', 'chat'])
    elif args.model_type == 'hf_base':
        cmd.extend(['--hf-type', 'base'])
    else:
        cmd.extend(['--hf-type', 'chat'])

    # Few-shot 在 OpenCompass 数据集中配置，不作为 CLI 参数
    # 用户如果配置了 --few_shot，通过环境变量传递给数据集配置
    if args.few_shot > 0:
        os.environ['OPENCOMPASS_FEW_SHOT'] = str(args.few_shot)
        print(f'[INFO] Few-shot 设置为 {args.few_shot}，'
              '请确保数据集配置中引用了该环境变量')

    # 额外的模型加载参数（OpenCompass 原生支持 key=value 格式）
    if args.model_kwargs:
        try:
            model_kwargs = json.loads(args.model_kwargs)
            if isinstance(model_kwargs, dict):
                for k, v in model_kwargs.items():
                    cmd.extend(['--model-kwargs', f'{k}={v}'])
            else:
                print(f'[WARN] model_kwargs 不是 JSON 对象，已跳过: {args.model_kwargs}')
        except json.JSONDecodeError:
            print(f'[WARN] model_kwargs 不是合法 JSON，已跳过: {args.model_kwargs}')

    return cmd


def run_opencompass(cmd: list, log_path: str):
    """执行 OpenCompass 命令，实时输出日志。"""
    print(f'[INFO] 执行命令: {" ".join(cmd)}')
    print(f'[INFO] 日志输出: {log_path}')
    print('=' * 60)

    process = None
    try:
        with open(log_path, 'w', encoding='utf-8') as log_f:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                universal_newlines=True,
            )

            # 实时输出日志
            for line in iter(process.stdout.readline, ''):
                print(line, end='', flush=True)
                log_f.write(line)

            process.wait()
    except FileNotFoundError:
        print(f'[ERROR] 命令未找到: {cmd[0]}，请确认 OpenCompass 已正确安装')
        # 仍尝试写入错误信息到日志文件
        with open(log_path, 'a', encoding='utf-8') as log_f:
            log_f.write(f'\n[FATAL] 命令未找到: {cmd[0]}\n')
        sys.exit(1)
    except OSError as e:
        print(f'[ERROR] 系统调用失败: {e}')
        with open(log_path, 'a', encoding='utf-8') as log_f:
            log_f.write(f'\n[FATAL] 系统调用失败: {e}\n')
        sys.exit(1)
    except Exception as e:
        print(f'[ERROR] 执行过程中发生未知错误: {e}')
        with open(log_path, 'a', encoding='utf-8') as log_f:
            log_f.write(f'\n[FATAL] 未知错误: {e}\n')
        sys.exit(1)

    print('=' * 60)
    if process is not None and process.returncode != 0:
        print(f'[ERROR] OpenCompass 运行失败，退出码: {process.returncode}')
        sys.exit(process.returncode)
    elif process is None:
        print('[ERROR] 进程未能启动，无法获取退出码')
        sys.exit(1)

    print(f'[OK] OpenCompass 运行完成')


def run_parse_and_save(opencompass_dir: str, output_path: str,
                       model_name: str, model_version: str,
                       model_path: str, datasets: str):
    """调用 parse_and_save.py 解析结果。"""
    parse_script = os.path.join(os.path.dirname(__file__), 'parse_and_save.py')
    if not os.path.exists(parse_script):
        print(f'[WARN] parse_and_save.py 未找到: {parse_script}，跳过解析')
        return

    cmd = [
        sys.executable,
        parse_script,
        '--opencompass-dir', opencompass_dir,
        '--output-path', output_path,
        '--model-name', model_name,
        '--model-version', model_version,
        '--model-path', model_path,
        '--datasets', datasets,
    ]

    print(f'[INFO] 解析评测结果: {" ".join(cmd)}')
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f'[WARN] 结果解析异常: {result.stderr}')

    # 软链 metric.json 到容器根目录（平台采集用）
    metric_json_src = os.path.join(output_path, 'metric.json')
    metric_json_dst = '/metric.json'
    if os.path.exists(metric_json_src):
        if os.path.exists(metric_json_dst):
            os.remove(metric_json_dst)
        os.symlink(metric_json_src, metric_json_dst)
        print(f'[OK] 软链 /metric.json -> {metric_json_src}')


def main():
    parser = argparse.ArgumentParser(
        description='OpenCompass 模型评测（作为 Cube Studio Pipeline 节点）'
    )

    # ---- 模型信息 ----
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型路径（HuggingFace 本地路径或模型目录）')
    parser.add_argument('--model_name', type=str, default='model',
                        help='模型名称')
    parser.add_argument('--model_version', type=str, default='v1',
                        help='模型版本号')
    parser.add_argument('--model_type', type=str, default='hf_chat',
                        choices=['hf_chat', 'hf_base'],
                        help='模型类型: hf_chat（对话模型）/ hf_base（基座模型）')

    # ---- 评测配置 ----
    parser.add_argument('--datasets', type=str, required=True,
                        help='评测数据集名称，多个用逗号分隔，如: ceval_gen,gsm8k_gen,mmlu_gen')
    parser.add_argument('--output_path', type=str, required=True,
                        help='评测结果输出根目录，所有文件写入此路径')

    # ---- 推理参数 ----
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='使用的 GPU 数量，默认 1')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='推理 batch size，默认 64')
    parser.add_argument('--max_seq_len', type=int, default=2048,
                        help='模型最大输入长度，默认 2048')
    parser.add_argument('--max_out_len', type=int, default=512,
                        help='模型最大输出长度，默认 512')
    parser.add_argument('--few_shot', type=int, default=0,
                        help='Few-shot 样本数，默认 0')
    parser.add_argument('--model_kwargs', type=str, default='',
                        help='模型加载参数（JSON 字符串），如: {"device_map":"auto"}')

    args = parser.parse_args()

    # ---- 打印参数 ----
    print('========== OpenCompass 评测配置 ==========')
    for k, v in vars(args).items():
        print(f'  {k}: {v}')
    print(f'  KFJ_CREATOR: {KFJ_CREATOR}')
    print(f'  KFJ_RUN_ID: {KFJ_RUN_ID}')
    print(f'  KFJ_PIPELINE_ID: {KFJ_PIPELINE_ID}')
    print('==========================================\n')

    # ---- 校验 ----
    check_opencompass()
    if not os.path.exists(args.model_path):
        print(f'[ERROR] 模型路径不存在: {args.model_path}')
        sys.exit(1)

    # 创建输出目录
    os.makedirs(args.output_path, exist_ok=True)

    # ---- 1. 运行 OpenCompass ----
    cmd = build_opencompass_cmd(args)
    log_path = os.path.join(args.output_path, 'eval_run.log')
    run_opencompass(cmd, log_path)

    # ---- 2. 解析结果 ----
    opencompass_out = os.path.join(args.output_path, 'opencompass_results')
    run_parse_and_save(
        opencompass_dir=opencompass_out,
        output_path=args.output_path,
        model_name=args.model_name,
        model_version=args.model_version,
        model_path=args.model_path,
        datasets=args.datasets,
    )

    print(f'\n[OK] 评测完成！结果目录: {args.output_path}')
    print(f'  - metric.json:      平台自动采集的指标')
    print(f'  - eval_summary.json: 结构化评测摘要')
    print(f'  - eval_report.csv:   评测结果表格')
    print(f'  - eval_run.log:      OpenCompass 运行日志')
    print(f'  - opencompass_results/: OpenCompass 原始输出\n')


if __name__ == '__main__':
    main()
