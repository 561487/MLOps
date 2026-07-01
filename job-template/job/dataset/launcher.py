import shutil, os
shutil.rmtree('/tmp/ms_cache', ignore_errors=True)
import sys
import argparse
import datetime
import json
import time
from multiprocessing import Pool
from functools import partial
import uuid
import pysnooper
import re
import requests
import copy
import os
KFJ_CREATOR = os.getenv('KFJ_CREATOR', 'admin')
KFJ_TASK_PROJECT_NAME = os.getenv('KFJ_TASK_PROJECT_NAME','public')
SENTINEL_FILE = '.dataset_downloaded'

host = os.getenv('HOST',os.getenv('KFJ_MODEL_REPO_API_URL','http://kubeflow-dashboard.infra')).strip('/')

# @pysnooper.snoop()
def download_file(url,des_dir=None,local_path=None):
    if des_dir:
        local_path = os.path.join(des_dir, url.split('/')[-1])
    # 先检查文件是否已存在，避免不必要的 HTTP 请求
    if os.path.exists(local_path):
        print(f'{local_path} 已经存在，跳过下载')
        return
    print(f'begin download {local_path} from {url}')
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=102400):
                f.write(chunk)

# @pysnooper.snoop()
def download(name,version,partition,save_dir,download_limit='0',**kwargs):
    # 检查数据集状态
    status = check_dataset_exists(save_dir, download_limit)
    if status == 'skip':
        exit(0)
    elif status == 'apply_limit':
        apply_row_limit(save_dir, download_limit)
        write_sentinel(save_dir, download_limit)
        exit(0)
    # status == 'download' → 继续正常下载流程

    # print(kwargs)
    headers = {
        'Content-Type': 'application/json',
        'Authorization': KFJ_CREATOR
    }

    # 获取项目组
    url = host + "/dataset_modelview/api/?form_data=" + json.dumps({
        "filters": [
            {
                "col": "name",
                "opr": "eq",
                "value":name
            },
            {
                "col": "version",
                "opr": "eq",
                "value": version
            }
        ]
    })
    res = requests.get(url, headers=headers)
    exist_dataset = res.json().get('result', {}).get('data', [])
    if not exist_dataset:
        print('不存在指定数据集或指定版本')
        exit(1)
    exist_dataset = exist_dataset[0]

    # 查询同名是否存在，创建者是不是指定用户
    url = host+f"/dataset_modelview/api/download/{exist_dataset['id']}"
    if partition:
        url = url + "/" + partition
    # print(url)
    res = requests.get(url,headers=headers, allow_redirects=False)
    # print(res.content)
    if res.status_code==200:
        donwload_urls = res.json().get("result", {}).get("download_urls", [])
        # 限制下载数量
        limit = int(download_limit) if download_limit else 0
        if limit > 0:
            total_before = len(donwload_urls)
            donwload_urls = donwload_urls[:limit]
            print(f'下载数量限制: {limit}，共 {total_before} 个文件，实际下载前 {len(donwload_urls)} 个')
        print('启动并行下载:',donwload_urls)
        os.makedirs(save_dir, exist_ok=True)
        pool = Pool(len(donwload_urls))  # 开辟包含指定数目线程的线程池
        pool.map(partial(download_file, des_dir=save_dir), donwload_urls)  # 当前worker，只处理分配给当前worker的任务
        pool.close()
        pool.join()
        print('启动并行完成', save_dir)

        # 对目录下的压缩文件进行解压
        files = os.listdir(save_dir)
        for file in files:
            try:
                if '.tar.gz' in file:
                    expected_output = file.replace('.tar.gz', '')
                    expected_path = os.path.join(save_dir, expected_output)
                    if os.path.exists(expected_path):
                        print(f'{file} 已解压至 {expected_path}，跳过')
                        continue
                    exe_command(f'cd {save_dir} && tar -zxvf {file}')
                    print('解压文件完成', save_dir)
                elif '.zip' in file:
                    expected_output = file.replace('.zip', '')
                    expected_path = os.path.join(save_dir, expected_output)
                    if os.path.exists(expected_path):
                        print(f'{file} 已解压至 {expected_path}，跳过')
                        continue
                    exe_command(f'cd {save_dir} && unzip {file}')
                    print('解压文件完成',save_dir)
                elif '.gz' in file:
                    expected_output = file.replace('.gz', '')
                    expected_path = os.path.join(save_dir, expected_output)
                    if os.path.exists(expected_path):
                        print(f'{file} 已解压至 {expected_path}，跳过')
                        continue
                    exe_command(f'cd {save_dir} && gzip -d {file}')
                    print('解压文件完成', save_dir)
            except Exception as e:
                print(e)

        # 写入哨兵文件
        write_sentinel(save_dir, download_limit)
        exit(0)

    exit(1)

from subprocess import Popen, PIPE, STDOUT

def exe_command(command):
    """
    执行 shell 命令并实时打印输出
    :param command: shell 命令
    :return: process, exitcode
    """
    print(command)
    process = Popen(command, stdout=PIPE, stderr=STDOUT, shell=True)
    with process.stdout:
        for line in iter(process.stdout.readline, b''):
            print(line.decode().strip(),flush=True)
    exitcode = process.wait()
    return exitcode


def check_dataset_exists(save_dir, download_limit='0'):
    """
    检查 save_dir 下数据集状态，返回:
      'skip'        — 数据已满足需求，无需任何操作
      'apply_limit' — 数据已存在，仅需对现有数据重新应用行数限制
      'download'    — 需要重新下载（数据不存在或需要更多数据）
    """
    sentinel_path = os.path.join(save_dir, SENTINEL_FILE)
    if not os.path.isfile(sentinel_path):
        return 'download'

    # 读取已存储的 limit
    import re
    stored_limit = 0
    try:
        with open(sentinel_path, 'r') as sf:
            m = re.search(r'limit=(\d+)', sf.read())
            stored_limit = int(m.group(1)) if m else 0
    except Exception:
        stored_limit = 0

    req_limit = int(download_limit) if download_limit else 0

    # 1. 请求和存储完全一致 → 跳过
    if stored_limit == req_limit:
        print(f'目录已存在当前数据集: {save_dir} (limit={req_limit})')
        return 'skip'

    # 2. 请求 limit=0（全量），但存储数据已被截断 → 必须重新下载
    if req_limit == 0 and stored_limit > 0:
        print(f'已有限制数据(limit={stored_limit})无法恢复为全量，重新下载')
        os.remove(sentinel_path)
        return 'download'

    # 3. 请求 limit > 存储 limit → 需要更多数据，重新下载
    if req_limit > stored_limit and stored_limit > 0:
        print(f'请求 limit({req_limit}) > 已有 limit({stored_limit})，需要更多数据，重新下载')
        os.remove(sentinel_path)
        return 'download'

    # 4. 存储为全量(stored=0) 或 已有数据多于需求(stored > req) → 对现有数据应用限制即可
    print(f'目录已存在数据集: {save_dir}，对现有数据应用 limit={req_limit}')
    return 'apply_limit'


def write_sentinel(save_dir, download_limit='0'):
    """写入哨兵文件，记录 limit 值"""
    sentinel_path = os.path.join(save_dir, SENTINEL_FILE)
    try:
        req_limit = str(download_limit) if download_limit else '0'
        with open(sentinel_path, 'w') as sf:
            sf.write(f'downloaded at {datetime.datetime.now().isoformat()} limit={req_limit}\n')
        print(f'写入标记文件: {sentinel_path}')
    except Exception as e:
        print(f'写入标记文件失败: {e}')


def apply_row_limit(save_dir, limit):
    """
    对 save_dir 下的 Parquet 数据文件应用行数限制，
    只保留前 limit 行，用于大数据集快速测试。
    """
    import pandas as pd
    limit = int(limit) if limit else 0
    if limit <= 0:
        return
    print(f'应用行数限制: 每个 parquet 文件最多保留 {limit} 行')
    for root, dirs, files in os.walk(save_dir):
        for f in files:
            fpath = os.path.join(root, f)
            if not f.endswith('.parquet'):
                continue
            try:
                df = pd.read_parquet(fpath)
                if len(df) > limit:
                    df.head(limit).to_parquet(fpath, index=False)
                    print(f'  {fpath}: {len(df)} → {limit} 行')
            except Exception as e:
                print(f'  {fpath} 行数限制失败: {e}')


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser("download dataset launcher")
    arg_parser.add_argument('--src_type', type=str, help="数据集来源", default='当前平台')
    arg_parser.add_argument('--name', type=str, help="数据集名称", default='')
    arg_parser.add_argument('--version', type=str, help="数据集版本", default='latest')
    arg_parser.add_argument('--partition', type=str, help="数据集分区", default='')
    arg_parser.add_argument('--save_dir', type=str, help="保存目录", default='')
    arg_parser.add_argument('--download_limit', type=str, help="下载限制：当前平台=文件数，modelscope=parquet行数，0=全部", default='0')

    args = arg_parser.parse_args()
    if not args.save_dir:
        args.save_dir = f'/mnt/{KFJ_CREATOR}/dataset/{args.name}/{args.version}'
        if args.partition:
            args.save_dir=f'/mnt/{KFJ_CREATOR}/dataset/{args.name}/{args.version}/{args.partition}'
    # print("{} args: {}".format(__file__, args))
    if args.src_type=='cube-studio' or args.src_type=='当前平台':
        download(**args.__dict__)
    elif args.src_type=='modelscope' or args.src_type=='魔塔':
        # 检查数据集状态
        status = check_dataset_exists(args.save_dir, args.download_limit)
        if status == 'skip':
            exit(0)
        elif status == 'apply_limit':
            apply_row_limit(args.save_dir, args.download_limit)
            write_sentinel(args.save_dir, args.download_limit)
            exit(0)
        # status == 'download' → 继续正常下载流程
        command = f'modelscope download --dataset {args.name} --repo-type dataset --local_dir {args.save_dir} --cache_dir /tmp/ms_cache'
        exitcode = exe_command(command)
        if exitcode == 0:
            # 应用行数限制
            apply_row_limit(args.save_dir, args.download_limit)
            write_sentinel(args.save_dir, args.download_limit)
        exit(exitcode)


