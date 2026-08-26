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
import math
from urllib.parse import quote
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
def parse_download_percent(value):
    """Parse and validate a download percentage in the range (0, 100]."""
    try:
        percent = float(value if value not in (None, '') else 100)
    except (TypeError, ValueError):
        raise ValueError(f'下载百分比必须是数字，当前值: {value}')
    if percent <= 0 or percent > 100:
        raise ValueError(f'下载百分比必须大于 0 且不超过 100，当前值: {percent}')
    return percent


def download(name,version,partition,save_dir,download_limit='0',download_percent='100',**kwargs):
    download_percent = parse_download_percent(download_percent)
    # 检查数据集状态
    status = check_dataset_exists(save_dir, download_limit, download_percent)
    if status == 'skip':
        exit(0)
    elif status == 'apply_limit':
        apply_row_limit(save_dir, download_limit)
        write_sentinel(save_dir, download_limit, download_percent)
        exit(0)
    elif status == 'conflict':
        exit(1)
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
        if download_percent < 100:
            total_before = len(donwload_urls)
            selected_count = max(1, math.floor(total_before * download_percent / 100))
            donwload_urls = donwload_urls[:selected_count]
            print(
                f'按百分比下载: 请求 {download_percent:g}%，当前平台按文件数量取前 '
                f'{selected_count}/{total_before} 个文件'
            )
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
        write_sentinel(save_dir, download_limit, download_percent)
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


def exe_command_args(command):
    """Execute an argv list without a shell, preserving spaces in file paths."""
    print(' '.join(command))
    process = Popen(command, stdout=PIPE, stderr=STDOUT)
    with process.stdout:
        for line in iter(process.stdout.readline, b''):
            print(line.decode(errors='replace').strip(), flush=True)
    return process.wait()


def _modelscope_endpoint():
    endpoint = os.getenv(
        'MODELSCOPE_ENDPOINT',
        os.getenv('MODELSCOPE_DOMAIN', 'https://modelscope.cn')
    ).rstrip('/')
    if not endpoint.startswith(('http://', 'https://')):
        endpoint = 'https://' + endpoint
    return endpoint


def _modelscope_revision(version):
    # “1998” is the historical default for the built-in MNIST example. It was
    # previously ignored for ModelScope downloads, so keep that path backward
    # compatible and resolve it to ModelScope's default branch.
    return 'master' if not version or version in ('latest', '1998') else version


def list_modelscope_dataset_files(dataset_id, version='latest'):
    """Return all regular repository files as path/size dictionaries."""
    if dataset_id.count('/') != 1:
        raise ValueError('ModelScope 数据集名称必须是 namespace/dataset_name 格式')
    namespace, dataset_name = dataset_id.split('/', 1)
    endpoint = _modelscope_endpoint()
    url = (
        f'{endpoint}/api/v1/datasets/{quote(namespace, safe="")}/'
        f'{quote(dataset_name, safe="")}/repo/tree'
    )
    headers = {}
    token = os.getenv('MODELSCOPE_API_TOKEN', '').strip()
    if token:
        headers['Authorization'] = f'Bearer {token}'

    revision = _modelscope_revision(version)
    pending_dirs = ['']
    visited_dirs = set()
    result = []
    while pending_dirs:
        root = pending_dirs.pop(0)
        if root in visited_dirs:
            continue
        visited_dirs.add(root)
        page_number = 1
        while True:
            params = {
                'Revision': revision,
                'Root': root,
                'PageNumber': page_number,
                'PageSize': 100,
            }
            response = requests.get(url, params=params, headers=headers, timeout=120)
            response.raise_for_status()
            payload = response.json()
            if payload.get('Code') != 200:
                raise RuntimeError(
                    f'ModelScope 文件列表获取失败: {payload.get("Message", payload)}'
                )
            data = payload.get('Data') or {}
            entries = data.get('Files') or []
            for entry in entries:
                path = str(entry.get('Path') or entry.get('Name') or '').strip('/')
                if not path:
                    continue
                entry_type = str(entry.get('Type') or '').lower()
                if entry_type in ('tree', 'dir', 'directory'):
                    pending_dirs.append(path)
                    continue
                if entry_type and entry_type != 'blob':
                    continue
                result.append({'path': path, 'size': max(0, int(entry.get('Size') or 0))})

            total_count = int(data.get('TotalCount') or payload.get('TotalCount') or len(entries))
            if not entries or page_number * 100 >= total_count:
                break
            page_number += 1

    deduplicated = {}
    for item in result:
        deduplicated[item['path']] = item
    return list(deduplicated.values())


def _natural_path_key(path):
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r'(\d+)', path)]


def _is_repository_metadata(path):
    name = os.path.basename(path).lower()
    return (
        name.startswith('readme')
        or name.startswith('license')
        or name in {
            '.gitattributes', '.gitignore',
            'dataset_infos.json', 'dataset_info.json',
        }
    )


def select_modelscope_files_by_percent(files, download_percent):
    """
    Select a deterministic subset whose data-file bytes do not exceed the
    requested percentage. Repository metadata is always included and is not
    counted in the percentage.
    """
    percent = parse_download_percent(download_percent)
    files = sorted(files, key=lambda item: _natural_path_key(item['path']))
    metadata_files = [item for item in files if _is_repository_metadata(item['path'])]
    data_files = [item for item in files if not _is_repository_metadata(item['path'])]
    if not data_files:
        raise RuntimeError('ModelScope 仓库中未发现可下载的数据文件')
    if percent >= 100:
        return files, 100.0

    total_size = sum(item['size'] for item in data_files)
    if total_size <= 0:
        raise RuntimeError('ModelScope 未返回有效文件大小，无法按数据量百分比下载')
    target_size = total_size * percent / 100
    selected_data = []
    selected_size = 0
    for item in data_files:
        if item['size'] <= 0:
            selected_data.append(item)
            continue
        if selected_size + item['size'] <= target_size:
            selected_data.append(item)
            selected_size += item['size']

    if not any(item['size'] > 0 for item in selected_data):
        smallest = min((item for item in data_files if item['size'] > 0),
                       key=lambda item: item['size'])
        smallest_percent = smallest['size'] * 100 / total_size
        raise RuntimeError(
            f'数据集分片过大：最小数据文件 {smallest["path"]} 占 '
            f'{smallest_percent:.2f}%，无法在不超过 {percent:g}% 的前提下部分下载。'
            '请使用更高百分比或选择已分片的数据集。'
        )

    actual_percent = selected_size * 100 / total_size
    return metadata_files + selected_data, actual_percent


def download_modelscope_by_percent(dataset_id, version, save_dir, download_percent):
    files = list_modelscope_dataset_files(dataset_id, version)
    selected_files, actual_percent = select_modelscope_files_by_percent(
        files, download_percent
    )
    print(
        f'按数据量百分比下载: 请求 {float(download_percent):g}%，'
        f'实际选择约 {actual_percent:.2f}%，共 {len(selected_files)}/{len(files)} 个仓库文件'
    )
    revision = _modelscope_revision(version)
    os.makedirs(save_dir, exist_ok=True)
    batch_size = 100
    for start in range(0, len(selected_files), batch_size):
        batch = selected_files[start:start + batch_size]
        command = [
            'modelscope', 'download', '--dataset', dataset_id,
            '--repo-type', 'dataset',
        ]
        command.extend(item['path'] for item in batch)
        command.extend([
            '--revision', revision,
            '--local_dir', save_dir,
            '--cache_dir', '/tmp/ms_cache',
        ])
        exitcode = exe_command_args(command)
        if exitcode != 0:
            return exitcode
    return 0


def check_dataset_exists(save_dir, download_limit='0', download_percent='100'):
    """
    检查 save_dir 下数据集状态，返回:
      'skip'        — 数据已满足需求，无需任何操作
      'apply_limit' — 数据已存在，仅需对现有数据重新应用行数限制
      'download'    — 需要重新下载（数据不存在或需要更多数据）
    """
    sentinel_path = os.path.join(save_dir, SENTINEL_FILE)
    if not os.path.isfile(sentinel_path):
        return 'download'

    # 读取已存储的 limit 和 percent
    import re
    stored_limit = 0
    stored_percent = 100.0
    try:
        with open(sentinel_path, 'r') as sf:
            sentinel_content = sf.read()
            m = re.search(r'limit=(\d+)', sentinel_content)
            stored_limit = int(m.group(1)) if m else 0
            m_percent = re.search(r'percent=([0-9.]+)', sentinel_content)
            stored_percent = float(m_percent.group(1)) if m_percent else 100.0
    except Exception:
        stored_limit = 0
        stored_percent = 100.0

    req_limit = int(download_limit) if download_limit else 0
    req_percent = parse_download_percent(download_percent)

    if stored_percent != req_percent:
        print(
            f'保存目录已包含 percent={stored_percent:g}% 的数据，不能直接改为 '
            f'percent={req_percent:g}%，否则会混入旧分片。请更换保存目录。'
        )
        return 'conflict'

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


def write_sentinel(save_dir, download_limit='0', download_percent='100'):
    """写入哨兵文件，记录 limit 值"""
    sentinel_path = os.path.join(save_dir, SENTINEL_FILE)
    try:
        req_limit = str(download_limit) if download_limit else '0'
        req_percent = parse_download_percent(download_percent)
        with open(sentinel_path, 'w') as sf:
            sf.write(
                f'downloaded at {datetime.datetime.now().isoformat()} '
                f'limit={req_limit} percent={req_percent:g}\n'
            )
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
    arg_parser.add_argument(
        '--download_percent',
        type=str,
        help="下载数据量百分比，范围 0-100；ModelScope 按文件体积计算，100=全量",
        default='100'
    )

    args = arg_parser.parse_args()
    if not args.save_dir:
        args.save_dir = f'/mnt/{KFJ_CREATOR}/dataset/{args.name}/{args.version}'
        if args.partition:
            args.save_dir=f'/mnt/{KFJ_CREATOR}/dataset/{args.name}/{args.version}/{args.partition}'
    # print("{} args: {}".format(__file__, args))
    if args.src_type=='cube-studio' or args.src_type=='当前平台':
        download(**args.__dict__)
    elif args.src_type=='modelscope' or args.src_type=='魔塔':
        try:
            download_percent = parse_download_percent(args.download_percent)
        except ValueError as e:
            print(e)
            exit(1)
        # 检查数据集状态
        status = check_dataset_exists(
            args.save_dir, args.download_limit, download_percent
        )
        if status == 'skip':
            exit(0)
        elif status == 'apply_limit':
            apply_row_limit(args.save_dir, args.download_limit)
            write_sentinel(
                args.save_dir, args.download_limit, download_percent
            )
            exit(0)
        elif status == 'conflict':
            exit(1)
        # status == 'download' → 继续正常下载流程
        if download_percent < 100:
            try:
                exitcode = download_modelscope_by_percent(
                    args.name,
                    args.version,
                    args.save_dir,
                    download_percent,
                )
            except Exception as e:
                print(f'ModelScope 按百分比下载失败: {e}')
                exit(1)
        else:
            command = [
                'modelscope', 'download', '--dataset', args.name,
                '--repo-type', 'dataset',
                '--revision', _modelscope_revision(args.version),
                '--local_dir', args.save_dir,
                '--cache_dir', '/tmp/ms_cache',
            ]
            exitcode = exe_command_args(command)
        if exitcode == 0:
            # 应用行数限制
            apply_row_limit(args.save_dir, args.download_limit)
            write_sentinel(
                args.save_dir, args.download_limit, download_percent
            )
        exit(exitcode)
