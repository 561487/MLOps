#!/usr/bin/env python3
"""
LabelStudio 标注数据导入工具
从 LabelStudio API 导出标注数据，保存到指定 PVC 目录。

安全说明：
  API Token 通过环境变量 LS_API_TOKEN 或 --ls_api_token 参数传入。
  推荐使用平台加密参数功能，Token 在数据库中加密存储、运行时通过环境变量注入，
  不在 ps aux 或 kubectl describe pod 中暴露。

哨兵文件：
  每次成功导出后会写入 .labelstudio_synced 哨兵文件，记录同步时间和任务数量，
  便于外部脚本判断是否需要后续处理。
"""

import argparse
import datetime
import json
import os
import re
import sys
import time
from urllib.parse import urlparse

import requests

SENTINEL_FILE = '.labelstudio_synced'
# 优先从环境变量读取 API Token（平台加密敏感参数通过环境变量注入）
LS_API_TOKEN = os.getenv('LS_API_TOKEN', '')


def arg_parser():
    parser = argparse.ArgumentParser(description='LabelStudio 标注数据导入')
    parser.add_argument('--ls_url', type=str,
                        default='http://labelstudio.kubeflow:8080',
                        help='LabelStudio 服务地址')
    parser.add_argument('--ls_api_token', type=str,
                        default=LS_API_TOKEN or None,
                        required=not bool(LS_API_TOKEN),
                        help='LabelStudio API Token（也可通过环境变量 LS_API_TOKEN 传入）')
    parser.add_argument('--project_id', type=int, required=True,
                        help='LabelStudio 项目 ID')
    parser.add_argument('--export_format', type=str,
                        default='JSON',
                        choices=['JSON', 'JSON_MIN', 'COCO', 'YOLO', 'CSV', 'TSV'],
                        help='导出格式')
    parser.add_argument('--save_path', type=str, required=True,
                        help='保存目录')
    parser.add_argument('--download_media', type=str,
                        default='true',
                        choices=['true', 'false'],
                        help='是否下载媒体文件（图片/音频）')
    return parser


def check_sentinel(save_path, project_id):
    """检查哨兵文件，获取上次同步时间"""
    sentinel_path = os.path.join(save_path, SENTINEL_FILE)
    if not os.path.isfile(sentinel_path):
        return None

    try:
        with open(sentinel_path, 'r') as f:
            content = f.read()
            m = re.search(r'project_id=(\d+)', content)
            stored_project = int(m.group(1)) if m else 0
            m = re.search(r'synced_at=(.+)', content)
            stored_time = m.group(1).strip() if m else ''
            if stored_project == project_id and stored_time:
                return stored_time
    except Exception as e:
        print(f'读取哨兵文件失败: {e}')
    return None


def write_sentinel(save_path, project_id, task_count):
    """写入哨兵文件"""
    sentinel_path = os.path.join(save_path, SENTINEL_FILE)
    try:
        synced_at = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        with open(sentinel_path, 'w') as f:
            f.write(f'project_id={project_id}\n')
            f.write(f'task_count={task_count}\n')
            f.write(f'synced_at={synced_at}\n')
        print(f'哨兵文件写入: {sentinel_path}')
    except Exception as e:
        print(f'哨兵文件写入失败: {e}')


def get_project_info(ls_url, headers, project_id):
    """获取 LabelStudio 项目信息"""
    url = f'{ls_url}/api/projects/{project_id}'
    print(f'获取项目信息: {url}')
    resp = requests.get(url, headers=headers)
    if resp.status_code != 200:
        print(f'获取项目信息失败: {resp.status_code} {resp.text}')
        return None
    project = resp.json()
    print(f'项目名称: {project.get("title", "unknown")}')
    return project


def export_annotations(ls_url, headers, project_id, export_format):
    """从 LabelStudio 导出标注数据"""
    url = f'{ls_url}/api/projects/{project_id}/export?exportType={export_format}'
    print(f'导出标注数据: {url}')
    resp = requests.get(url, headers=headers, timeout=300)
    if resp.status_code != 200:
        print(f'导出失败: {resp.status_code} {resp.text[:500]}')
        return None

    # LabelStudio export API: JSON 格式返回 list，其他格式返回二进制
    try:
        data = resp.json()
        print(f'导出成功: {len(data)} 条标注记录')
        return data
    except Exception:
        print(f'导出成功: {export_format} 格式（非 JSON），原始字节数: {len(resp.content)}')
        return resp.content


def download_media_file(ls_url, headers, media_path, save_dir):
    """从 LabelStudio 下载单个媒体文件"""
    # 处理不同的 URL 格式
    if media_path.startswith('http://') or media_path.startswith('https://'):
        # 外部 URL，直接下载
        file_name = os.path.basename(urlparse(media_path).path)
        if not file_name or len(file_name) > 200:
            file_name = f'{int(time.time() * 1000)}.dat'
        save_path = os.path.join(save_dir, file_name)
        try:
            resp = requests.get(media_path, timeout=60)
            if resp.status_code == 200:
                with open(save_path, 'wb') as f:
                    f.write(resp.content)
                return os.path.relpath(save_path, save_dir)
        except Exception as e:
            print(f'  下载外部文件失败 {media_path}: {e}')
        return media_path

    # LabelStudio 内部路径: /data/upload/... 或 /data/local-files/...
    clean_path = media_path.lstrip('/')
    url = f'{ls_url}/{clean_path}'
    file_name = os.path.basename(clean_path)
    if not file_name or file_name in ('upload', 'local-files', 'data'):
        file_name = f'{int(time.time() * 1000)}.dat'

    save_path = os.path.join(save_dir, file_name)

    # 避免重复下载
    if os.path.exists(save_path):
        print(f'  文件已存在，跳过: {file_name}')
        return os.path.relpath(save_path, save_dir)

    try:
        print(f'  下载: {media_path} → {file_name}')
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(resp.content)
            return os.path.relpath(save_path, save_dir)
        else:
            print(f'  下载失败: {url} status={resp.status_code}')
    except Exception as e:
        print(f'  下载异常 {url}: {e}')

    return media_path


def collect_media_paths(annotations):
    """从标注数据中收集所有媒体文件路径"""
    paths = set()
    for task in annotations:
        if not isinstance(task, dict):
            continue
        data = task.get('data', {})
        for key, value in data.items():
            if isinstance(value, str):
                # 图片、音频、视频等媒体路径
                if any(value.startswith(p) for p in ('/data/', 'http://', 'https://')):
                    paths.add(value)
                # 文件扩展名判断
                if any(value.lower().endswith(ext) for ext in
                       ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp',
                        '.wav', '.mp3', '.ogg', '.flac', '.aac',
                        '.mp4', '.avi', '.mov', '.webm')):
                    paths.add(value)
    return list(paths)


def replace_media_paths(annotations, path_map):
    """将标注 JSON 中的原始媒体路径替换为下载后的本地路径"""
    for task in annotations:
        if not isinstance(task, dict):
            continue
        data = task.get('data', {})
        for key, value in list(data.items()):
            if isinstance(value, str) and value in path_map:
                data[key] = path_map[value]
    return annotations


def build_data_csv(annotations, save_path):
    """从标注数据生成 data.csv 元信息文件"""
    import csv
    csv_path = os.path.join(save_path, 'data.csv')
    if not annotations or not isinstance(annotations, list):
        return

    # 收集所有 data 字段的 key
    all_keys = set()
    for task in annotations:
        if isinstance(task, dict):
            all_keys.update(task.get('data', {}).keys())

    if not all_keys:
        all_keys = {'data'}

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['task_id'] + sorted(all_keys))
        writer.writeheader()
        for task in annotations:
            if not isinstance(task, dict):
                continue
            row = {'task_id': task.get('id', '')}
            data = task.get('data', {})
            for key in all_keys:
                row[key] = data.get(key, '')
            writer.writerow(row)

    print(f'元信息 CSV 写入: {csv_path}')


def build_meta_json(save_path, project_info, export_format, task_count):
    """生成 meta.json"""
    meta = {
        'exported_at': datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
        'project_id': project_info.get('id', ''),
        'project_title': project_info.get('title', ''),
        'export_format': export_format,
        'task_count': task_count,
    }
    meta_path = os.path.join(save_path, 'meta.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f'元数据写入: {meta_path}')


def main():
    parser = arg_parser()
    args = parser.parse_args()

    ls_url = args.ls_url.rstrip('/')
    api_token = args.ls_api_token
    project_id = args.project_id
    export_format = args.export_format
    save_path = args.save_path
    download_media = args.download_media.lower() == 'true'

    print(f'=== LabelStudio 标注导入 ===')
    print(f'服务地址: {ls_url}')
    print(f'项目 ID: {project_id}')
    print(f'导出格式: {export_format}')
    print(f'保存路径: {save_path}')
    print(f'下载媒体: {download_media}')

    # 认证：LabelStudio 1.23 Personal Access Token 需先 refresh 换取 access token
    # 流程：POST /api/token/refresh/ → 获取 access token → Bearer 鉴权
    print('获取 access token...')
    try:
        resp = requests.post(f'{ls_url}/api/token/refresh/',
                             json={'refresh': api_token}, timeout=10)
        if resp.status_code == 200:
            access_token = resp.json().get('access', api_token)
            print('access token 获取成功')
        else:
            # 如果不是 refresh token，直接用原 token 作为 access token
            print(f'refresh 失败 (HTTP {resp.status_code})，使用原始 token')
            access_token = api_token
    except Exception as e:
        print(f'refresh 异常: {e}，使用原始 token')
        access_token = api_token

    headers = {'Authorization': f'Bearer {access_token}'}

    # 1. 获取项目信息
    project_info = get_project_info(ls_url, headers, project_id)
    if not project_info:
        print('无法获取项目信息，退出')
        sys.exit(1)

    # 2. 检查哨兵文件（记录上次同步状态，便于外部判断是否需要后续处理）
    last_sync = check_sentinel(save_path, project_id)
    if last_sync:
        print(f'上次同步时间: {last_sync}')

    # 3. 导出标注数据
    annotations = export_annotations(ls_url, headers, project_id, export_format)
    if annotations is None:
        print('导出标注数据失败')
        sys.exit(1)

    task_count = len(annotations) if isinstance(annotations, list) else -1

    # 4. 准备保存目录
    os.makedirs(save_path, exist_ok=True)

    # 5. 下载媒体文件
    if download_media and isinstance(annotations, list):
        media_paths = collect_media_paths(annotations)
        if media_paths:
            print(f'\n=== 下载媒体文件 ({len(media_paths)} 个) ===')
            media_dir = os.path.join(save_path, 'media')
            os.makedirs(media_dir, exist_ok=True)

            path_map = {}
            for mp in media_paths:
                local_rel = download_media_file(ls_url, headers, mp, media_dir)
                path_map[mp] = local_rel

            # 替换标注 JSON 中的媒体路径
            replace_media_paths(annotations, path_map)
            print(f'媒体文件下载完成，已替换 {len(path_map)} 个路径')

    # 6. 保存标注数据
    if isinstance(annotations, list):
        # JSON 格式：保存为 annotations.json
        annotations_path = os.path.join(save_path, 'annotations.json')
        with open(annotations_path, 'w', encoding='utf-8') as f:
            json.dump(annotations, f, indent=2, ensure_ascii=False)
        print(f'\n标注数据保存: {annotations_path}')

        # 生成 data.csv
        build_data_csv(annotations, save_path)

    elif isinstance(annotations, bytes):
        # 非 JSON 格式（COCO/YOLO/CSV/TSV）：直接写文件
        ext_map = {
            'JSON': 'json', 'JSON_MIN': 'json',
            'COCO': 'json', 'YOLO': 'zip',
            'CSV': 'csv', 'TSV': 'tsv'
        }
        ext = ext_map.get(export_format, 'txt')
        annotations_path = os.path.join(save_path, f'annotations.{ext}')
        with open(annotations_path, 'wb') as f:
            f.write(annotations)
        print(f'\n标注数据保存: {annotations_path}')

    # 7. 生成 meta.json
    build_meta_json(save_path, project_info, export_format, task_count)

    # 8. 写哨兵文件
    write_sentinel(save_path, project_id, task_count)

    print(f'\n=== 导入完成 ===')
    print(f'数据路径: {save_path}')
    print(f'标注数量: {task_count}')
    print(f'项目: {project_info.get("title", "N/A")}')


if __name__ == '__main__':
    main()
