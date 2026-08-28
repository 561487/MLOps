# -*- coding: utf-8 -*-
"""sitecustomize.py — Python 启动时自动加载，为 MsDataset.load 注入 trust_remote_code=True。

在 OpenCompass 子进程中也会生效，解决 C-Eval 等数据集需要远程脚本执行的问题。
同时为 MsDataset.load 加容错：当 HF datasets 库因 name=None 等格式问题崩溃时，
自动 fallback 到直接加载 jsonl/csv 文件。
"""
import os

os.environ.setdefault('HF_DATASETS_TRUST_REMOTE_CODE', '1')


def _patch_msdataset():
    try:
        from modelscope import MsDataset
        _original_load = MsDataset.load

        def _patched_load(*args, **kwargs):
            kwargs.setdefault('trust_remote_code', True)
            try:
                return _original_load(*args, **kwargs)
            except TypeError as e:
                # 典型症状: datasets/arrow_reader.py "Expected str 'name', but got: NoneType"
                # ModelScope 下载的数据格式与当前 HF datasets 库不兼容时触发
                if "Expected str 'name'" in str(e) or "NoneType" in str(e):
                    print(f'[sitecustomize] MsDataset.load 因格式不兼容失败({e})，'
                          f'尝试 fallback 直接加载')
                    return _fallback_load(*args, **kwargs)
                raise

        MsDataset.load = staticmethod(_patched_load)

        if hasattr(MsDataset, '_load'):
            _original_load2 = MsDataset._load

            @staticmethod
            def _patched_load2(*args, **kwargs):
                kwargs.setdefault('trust_remote_code', True)
                try:
                    return _original_load2(*args, **kwargs)
                except TypeError as e:
                    if "Expected str 'name'" in str(e) or "NoneType" in str(e):
                        print(f'[sitecustomize] MsDataset._load 因格式不兼容失败({e})，'
                              f'尝试 fallback 直接加载')
                        return _fallback_load(*args, **kwargs)
                    raise

            MsDataset._load = _patched_load2

        print('[sitecustomize] MsDataset.load patched: trust_remote_code=True + format-fallback')
    except ImportError:
        pass


def _fallback_load(*args, **kwargs):
    """当 MsDataset.load 因格式不兼容崩溃时，尝试直接加载本地文件。

    常见场景: ModelScope 下载的 jsonl 文件，用 datasets.load_dataset('json') 直接读。
    """
    from datasets import load_dataset as hf_load_dataset
    import json

    path = args[0] if args else kwargs.get('path', '')
    split = kwargs.get('split', 'train')

    if not path or not os.path.exists(str(path)):
        raise FileNotFoundError(f'_fallback_load: path={path} 不存在')

    # 如果是目录，找 jsonl 文件
    if os.path.isdir(str(path)):
        jsonl_files = [os.path.join(str(path), f) for f in os.listdir(str(path))
                       if f.endswith('.jsonl') or f.endswith('.json')]
        if jsonl_files:
            print(f'[sitecustomize] fallback: 从目录 {path} 加载 {len(jsonl_files)} 个 jsonl 文件')
            return hf_load_dataset('json', data_files=jsonl_files, split=split)

    # 如果是单文件
    if str(path).endswith('.jsonl') or str(path).endswith('.json'):
        print(f'[sitecustomize] fallback: 直接加载 jsonl 文件 {path}')
        return hf_load_dataset('json', data_files=str(path), split=split)

    # 兜底：用原始 MsDataset.load 的路径参数再试一次（不带 split）
    print(f'[sitecustomize] fallback: 尝试 MsDataset.load(path={path}, 不带 split)')
    from modelscope import MsDataset
    return MsDataset.load(str(path))


def _patch_smart_data_source():
    """智能数据源选路：包装 opencompass 的 get_data_path。

    每个数据集独立决策：
      1. 绝对路径（自定义数据集）→ 原样放行
      2. 本地缓存已存在 → 直接返回
      3. 官方链路优先：OSS 直链命中则自动下载（download_dataset 内部匹配）
      4. 官方断言 "No valid url" 且有 ms_id → 本次调用临时启用 ModelScope
      5. 都不可用 → 报错并指引离线数据包

    关键坑：opencompass/utils/__init__.py 里 `from .datasets import *` 会把旧函数
    拷贝进父包命名空间，而 datasets/*.py 大多 `from opencompass.utils import get_data_path`，
    因此必须同时覆盖「子模块」和「父包」两处绑定。
    """
    try:
        import opencompass.utils.datasets as _ocd
        import opencompass.utils as _ocdu
        import traceback
    except Exception:
        print('[sitecustomize] opencompass 未安装，跳过智能选源补丁')
        return

    _orig_get_data_path = _ocd.get_data_path

    def _smart_get_data_path(dataset_id, local_mode=False):
        if os.path.isabs(str(dataset_id)):
            return dataset_id
        try:
            mapping = (_ocd.DATASETS_MAPPING or {}).get(dataset_id) or {}
        except Exception:
            mapping = {}
        local_rel = mapping.get('local') or str(dataset_id)
        cache_dir = os.environ.get('COMPASS_DATA_CACHE', '')
        local_full = os.path.join(cache_dir, local_rel)

        # 缓存命中，直接返回，避免联网/下载日志噪音
        if os.path.exists(local_full):
            return local_full

        try:
            # 先走官方原始链路（本地 else 分支的 download_dataset：
            # 本地缓存不存在时用 DATASETS_URL 的 OSS 直链自动下载）
            return _orig_get_data_path(dataset_id, local_mode)
        except AssertionError as e:
            if 'No valid url' not in str(e):
                raise
            # 无 OSS 直链 → 若该数据集在 DATASETS_MAPPING 有 ms_id，
            # 本次调用临时启用 ModelScope repo 加载（PIQA 等格式 bug 数据集除外）
            ms_id = mapping.get('ms_id')
            if not ms_id:
                raise FileNotFoundError(
                    f'内置数据集 {dataset_id} 既无 OSS 直链也无可用 ModelScope 源。'
                    f'请下载 OpenCompass 离线数据包 '
                    f'(OpenCompassData-complete-20240207.zip) '
                    f'并解压到 datasets_cache_dir 后重试') from e
            print(f'[sitecustomize] {dataset_id}: 无 OSS 直链 → '
                  f'本次调用走 ModelScope ({ms_id})')
            os.environ['DATASET_SOURCE'] = 'ModelScope'
            try:
                return _orig_get_data_path(dataset_id, local_mode)
            finally:
                os.environ.pop('DATASET_SOURCE', None)

    _ocd.get_data_path = _smart_get_data_path
    # 同步覆盖父包绑定（关键！datasets/*.py 多从 opencompass.utils 导入本函数）
    if hasattr(_ocdu, 'get_data_path'):
        _ocdu.get_data_path = _smart_get_data_path

    print('[sitecustomize] smart data-source routing installed '
          '(submodule + package)')


_patch_msdataset()
_patch_smart_data_source()
