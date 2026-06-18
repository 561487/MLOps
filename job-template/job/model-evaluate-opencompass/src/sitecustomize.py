# -*- coding: utf-8 -*-
"""sitecustomize.py — Python 启动时自动加载，为 MsDataset.load 注入 trust_remote_code=True。

在 OpenCompass 子进程中也会生效，解决 C-Eval 等数据集需要远程脚本执行的问题。
"""
import os

os.environ.setdefault('HF_DATASETS_TRUST_REMOTE_CODE', '1')


def _patch_msdataset():
    try:
        from modelscope import MsDataset
        _original_load = MsDataset.load

        def _patched_load(*args, **kwargs):
            kwargs.setdefault('trust_remote_code', True)
            return _original_load(*args, **kwargs)

        MsDataset.load = staticmethod(_patched_load)

        if hasattr(MsDataset, '_load'):
            _original_load2 = MsDataset._load

            @staticmethod
            def _patched_load2(*args, **kwargs):
                kwargs.setdefault('trust_remote_code', True)
                return _original_load2(*args, **kwargs)

            MsDataset._load = _patched_load2

        print('[sitecustomize] MsDataset.load patched: trust_remote_code=True')
    except ImportError:
        pass


_patch_msdataset()
