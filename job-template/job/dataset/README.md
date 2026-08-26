# dataset 模板
镜像：10.121.177.20:8082/mlops/dataset:20260824

参数
```bash
{
    "参数": {
      "--src_type": {
            "type": "str",
            "item_type": "str",
            "label": "数据集的来源",
            "require": 1,
            "choice": ["当前平台","modelscope"],
            "range": "",
            "default": "当前平台",
            "placeholder": "",
            "describe": "数据集的来源",
            "editable": 1
        },
        "--name": {
            "type": "str",
            "item_type": "str",
            "label": "数据集的名称",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "",
            "placeholder": "",
            "describe": "数据集的名称",
            "editable": 1
        },
        "--version": {
            "type": "str",
            "item_type": "str",
            "label": "数据集的版本",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "latest",
            "placeholder": "",
            "describe": "数据集的版本；ModelScope留空或填写latest时使用master分支",
            "editable": 1
        },
        "--partition": {
            "type": "str",
            "item_type": "str",
            "label": "数据集的分区，或者子数据集",
            "require": 0,
            "choice": [],
            "range": "",
            "default": "",
            "placeholder": "",
            "describe": "数据集的分区，或者子数据集",
            "editable": 1
        },
        "--save_dir": {
            "type": "str",
            "item_type": "str",
            "label": "数据集的保存地址",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "",
            "placeholder": "",
            "describe": "数据集的保存地址",
            "editable": 1
        },
        "--download_percent": {
            "type": "float",
            "item_type": "float",
            "label": "下载百分比",
            "require": 0,
            "choice": [],
            "range": "0,100",
            "default": "100",
            "placeholder": "例如 30",
            "describe": "按数据量下载指定百分比；100表示全量下载。ModelScope按文件体积选择分片，单个大文件无法拆分。",
            "editable": 1
        }
    }
}
```
