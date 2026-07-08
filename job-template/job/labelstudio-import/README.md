# labelstudio-import 模板

从 LabelStudio 标注平台导出标注数据，保存到平台 PVC 目录。

## 功能

- 连接 LabelStudio API，导出指定项目的标注数据
- 支持 JSON / JSON_MIN / COCO / YOLO / CSV / TSV 多种导出格式
- 支持自动下载标注关联的图片/音频等媒体文件到 media/ 目录
- 生成 data.csv 用于平台数据集预览
- 生成 meta.json 记录导出元数据
- 哨兵文件 (.labelstudio_synced) 记录同步状态，便于定时调度和外部脚本判断

## 安全说明

API Token 通过以下方式传入（优先级从高到低）：
1. `--ls_api_token` 命令行参数（明文，不推荐）
2. `LS_API_TOKEN` 环境变量（平台加密参数功能自动注入）

平台会对 Token 进行 **加密存储**（Fernet 对称加密），运行时不通过 CLI args 传递，
而是解密后通过环境变量注入容器，避免在 `ps aux` 或 Pod spec 中暴露。

## 使用方式

### 单次导入
在 MLops 平台创建 Pipeline，添加 labelstudio-import 节点，填好参数运行即可。

### 定时同步
将 Pipeline 的 schedule_type 设为 crontab，设置定时表达式，平台会自动周期性执行导入。
通过哨兵文件可判断上次同步时间。

## 参数说明

| 参数 | 必填 | 说明 |
|------|------|------|
| `--ls_api_token` | 是 | LabelStudio API Token（加密存储，运行时通过环境变量注入） |
| `--project_id` | 是 | LabelStudio 项目 ID |
| `--save_path` | 是 | PVC 保存路径 |
| `--ls_url` | 否 | LabelStudio 服务地址，默认集群内部地址 |
| `--export_format` | 否 | 导出格式，默认 JSON |
| `--download_media` | 否 | 是否下载媒体文件，默认 true |

## 输出文件

```
{保存路径}/
├── annotations.json      # 完整标注数据
├── data.csv              # 任务元信息表（task_id + data 字段）
├── meta.json             # 导出元数据（时间、项目、格式、数量）
├── media/                # 媒体文件（图片/音频/视频）
│   ├── xxx.jpg
│   └── ...
└── .labelstudio_synced   # 哨兵文件（project_id / task_count / synced_at）
```

## 构建镜像

```bash
bash build.sh
```
