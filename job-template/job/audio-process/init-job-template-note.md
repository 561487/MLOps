# init-job-template.json 配置说明

如果采用 `job-template/job/audio-process` 单目录方案，三个平台节点可以共用同一个镜像：

```text
10.121.177.20:8082/mlops/audio-process:20260630
```

三个节点的差异由 `process_type` 参数控制：

| 平台节点 | process_type |
|---|---|
| audio-quality-assessment | quality_assessment |
| audio-clean | clean |
| audio-augment | augment |

配置时保留三个独立的 `job_template_name`，但 `job_template_image` 都指向同一个镜像。

示例核心字段：

```json
{
  "job_template_name": "audio-clean",
  "job_template_describe": "音频数据清洗",
  "job_template_image": "10.121.177.20:8082/mlops/audio-process:20260630",
  "job_template_command": "python3 launcher.py",
  "job_template_workdir": "/app"
}
```

参数中需要增加：

```text
process_type=clean
```

质量评估节点设置：

```text
process_type=quality_assessment
```

数据增强节点设置：

```text
process_type=augment
```

路径类参数默认值保持为空，由用户按 `/mnt/storage/models-storage/...` 填写。
