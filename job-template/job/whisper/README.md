# Whisper Base

模型市场语音识别最小可运行模板，支持一键开发、一键部署和在线音频转写。

## 镜像

```text
10.121.177.20:8082/mlops/whisper:20260811
```

## 构建

```bash
docker build --network=host \
  -t 10.121.177.20:8082/mlops/whisper:20260811 .
docker push 10.121.177.20:8082/mlops/whisper:20260811
```

构建镜像时会通过 ModelScope 下载 `openai-mirror/whisper-base` 到
`/whisper/models/whisper-base`。运行阶段强制只从该本地目录加载，不依赖外网。

如需使用其他 ModelScope 仓库，可在构建时传入
`--build-arg WHISPER_MODEL_SOURCE=<model-id>`。

## 启动

```bash
MODELPATH=/whisper/models/whisper-base python server.py
```

服务端口为 `8000`，健康检查地址为 `/health`，推理地址为：

```text
/v1/models/whisper-base/versions/v1/predict
```

首期不包含微调 Pipeline、流式识别和说话人分离。
