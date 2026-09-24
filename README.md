# MLOps 平台

基于 Kubernetes 的一站式 AI 研发与交付平台，覆盖在线开发、数据处理、模型训练、可视化工作流、模型评估与注册、推理服务以及集群资源治理。平台将 Kubernetes、容器镜像、共享存储和工作流引擎封装为统一的 Web 控制台，帮助算法、数据和运维团队协同完成模型从实验到上线的全生命周期管理。

## 核心能力

| 领域 | 能力 |
| --- | --- |
| 在线开发 | Notebook、Jupyter、VS Code、Web 终端、CPU/GPU 规格与数据目录挂载 |
| 数据工程 | 数据集管理、数据清洗、格式转换、合并拆分、特征处理与 ETL Pipeline |
| 训练与调度 | 单机/分布式训练、可视化 Pipeline、定时任务、失败重试、Ray、Volcano、NNI |
| 大模型工具链 | 模型下载、MS-SWIFT、LLaMA-Factory、DeepSpeed、离线推理、量化、剪枝、蒸馏与评估 |
| 模型资产 | 模型注册、版本管理、指标与评估报告、模型市场、训练链路追踪 |
| 推理服务 | CPU/GPU/vGPU 部署、版本升级、健康检查、弹性伸缩、日志与运行指标 |
| 平台治理 | 多项目/多租户、角色权限、多集群、资源配额、GPU 监控与任务状态管理 |

## 平台架构

```text
Web 控制台
  ├─ React 管理前端
  ├─ AI Pipeline / ETL Pipeline 可视化编排
  └─ Python 平台后端
          │
          ├─ 用户、项目、权限与资源管理
          ├─ Notebook、数据集、镜像与模型管理
          ├─ Pipeline、训练任务与推理服务管理
          └─ 定时任务、状态同步与通知
                  │
                  ▼
Kubernetes / Argo Workflow / Volcano / Ray
  ├─ CPU、GPU 与 vGPU 节点
  ├─ MySQL / Redis / MinIO
  ├─ NFS、Ceph、JuiceFS 等共享存储
  ├─ Harbor 或其他容器镜像仓库
  └─ Prometheus / Grafana / DCGM Exporter
```

## 仓库结构

```text
.
├── myapp/                 # 平台后端、管理前端和可视化工作流
│   ├── frontend/          # React 管理前端
│   ├── vision/            # AI Pipeline 编排器
│   ├── visionPlus/        # 数据 ETL Pipeline 编排器
│   ├── models/            # 数据模型
│   ├── views/             # Web/API 视图
│   ├── tasks/             # 后台任务
│   └── init/              # 初始化数据与任务模板定义
├── job-template/          # 数据、训练、评估、量化和部署等任务模板
├── install/
│   ├── docker/            # 本地开发与 Docker Compose 环境
│   ├── kubernetes/        # Kubernetes 部署清单
│   └── ops/               # 运维脚本
├── images/                # Notebook、CUDA、Serving 等基础镜像
├── harbor/                # 私有镜像仓库部署配置
└── aihub/                 # AI Hub 相关资源
```

## 快速开始

### 1. 获取代码

```bash
git clone https://github.com/561487/MLOps.git
cd MLOps
```

### 2. 准备本地环境

本地开发推荐使用 Docker Compose。基础要求：

- Docker 与 Docker Compose
- Python 3.9（后端本地依赖与代码检查）
- Node.js 16.15+、npm 6.14.8+（前端开发）
- Git，并建议关闭自动换行转换：`git config --global core.autocrlf false`

完整步骤、镜像构建和调试方式见 [本地开发文档](install/docker/README.md)。

### 3. 启动开发环境

```bash
cd install/docker
docker compose up -d
```

首次启动后进入后端容器执行初始化脚本：

```bash
docker exec -it docker-myapp-1 bash
/entrypoint.sh
```

后续可在容器内启动后端：

```bash
python myapp/run.py
```

> Docker Compose 文件和容器名可能随环境调整，请以 `install/docker/` 中的配置为准。

### 4. 前端开发

启动前端前，需要先启动可访问的后端服务并按需修改 `src/setupProxy.js`。

```bash
cd myapp/frontend
npm install
npm run start
```

默认开发入口：`http://localhost:3000/frontend/`。

AI Pipeline 与 ETL Pipeline 前端分别位于 `myapp/vision` 和 `myapp/visionPlus`，详细命令见 [本地开发文档](install/docker/README.md)。

## Kubernetes 部署

生产环境运行在 Kubernetes 上。当前部署文档建议 Kubernetes 1.25～1.31（推荐 1.28），并根据使用规模准备：

- 控制节点和 CPU/GPU 任务节点；
- MySQL、Redis、MinIO 或兼容 S3 的对象存储；
- NFS、Ceph、JuiceFS 等共享文件系统；
- Harbor 或其他私有镜像仓库；
- Prometheus、Grafana、Node Exporter 与 DCGM Exporter；
- 可选的 Istio、Volcano、Ray 与分布式训练组件。

部署前请阅读 [平台基础架构与部署说明](install/README.md)。不同集群的存储类、镜像仓库、域名、GPU 驱动和网络配置需要按实际环境修改。

## 内置任务模板

`job-template/job/` 提供可组合进 Pipeline 的任务模板，包括：

- 数据处理：数据集转换、合并拆分、预处理、DataX、文本清洗、隐私替换、音视频与视觉处理；
- 传统机器学习：XGBoost、LightGBM、GBDT、AdaBoost、决策树；
- 深度学习与分布式训练：PyTorch、TensorFlow、Ray、Volcano、DeepSpeed；
- 大模型：MS-SWIFT、LLaMA-Factory、模型下载、离线推理、量化、剪枝和蒸馏；
- 评估与交付：模型评估、模型转换、模型注册与服务部署；
- 标注与语音：Label Studio、Whisper 等。

新增模板时请遵循 [任务模板开发规范](job-template/README.md)。典型目录如下：

```text
job-template/job/<template-name>/
├── Dockerfile
├── build.sh
├── readme.md
└── src/
```

## 开发说明

### 后端

后端代码位于 `myapp/`。依赖清单和容器构建配置位于 `install/docker/`，开发环境支持通过代码挂载进行热更新。

### 前端

管理前端使用 React 17、TypeScript、Ant Design、MobX、ECharts 和 D3。常用命令：

```bash
cd myapp/frontend
npm run start
npm run build
npm run test
```

### 修改前的建议

1. 从目标分支创建功能分支；
2. 不要提交 kubeconfig、密码、Token、私钥或内部服务凭据；
3. 修改任务模板时同步更新模板说明与初始化配置；
4. 提交前执行相关单元测试、前端构建或 Python 语法检查；
5. 涉及部署配置时，在独立测试集群验证后再用于生产环境。

## 适用场景

- 企业内部 AI 中台和私有化 MLOps 平台；
- 多团队共享 CPU/GPU 集群和统一资源治理；
- 机器学习、深度学习及大模型任务的标准化编排；
- 数据处理、训练、评估、模型注册和服务发布的端到端流程；
- 内网环境中的镜像、模型、数据与推理服务管理。

## License

本项目采用 [MIT License](LICENSE)。第三方依赖的许可证说明见仓库根目录 `LICENSE`。
