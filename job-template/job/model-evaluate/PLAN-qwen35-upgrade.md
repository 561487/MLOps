# model-evaluate 镜像升级计划：支持 Qwen3.5 评测

> 状态：已实施（Dockerfile/build.sh 已改，待构建推送与 GPU 实测）
> 日期：2026-08-20
> 目标镜像：`10.121.177.20:8082/mlops/model-evaluate:main-py310-cu124-r1`

## 1. 背景与根因

用户用现有镜像评测 Qwen3.5-2B 失败，报错：

```
KeyError: 'qwen3_5'
ValueError: Transformers does not recognize this architecture
```

三层根因（已逐条核查）：

1. **transformers 版本锁死**：当前 Dockerfile 固定 `transformers>=4.44,<5.0`，而 `qwen3_5` 架构需 transformers ≥ 5.2.0 才注册（opencompass issue #2573 确认）。
2. **torch 连带约束**：transformers 5.x 要求 torch ≥ 2.6；当前 torch 2.5.1 + cu121。而 torch 2.6.0 **没有 cu121 wheel**（只有 cu124/cu126/cu118），基础镜像必须换 CUDA 12.4+。
3. **OpenCompass tokenization 不兼容**：transformers 5.x 移除了 `batch_encode_plus` API，而 OpenCompass main 分支 `opencompass/models/huggingface.py` 第 267-268 行仍在调用它。官方修复 PR #2575 **已关闭未合并**。→ 必须自己打补丁。

已验证的正面事实：

- transformers 5.x 能正确加载 Qwen3.5（issue #2573 报告者环境 torch 2.11 + transformers 5.12.1，模型加载成功，仅挂在 tokenize）。
- torch 2.6.0+cu124 wheel 存在且支持 Python 3.10。

## 2. 已确认的决策

| 决策点 | 选择 |
|--------|------|
| OpenCompass 版本 | **git main**（含 Qwen3.5 支持）+ 源码补丁 |
| transformers 版本 | **>=5.2**（不固定小版本） |
| causal_conv1d / fla kernel | **安装，允许失败**（失败不阻断构建，仅影响性能） |
| 基础镜像 | nvidia/cuda:12.4.1-runtime-ubuntu22.04 |
| torch | 2.6.0 + cu124（满足 transformers 5.x 的最低版本，风险最小） |

## 3. 实施步骤

### Step 1 — 基础镜像换 CUDA 12.4

`Dockerfile` 第 7 行：

```dockerfile
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
```

原因：torch 2.6 无 cu121 wheel。RTX 4090 驱动支持 CUDA 12.4。

### Step 2 — 升级 torch

`Dockerfile` 第 42-47 行：

```dockerfile
RUN pip3 install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu124 \
    --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    torch==2.6.0 \
    torchvision==0.21.0 \
    torchaudio==2.6.0
```

### Step 3 — OpenCompass main + transformers >= 5.2

> 实施时修正：构建机直连 GitHub 不稳定（HTTP2 framing 被重置 / 连接超时，实测两种都出现过），
> 从 `pip install git+https://...` 改为**三级回退克隆链 + 本地安装**，并强制 git HTTP/1.1。

`Dockerfile` 第 54-64 行（已写入）：

```dockerfile
RUN git config --global http.version HTTP/1.1 && \
    (git clone --depth 1 https://github.com/open-compass/opencompass.git /tmp/opencompass \
     || git clone --depth 1 https://ghfast.top/https://github.com/open-compass/opencompass.git /tmp/opencompass \
     || git clone --depth 1 https://gh-proxy.com/https://github.com/open-compass/opencompass.git /tmp/opencompass \
    ) && \
    pip3 install --no-cache-dir /tmp/opencompass Evaluate modelscope "transformers>=5.2" && \
    rm -rf /tmp/opencompass
```

注意：

- OpenCompass 从 PyPI 0.5.3 改为 git main（0.5.3 不含 Qwen3.5 支持）。
- ghfast.top / gh-proxy.com 为公益 GitHub 加速镜像，仅克隆公开仓库，无认证泄露风险。
- 回退链已真实环境验证：直连超时后 ghfast.top 成功克隆（commit a5acc4b）。
- `transformers>=5.2` 不固定小版本，pip 会装最新 5.x；若后续出现兼容问题再收紧。

### Step 4 —【强制】batch_encode_plus 补丁

> 实施时修正：PR #2575 涉及**多个 HF 后端文件**（huggingface.py 与 huggingface_above_v4_33.py 等），
> 补丁从"只改 huggingface.py"升级为**扫描整个 opencompass 包**，防止遗漏。

OpenCompass 安装后新增 RUN 步（已写入 Dockerfile Step 5）：

```dockerfile
RUN python3 - <<'EOF'
import opencompass, pathlib
pkg = pathlib.Path(opencompass.__file__).parent
total = 0
for f in sorted(pkg.rglob('*.py')):
    s = f.read_text(errors='ignore')
    n = s.count('tokenizer.batch_encode_plus(')
    if n:
        f.write_text(s.replace('tokenizer.batch_encode_plus(', 'tokenizer('))
        total += n
        print(f'[PATCH] {f.name}: {n} 处')
assert total > 0, '[PATCH FAIL] 未找到 batch_encode_plus 调用，OpenCompass 结构可能已变化，请人工检查'
print(f'[PATCH OK] 共替换 {total} 处 batch_encode_plus -> tokenizer.__call__')
EOF
```

要点：

- `tokenizer.batch_encode_plus(x, ...)` → `tokenizer(x, ...)`，即 PR #2575 的修复思路。
- `self.tokenizer.batch_encode_plus(...)` 因子串匹配同样被正确替换为 `self.tokenizer(...)`。
- 包级扫描：覆盖所有模型后端文件，不只 huggingface.py。
- `assert total > 0`：OpenCompass 结构变化时构建立即失败，绝不产出坏镜像。
- 若上游日后合并官方修复，此 assert 会失败 → 届时删除本补丁步。
- heredoc 语法需要 BuildKit 构建（用户构建机已是 BuildKit，实测日志确认）。

### Step 5 — Qwen3.5 可选 kernel（允许失败）

```dockerfile
RUN pip3 install --no-cache-dir causal-conv1d fla \
    || echo "[WARN] causal-conv1d/fla 安装失败，Qwen3.5 将回退到慢速 PyTorch 实现（功能不受影响）"
```

不装能跑但慢且吃显存；装失败不阻断构建。

### Step 6 — 冒烟测试升级

> 实施时修正：残留检测按**调用模式** `tokenizer.batch_encode_plus(` 匹配，
> 而非裸词 `batch_encode_plus`（源码注释里的字样会造成误报）。

`build.sh` 冒烟测试增加（已写入）：

```bash
# 1. 版本断言
docker run --rm --entrypoint python3 "${IMAGE}" -c "
import opencompass, transformers, torch
assert transformers.__version__.split('.')[0] == '5', transformers.__version__
assert torch.__version__.split('.')[1] == '6', torch.__version__
print('opencompass', opencompass.__version__,
      '| transformers', transformers.__version__,
      '| torch', torch.__version__)"

# 2. 补丁生效断言: opencompass 内不应再有 batch_encode_plus 调用
docker run --rm --entrypoint python3 "${IMAGE}" -c "
import opencompass, pathlib
pkg = pathlib.Path(opencompass.__file__).parent
hits = [str(f) for f in pkg.rglob('*.py')
        if 'tokenizer.batch_encode_plus(' in f.read_text(errors='ignore')]
assert not hits, hits
print('[OK] batch_encode_plus 补丁验证通过')"
```

### Step 7 — Tag 与平台配置同步

- `build.sh` 的 `IMAGE_VERSION`：`0.5.3-py310-cu121-r1` → `main-py310-cu124-r1`
  （framework 从 0.5.3 变 main，CUDA 从 121 变 124，符合平台 Tag 规范）
- 推送成功并确认 Harbor 有该 Tag 后，更新 `myapp/init/init-job-template.json` 中 model-evaluate 的 `image_name`。

## 4. 可靠性审查表

| 环节 | 状态 | 依据 |
|------|------|------|
| transformers 5.x 加载 Qwen3.5 | ✅ 上游真实环境验证 | opencompass issue #2573 |
| torch 2.6+cu124 wheel（py310） | ✅ 已验证 | PyTorch 官方版本表 |
| OpenCompass main 原生支持 transformers 5 tokenize | ❌ 不支持（未打补丁必崩） | main 分支 huggingface.py 实测 |
| 补丁后 tokenize 可用 | ⚠️ 思路经 PR #2575 验证（15 tests passed），本镜像未实测 | PR #2575 |
| 补丁后 Qwen3.5 端到端出分 | ❌ 无人验证，必须实测 | — |
| causal_conv1d / fla 可安装 | ⚠️ 对 CUDA/torch 版本敏感，需实测 | transformers 文档 |

**总体判断：可行，但属高风险、必须实测级别，不是改版本号就能成。**

## 5. 验证门槛（全过才算成功）

1. 镜像内 `import opencompass, transformers, torch` 无错，版本符合预期。
2. opencompass 源码中 `batch_encode_plus` 出现次数为 0。
3. **真实跑**（需 GPU 环境，本地无法执行）：Qwen3.5-2B + C-Eval `--max_samples=10` 在 RTX 4090 上出分。
4. **回归**：Qwen2.5 等老模型仍能正常评测（确认升级未破坏存量能力）。

## 6. 回滚策略

- 旧镜像 `0.5.3-py310-cu121-r1` 保留不删。
- 新镜像出问题 → 平台配置切回旧 Tag（旧镜像继续支持 Qwen2.5 等）。

## 7. 备选方案（若方案 A 实测失败）

| 备选 | 说明 | 可靠性 |
|------|------|--------|
| B. vLLM 后端 | Qwen2.5 稳；Qwen3.5 存在 FLA 乱码 bug（截至 2026-06 未修复） | 中 |
| C. 等官方 | 等 OpenCompass 合并 transformers 5 修复并发新版 | 最稳但被动 |
| D. VLMEvalKit | OpenCompass 生态多模态评测包，已支持 Qwen3.5 | 中（评测流程不同） |

## 8. 改动文件清单

| 文件 | 改动 |
|------|------|
| `job-template/job/model-evaluate/Dockerfile` | Step 1-5 |
| `job-template/job/model-evaluate/build.sh` | Step 6-7（冒烟测试 + Tag） |
| `myapp/init/init-job-template.json` | Step 7（推送成功后同步 image_name） |

不涉及：`src/run_evaluation.py`、`src/parse_and_save.py`、`src/sitecustomize.py`（评测入口逻辑不变）。

## 9. 已知约束与提醒

- 本机（构建机）磁盘/内存紧张且 docker.io 拉取极慢，**构建需在资源充足、网络正常的机器上执行**。
- `max_samples=10` 是"每学科 10 条"，C-Eval 约 52 学科 → 约 520 条，不是总共 10 条。
- OpenCompass main 非稳定 release，若后续官方发布含修复的新版本，应迁移回 PyPI 固定版本并删除补丁。
