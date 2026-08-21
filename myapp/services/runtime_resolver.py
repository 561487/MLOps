"""
模型 Runtime 镜像版本管理 Resolver（第四版：模型身份优先沿 Pipeline 数据血缘解析）。

职责：任务创建时，根据「模型 + Job Template runtime_key + 场景」自动选择
人工验证兼容的 Runtime 镜像，普通用户无需感知镜像版本。

模型身份解析主链路（第四版）：
    resolve_task_image()
        → resolve_model_key()      # 模型标识（三优先级，见下）
        → resolve_runtime_image()  # model/scene/runtime_key → ModelRuntimeMapping → Images

模型标识解析优先级（resolve_model_key）：
    1. Pipeline 上游 model-download（仅 modelscope/魔塔 来源）：沿 dag_json 的 upstream
       边向上游可达闭包查找 save_path 与当前模型路径「精确匹配」的 model-download 节点，
       直接使用其 --model_name（ModelScope 完整 repo id，如 Qwen/Qwen3-0.6B）。
       PVC 路径只负责模型加载，不承担身份识别；禁止 basename(path) 猜身份。
    2. 参数本身已是 org/model 完整 repo id（非本地绝对路径且含 '/'）→ 原样保留完整 repo id。
    3. Legacy 兼容：无法从上游识别时，normalize_model_key() 取最后一级名称查询历史 mapping
       （历史数据可能只保存了 basename 如 Qwen3-0.6B）。

其他规则（保持第三版不变）：
- 兼容旧数据：primary key 未命中时按 legacy 候选（basename/完整路径）再次查询
- 只有 model_runtime_mapping 中 is_default=true 且 Images.runtime_enabled=true 的记录才会被选择
- 数据级双重校验：mapping.runtime_key == Images.runtime_key（若被人为构造不一致则查询不到，拒绝启动）
- 找不到匹配 → 抛 MyappException 阻止任务启动（不允许静默回退到模板默认镜像）
- 已保存 runtime_image 的历史任务直接复用（重跑/恢复不按最新 mapping 重新解析）
- job_template.runtime_key 为空的旧 Job 不受影响（返回 None，调用方走原逻辑）
- 归一化只用于 Runtime 匹配；任务真正传给训练框架的 --model_name 等原始参数不变
"""
import json

from sqlalchemy import case

from myapp import db
from myapp.exceptions import MyappException
from myapp.models.model_job import Images, Task
from myapp.models.model_runtime import ModelRuntimeMapping

# runtime_key → scene 映射（第二版分析确定；scene 与页面 SCENE_CHOICES 保持一致）
# scene: finetune / pretrain / quantization / inference
# - litgpt → pretrain 为 litgpt-pretrain 预留（模板尚未配置 runtime_key，接入后自动生效）
RUNTIME_SCENE = {
    'msswift': 'finetune',
    'llama_factory': 'finetune',
    'deepspeed': 'finetune',
    'litgpt': 'pretrain',
    'gptqmodel': 'quantization',
    'vllm': 'inference',
    'sglang': 'inference',
}

# runtime_key → 任务参数中的模型参数名（不同 Job Template 的参数名不同）
RUNTIME_MODEL_ARG = {
    'msswift': '--model',
    'llama_factory': '--model_name',
    'gptqmodel': '--model',
    'deepspeed': '--model_name_or_path',
}

# model-download 节点：Job Template 名（myapp/init/init-job-template.json "model-download"）
MODEL_DOWNLOAD_TEMPLATE_NAME = 'model-download'

# --from 中可作为可靠模型身份的来源：--model_name 是 ModelScope 完整 repo id。
# --from=模型管理 / 推理服务 的 model_name 属于自由命名（Training_Model.name / InferenceService.model_name），
# huggingface 暂不扩大范围 —— 这些来源继续走 legacy fallback。
RELIABLE_DOWNLOAD_SOURCES = ('modelscope', '魔塔')


def normalize_model_key(model):
    """模型标识归一化（legacy 兼容用）：只取最后一级名称作为 canonical model key（仅用于 Runtime 匹配）。

    /mnt/storage/models-share-volume/models/Qwen3-0.6B  → Qwen3-0.6B
    Qwen/Qwen3-0.6B                                     → Qwen3-0.6B
    Qwen3-0.6B                                          → Qwen3-0.6B

    - 保留原始大小写，不做 lowercase / 模糊匹配 / 前缀猜测
    - 空值（或只有斜杠）返回 ''，由调用方按原规则报错
    - 只改变 Resolver 匹配用的模型标识；任务真正传给训练框架的 --model_name 等参数不变
    - 第四版起仅作为 Priority 3（legacy）与 legacy 候选使用，不再是主身份解析
    """
    value = str(model or '').strip()
    if not value:
        return ''
    value = value.replace('\\', '/').rstrip('/')
    if not value:
        return ''
    return value.rsplit('/', 1)[-1]


def normalize_model_path(path):
    """模型路径轻量归一化（仅用于 save_path / model_path 精确比较）。

    只允许：strip、\\ → /、去尾部斜杠。例如 /mnt/A/model/ == /mnt/A/model。
    禁止 basename / lowercase / contains 模糊匹配 / 猜父目录 —— 路径必须精确对应。
    """
    value = str(path or '').strip()
    if not value:
        return ''
    value = value.replace('\\', '/').rstrip('/')
    return value


def _render_creator(value, creator):
    """模板变量 {{creator}} 最小渲染：与 view_pipeline/view_task 的 template_str(creator=...) 语义一致。

    只替换 {{creator}} 一种变量，其他 {{...}} 原样保留 —— 避免在 Resolver 复制另一套模板引擎。
    task.args 在数据库中始终是原始模板值（渲染发生在 Pod 启动时），
    因此路径比较两侧都用同一个 creator（pipeline.created_by.username）渲染后比较。
    """
    if not value or not creator:
        return value
    return str(value).replace('{{creator}}', creator)


def _is_local_model_path(model):
    """最小可靠判断：以 / 开头的绝对路径即视为 PVC/local 模型路径。
    ModelScope/HF 完整 repo id（org/model）永远不会以 / 开头。"""
    return str(model or '').startswith('/')


def _is_model_repo_id(model):
    """org/model 形式完整 repo id 判断：非本地绝对路径且含 '/'。"""
    return not _is_local_model_path(model) and '/' in str(model or '')


def _find_upstream_model_download(task, model_path):
    """沿 Pipeline dag_json 向上游可达闭包查找与 model_path「精确匹配」的 model-download 节点。

    - dag_json 结构：{task_name: {..., 'upstream': [直接上游任务名]}}（邻接表，见 Pipeline.fix_dag_json）
    - BFS 沿 upstream 边向上遍历（支持多级中间节点），近节点优先
    - 只认来源 modelscope/魔塔（RELIABLE_DOWNLOAD_SOURCES）的 model-download，
      --from=模型管理/推理服务/huggingface 不产生可靠模型身份，继续走 legacy fallback
    - 匹配条件：normalize_model_path(save_path) == normalize_model_path(model_path)，
      两侧都先做 {{creator}} 最小渲染（同一 pipeline 的 creator 一致）
    - 返回该节点的 --model_name（完整 repo id）；无匹配返回 None
    """
    pipeline = task.pipeline
    if not pipeline or not getattr(pipeline, 'dag_json', None):
        return None
    try:
        dag = json.loads(pipeline.dag_json)
    except (TypeError, ValueError):
        return None
    if not dag or task.name not in dag:
        return None
    creator = pipeline.created_by.username if pipeline.created_by else ''
    target = normalize_model_path(_render_creator(model_path, creator))
    if not target:
        return None
    tasks = {t.name: t for t in db.session.query(Task).filter_by(pipeline_id=pipeline.id).all()}
    visited = set()
    queue = list(dag[task.name].get('upstream', []) or [])
    while queue:
        name = queue.pop(0)
        if name in visited:
            continue
        visited.add(name)
        node = tasks.get(name)
        if node and node.job_template and node.job_template.name == MODEL_DOWNLOAD_TEMPLATE_NAME:
            dl_args = json.loads(node.args or '{}')
            source = str(dl_args.get('--from', '') or '').strip()
            if source in RELIABLE_DOWNLOAD_SOURCES:
                dl_path = normalize_model_path(
                    _render_creator(str(dl_args.get('--save_path', '') or ''), creator))
                if dl_path and dl_path == target:
                    return str(dl_args.get('--model_name', '') or '').strip() or None
        queue.extend(dag.get(name, {}).get('upstream', []) or [])
    return None


def resolve_model_key(task, task_args=None, model=None, runtime_key=None):
    """统一模型标识解析入口（Resolver 主链路第一环，第四版新增）。

    返回 (primary_model_key, legacy_model_keys)：
      primary：最可信的模型标识，用于 mapping 首次查询
      legacy： 兼容候选（历史 mapping 格式），按优先级排列，primary 未命中时依次查询

    解析优先级（严格顺序）：
    1. Pipeline 上游 model-download（modelscope/魔塔，save_path 精确匹配）→ 其 --model_name 完整 repo id
    2. 参数本身已是 org/model 完整 repo id → 原样保留（不做 basename）
    3. Legacy：normalize_model_key() 取最后一级 → 查询历史 mapping（如 Qwen3-0.6B）
    """
    if task_args is None:
        task_args = json.loads(task.args) if task.args else {}
    if runtime_key is None:
        job_template = task.job_template
        runtime_key = (job_template.runtime_key or '').strip() if job_template else ''
    if model is None:
        model_arg = RUNTIME_MODEL_ARG.get(runtime_key)
        model = str(task_args.get(model_arg, '') or '').strip() if model_arg else ''
    model = str(model or '').strip()
    if not model:
        return '', []

    if _is_local_model_path(model):
        # Priority 1：PVC/local 路径 → 沿 Pipeline 上游找 model-download（主机制）
        upstream_model = _find_upstream_model_download(task, model)
        if upstream_model:
            # legacy 候选：repo id 的 basename（历史 mapping 风格）+ 路径 basename + 完整路径（更旧历史）
            legacy = [normalize_model_key(upstream_model),
                      normalize_model_key(model), model]
            return upstream_model, list(dict.fromkeys(legacy))
        # Priority 3：无上游可识别 → 按历史行为取路径最后一级（兼容历史 mapping）
        return normalize_model_key(model), [model]

    if _is_model_repo_id(model):
        # Priority 2：org/model 完整 repo id，原样保留；legacy 候选为 basename（历史 mapping）
        return model, [normalize_model_key(model)]

    # 纯模型名（无斜杠、非路径）：原样使用，与 normalize_model_key 结果一致，无需额外候选
    return model, []


def resolve_runtime_image(model, runtime_key, scene, legacy_keys=None):
    """根据 模型标识 + Runtime + 场景 解析最终 Harbor 镜像地址。

    model 为 resolve_model_key() 返回的 primary key；legacy_keys 为兼容候选
    （历史 mapping 的 basename / 完整路径等，按传入顺序参与匹配，primary 优先）。

    链路：model → ModelRuntimeMapping → Images → Images.name
    - 只使用 is_default=true 且 Images.runtime_enabled=true 的记录
    - 双重校验：mapping.runtime_key == Images.runtime_key（数据上杜绝错误组合）
    - 匹配不到时抛异常（不允许 fallback 到未验证的 Runtime）
    """
    primary = str(model or '').strip()
    candidates = list(dict.fromkeys([k for k in [primary] + list(legacy_keys or []) if k]))
    if not candidates:
        raise MyappException(
            '模型标识为空，无法解析 %s Runtime（场景 %s）。' % (runtime_key, scene)
        )
    mapping = (
        db.session.query(ModelRuntimeMapping)
        .join(Images, ModelRuntimeMapping.images_id == Images.id)
        .filter(
            ModelRuntimeMapping.model.in_(candidates),
            ModelRuntimeMapping.scene == scene,
            ModelRuntimeMapping.runtime_key == runtime_key,
            ModelRuntimeMapping.images_id.isnot(None),
            ModelRuntimeMapping.is_default == True,  # noqa: E712
            Images.runtime_key == runtime_key,  # 数据级双重校验：mapping 与镜像类型必须一致
            Images.runtime_enabled == True,  # noqa: E712
        )
        .order_by(
            case([(ModelRuntimeMapping.model == c, i) for i, c in enumerate(candidates)],
                 else_=len(candidates)),
            ModelRuntimeMapping.id.desc(),
        )
        .first()
    )
    if not mapping:
        raise MyappException(
            '无法确定模型 %s 对应的已验证 Runtime（场景 %s）。'
            '如果模型来自模型导入节点，请确认该节点与当前任务的模型路径一致；'
            '否则请联系管理员配置 Runtime 映射（模型标识 %s）。'
            % (primary or '空', scene, runtime_key)
        )
    if not mapping.images or not mapping.images.name:
        raise MyappException(
            '模型 %s 的 %s Runtime 未配置镜像地址，请联系管理员在「镜像管理」中关联镜像。'
            % (primary or '空', runtime_key)
        )
    return mapping.images.name


def resolve_task_image(task, task_args=None):
    """任务级镜像解析入口（view_task / view_pipeline 创建 Pod 前调用）。

    返回最终镜像地址；job_template.runtime_key 为空时返回 None（调用方保留原逻辑）。

    - 历史任务已保存 runtime_image → 直接复用，不按最新 mapping 重新解析
    - runtime_key 非空 → 从任务参数取 model，经 resolve_model_key() 解析模型标识，
      再调用 resolve_runtime_image()，成功后写回 task.runtime_image（保证历史任务可追溯）
    """
    if task_args is None:
        task_args = json.loads(task.args) if task.args else {}

    job_template = task.job_template
    runtime_key = (job_template.runtime_key or '').strip() if job_template else ''

    # 历史任务：优先使用创建时保存的镜像，避免重跑被新 mapping 影响
    if getattr(task, 'runtime_image', None):
        return task.runtime_image

    # runtime_key 为空 = 未纳入 Runtime 管理，完全保持旧逻辑
    if not runtime_key:
        return None

    # 从任务参数中取模型（第一版明确映射，不做字符串猜测）
    model_arg = RUNTIME_MODEL_ARG.get(runtime_key)
    if not model_arg:
        raise MyappException(
            'Runtime %s 未配置模型参数映射（RUNTIME_MODEL_ARG），请联系管理员。' % runtime_key
        )
    model = str(task_args.get(model_arg, '') or '').strip()
    if not model:
        raise MyappException('任务缺少模型参数 %s，无法解析 %s Runtime 镜像。' % (model_arg, runtime_key))
    model_key, legacy_keys = resolve_model_key(task, task_args, model=model, runtime_key=runtime_key)
    if not model_key:
        raise MyappException(
            '模型参数 %s=%s 无法识别有效模型名，无法解析 %s Runtime 镜像。' % (model_arg, model, runtime_key))

    scene = RUNTIME_SCENE.get(runtime_key)
    if not scene:
        raise MyappException(
            'Runtime %s 未配置使用场景（RUNTIME_SCENE），请联系管理员。' % runtime_key
        )

    # 传入解析后的模型标识：血缘解析/归一化只用于 Runtime 匹配，
    # task.args 中的 --model_name 等原始参数绝不被替换（训练容器仍加载 PVC 路径）
    image = resolve_runtime_image(model_key, runtime_key, scene, legacy_keys=legacy_keys)
    # 保存任务实际使用的镜像（历史任务可追溯，不随 mapping 变化）
    task.runtime_image = image
    db.session.commit()
    return image
