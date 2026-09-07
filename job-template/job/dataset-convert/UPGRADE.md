# DatasetConvert 语义转换升级

本版在原有转换器上增加规则驱动的中英文字段识别、单选和多选支持，不调用大模型猜测字段含义。未知字段或歧义通过人工映射解决。只处理文本，不承担清洗、采样、合并或训练/验证集分割。

## 前端表单

| 分组 | 参数 | 默认值与行为 |
| --- | --- | --- |
| 基础 | input_path | PVC 文件或目录；优先 manifest 指向文件，其次 dataset.jsonl |
| 基础 | target_schema | messages；另可选 eval_qa |
| 基础 | system_prompt | 空；仅 messages 生效，已有 system 优先保留 |
| 基础 | output_dir | 输出目录 |
| 基础 | overwrite | false；不覆盖已有目录 |
| 高级 | field_mapping | 空，自动识别；有歧义时填写源字段到语义字段的 JSON |
| 高级 | answer_encoding | auto；可选 label、text、index0、index1、bool_array |
| 高级 | invalid_policy | reject，保存异常原因并继续；fail 遇错停止 |

输入格式、源结构自动识别；未消费字段默认收进 metadata。前端不再提供 text、输入格式、源结构等选项，但 CLI 保留旧参数兼容。系统提示词仍显示在基础配置中，评测模式下不生效；未新增前端条件渲染机制。

## 字段与答案规则

- 常见问题字段 question/query/problem/problem_text/prompt/input/问题/题目；答案 answer/response/solution/solution_text/output/target/答案/回答。
- context/background/passage/背景拼接在问题前，绝不自动提升为系统提示词。
- 原 Alpaca 的 instruction + input 继续按指令与补充输入拼接，不把 instruction 强行解释为背景。
- 保留 messages、ShareGPT 的角色结构。转评测时以最后一条 assistant 为参考答案，前文构成输入，最后答案不进入输入。
- 手工映射示例：`{"problem_text":"question","solution_text":"answer","background_text":"context"}`。语义字段还支持 system、choices、choice_labels、correct、id、category、type。旧 user/assistant、input/target 映射目标仍可使用。
- 选择题使用 options/choices，标签可来自 option_ids。`P)` 等标签规范化为 `P`，不强制重编为 A。
- 布尔数组必须与选项等长且至少一个 true；一个答案为 choice，多个为 multi_choice。明确多选题允许只有一个正确项；明确单选题不能有多个正确项。
- 数字索引必须指定 index0/index1，避免 0/1 起始误判。标签数组和逗号分隔标签支持多选；不猜测自由文本中的答案。
- 同一语义出现多个候选字段时拒绝；显式映射 answer 或 correct 可消除两者并存歧义。

## 输出契约

统一 UTF-8 JSONL：`converted.jsonl`；附 `dataset_manifest.json`、前三条 `conversion_preview.json`，有异常时附 `rejected.jsonl`。有效样本为零必须失败。

训练格式：messages 包含可选 system、user、assistant；选择题将选项拼接进 user，多选 assistant 如 `A, C`。

评测格式示例：

```json
{"input":"哪些部件异常？","target":["A","C"],"type":"multi_choice","choices":[{"label":"A","text":"泵"},{"label":"B","text":"阀"},{"label":"C","text":"电机"}]}
```

单选 target 为标签字符串；普通问答 target 为答案字符串。支持 short_answer、numeric、contains。参考答案只能用于评分，不拼入当前问题。

配套 ModelEvaluate 增加 multi_choice 集合完全匹配：顺序无关，少选、多选均不通过；最后一行采用 `Answer: A, C`。目录输入读取 manifest 指定文件，避免把预览、拒绝记录和报告当评测数据。

## 验证与上线

回归脚本放在服务器 `.test/convert-upgrade-20260907/`，不放入待提交测试目录。覆盖英文 FailureSensorIQ 同结构十选项布尔数组、中文问答、单选多选、字段覆盖、索引歧义、旧消息/Alpaca、manifest 读取和评分。

FailureSensorIQ 全量 PVC 数据及真实模型推理尚需流水线联调；规则测试不能替代 GPU 评测。当前仅更新源码与模板定义，镜像标签未变，未构建/推送、未刷新数据库、未提交。上线需分别部署 DatasetConvert 和 ModelEvaluate 新镜像并刷新模板；已有流水线节点需要更新参数和镜像引用。
