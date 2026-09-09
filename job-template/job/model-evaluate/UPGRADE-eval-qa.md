# 基于 master 的 eval_qa / LoRA 增量开发

基线：cfbabee（2026-09-08 检查的 origin/master）。保留主分支公共数据集、依赖版本、结果聚合及旧自定义 OpenCompass 流程；不使用 evaluation_scope/public_datasets 新接口。

## 表单

- 公共评测：填写 datasets，自定义路径留空。
- Convert 数据评测：datasets 清空，custom_dataset_path 指向 converted.jsonl 或含 dataset_manifest.json 的目录。
- model_path 可填写完整模型或 LoRA checkpoint。LoRA 从 adapter_config.json / args.json 解析基础模型；无法定位则报错。公共评测通过子进程挂载 adapter，自定义标准评测通过 PEFT 加载。
- max_samples、max_seq_len、max_out_len、num_gpus、model_type、model_kwargs 在标准评测中生效。
- 标准评测暂为逐条推理，batch_size 不生效并输出提示；few_shot 必须为 0。
- custom_columns/custom_metric/custom_prompt_template 是旧自定义 OpenCompass 参数；标准 eval_qa 下须留空，不静默忽略。

## 数据契约

目录 manifest 必须声明 target_schema/format_type=eval_qa，并以 output_file/data_file 指向目录内数据文件。无 manifest 的文件以 input、target、type 识别标准结构；缺少此标识的旧文件仍走原流程。训练 messages manifest 明确拒绝。

普通问答为 short_answer（归一化完全匹配及 token F1）；单选 choice（标签准确率）；多选 multi_choice（标签集合完全匹配，不计顺序，少选多选都错）；兼容 numeric 和 contains。

只将 input 和 choices 发送模型；target 仅用于评分。原有预览、manifest、拒绝记录不作为样本读取。样本校验失败记录到 custom_invalid.jsonl；零有效记录失败；推理异常不标记成功。

## 输出

沿用主分支 output_path；输出 metric.json、eval_summary.json、eval_report.csv，并增加 custom_summary.json、custom_details.jsonl、custom_invalid.jsonl、task_status.json。平台分数为百分制，详细报告 pass_rate 为 0–1。

## 发布边界

本轮只开发及 CPU/桩测试，未运行真实模型 GPU 推理、未构建/推送、未刷新数据库、未提交。源码模板恢复主分支表单，旧数据库新表单需在部署时一起刷新。build.sh 要求显式传入 IMAGE_VERSION，发布前检查 Harbor 标签未被使用，不覆盖 r11/r12。基础镜像继续使用 master 的 base-py310-cu128-r1。
