// 一键微调面板 — 创建真实 Pipeline 并触发 Run

import React, { useState, useCallback } from 'react';
import { Form, Input, InputNumber, Button, Switch, Alert, Descriptions, message } from 'antd';
import { ThunderboltOutlined, LinkOutlined } from '@ant-design/icons';
import { createFinetune } from '../api';
import { IModelMarketItem, IFinetuneConfig, IPipelineResponse } from '../types';

interface Props { model: IModelMarketItem; }

const ModelFinetunePanel: React.FC<Props> = ({ model }) => {
  const [loading, setLoading] = useState(false);
  const [task, setTask] = useState<IPipelineResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [form] = Form.useForm();

  const handleSubmit = useCallback(async () => {
    try {
      const values = await form.validateFields();
      setLoading(true);
      setError(null);
      const config: IFinetuneConfig = {
        dataset_id: values.dataset_id != null ? values.dataset_id : 1,
        epochs: values.epochs != null ? values.epochs : 10,
        batch_size: values.batch_size != null ? values.batch_size : 32,
        learning_rate: values.learning_rate != null ? values.learning_rate : 0.001,
        image: values.image || model.finetune_image || model.notebook_image || '',
        cpu: values.cpu != null ? values.cpu : 8,
        gpu: values.gpu != null ? values.gpu : 0,
        memory: values.memory != null ? values.memory : 32,
        output_model_name: values.output_model_name || `${model.name}-finetuned`,
        auto_register: values.auto_register ?? true,
      };
      const taskInfo = await createFinetune(model.id, config);
      console.log('finetune response:', taskInfo);
      // Backend returns {code:0, data:{target_id, url, ...}, message:"..."}
      const finetuneData = taskInfo.data || taskInfo;
      setTask(finetuneData);
      message.success(taskInfo.message || '微调 Pipeline 已创建，正在跳转...');

      // Open the Pipeline DAG page
      if (finetuneData.url) {
        const finalUrl = finetuneData.url.startsWith('http')
          ? finetuneData.url
          : `${window.location.origin}${finetuneData.url}`;
        setTimeout(() => window.open(finalUrl, '_blank'), 800);
      }
    } catch (err: any) {
      if (err.errorFields) return;
      const msg = err.response?.data?.message || err.message || '提交失败';
      setError(msg);
      message.error(msg);
    } finally {
      setLoading(false);
    }
  }, [model, form]);

  if (task && task.target_id) {
    const dagUrl = task.url
      ? (task.url.startsWith('http') ? task.url : `${window.location.origin}${task.url}`)
      : `${window.location.origin}/train/train_template/train_task/pipeline`;
    return (
      <div style={{ padding: 20 }}>
        <Alert
          type="success"
          message="微调 Pipeline 已创建"
          description="Pipeline 任务流页面正在新标签页打开。如未弹出，请点击下方按钮。"
          showIcon
          style={{ marginBottom: 16 }}
        />
        <Descriptions size="small" column={1} style={{ marginBottom: 16 }}>
          <Descriptions.Item label="Pipeline ID">{task.target_id}</Descriptions.Item>
          <Descriptions.Item label="名称">{task.target_name}</Descriptions.Item>
        </Descriptions>
        <Button
          type="primary"
          icon={<LinkOutlined />}
          onClick={() => window.open(dagUrl, '_blank')}
          style={{ marginRight: 8 }}
        >
          打开 Pipeline 任务流
        </Button>
        <Button onClick={() => setTask(null)} style={{ marginLeft: 8 }}>返回配置</Button>
      </div>
    );
  }

  return (
    <div style={{ maxWidth: 520 }}>
      <p style={{ color: '#666', marginBottom: 16, fontSize: 13 }}>基于 "{model.display_name}" 创建微调任务</p>
      <Form form={form} layout="vertical" initialValues={{
        epochs: 10, batch_size: 32, learning_rate: 0.001,
        image: model.finetune_image || model.notebook_image || '',
        cpu: 8, gpu: 0, memory: 32,
        output_model_name: `${model.name}-finetuned`, auto_register: true,
      }}>
        <Form.Item label="基础模型"><Input value={model.display_name} disabled /></Form.Item>
        <Form.Item label="数据集 ID" name="dataset_id"><InputNumber min={1} style={{ width: '100%' }} /></Form.Item>
        <div style={{ display: 'flex', gap: 12 }}>
          <Form.Item label="Epochs" name="epochs"><InputNumber min={1} /></Form.Item>
          <Form.Item label="Batch Size" name="batch_size"><InputNumber min={1} /></Form.Item>
          <Form.Item label="Learning Rate" name="learning_rate"><InputNumber min={0.0001} step={0.0001} /></Form.Item>
        </div>
        <Form.Item label="训练镜像" name="image"><Input placeholder={model.finetune_image || model.notebook_image || ''} /></Form.Item>
        <div style={{ display: 'flex', gap: 12 }}>
          <Form.Item label="CPU" name="cpu"><InputNumber min={1} /></Form.Item>
          <Form.Item label="GPU" name="gpu"><InputNumber min={0} /></Form.Item>
          <Form.Item label="内存 (GB)" name="memory"><InputNumber min={1} /></Form.Item>
        </div>
        <Form.Item label="输出模型名称" name="output_model_name"><Input /></Form.Item>
        <Form.Item label="自动注册模型" name="auto_register" valuePropName="checked"><Switch /></Form.Item>
      </Form>
      {error && <Alert message={error} type="error" showIcon style={{ marginBottom: 16 }} />}
      <Button icon={<ThunderboltOutlined />} onClick={handleSubmit} loading={loading}
        style={{ color: '#b45309', background: '#fef3c7', border: '1px solid #fde68a', borderRadius: 4, fontWeight: 500 }}>
        提交微调任务
      </Button>
    </div>
  );
};

export default ModelFinetunePanel;
