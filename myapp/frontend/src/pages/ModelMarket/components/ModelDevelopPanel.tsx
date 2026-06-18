// 一键开发面板 — 创建真实 Notebook（复用平台已有创建逻辑）

import React, { useState, useCallback, useRef, useEffect } from 'react';
import { Form, Input, InputNumber, Button, Switch, Alert, Descriptions, message, Spin } from 'antd';
import { CodeOutlined, LinkOutlined, LoadingOutlined } from '@ant-design/icons';
import { createNotebook, getNotebookStatus } from '../api';
import { IModelMarketItem, INotebookConfig, INotebookResponse } from '../types';

interface Props { model: IModelMarketItem; }

const FAILED_STATUSES = ['Failed', 'ImagePullBackOff', 'ErrImagePull', 'Unschedulable', 'Error'];
const POLL_INTERVAL = 3000;
const POLL_MAX_ATTEMPTS = 60; // 3 minutes

/** Normalize jupyter_url / url before window.open — handles protocol-relative // URLs correctly */
const normalizeOpenUrl = (url?: string | null) => {
  if (!url) return '';

  const trimmed = url.trim();

  // 完整 URL，直接打开
  if (/^https?:\/\//.test(trimmed)) {
    return trimmed;
  }

  // 协议相对 URL，例如 //10.121.177.20/notebook/...
  // 必须补当前协议，不能拼 window.location.origin
  if (trimmed.startsWith('//')) {
    return `${window.location.protocol}${trimmed}`;
  }

  // 普通站内相对路径，例如 /frontend/...
  if (trimmed.startsWith('/')) {
    return `${window.location.origin}${trimmed}`;
  }

  return `${window.location.origin}/${trimmed}`;
};

const ModelDevelopPanel: React.FC<Props> = ({ model }) => {
  const [loading, setLoading] = useState(false);
  const [task, setTask] = useState<INotebookResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [statusMsg, setStatusMsg] = useState('');
  const [polling, setPolling] = useState(false);
  const pollTimer = useRef<any>(null);
  const attemptsRef = useRef(0);
  const openedRef = useRef(false);
  const [form] = Form.useForm();

  // Cleanup timer on unmount
  useEffect(() => {
    return () => { if (pollTimer.current) clearInterval(pollTimer.current); };
  }, []);

  const startPolling = useCallback((notebookId: number) => {
    setPolling(true);
    attemptsRef.current = 0;
    openedRef.current = false;
    pollTimer.current = setInterval(async () => {
      attemptsRef.current += 1;
      try {
        const res = await getNotebookStatus(notebookId);
        const data = res.data || res;
        const msg = data.message || `当前状态: ${data.status || 'Unknown'}`;
        setStatusMsg(msg);

        if (data.ready && !openedRef.current) {
          const targetUrl = data.jupyter_url || data.url;
          const openUrl = normalizeOpenUrl(targetUrl);

          if (openUrl) {
            openedRef.current = true;
            clearInterval(pollTimer.current);
            setPolling(false);
            console.log('[ModelDevelopPanel] auto-open url:', openUrl);
            window.open(openUrl, '_blank');
            message.success('JupyterLab 已打开');
            return;
          }
        }

        if (FAILED_STATUSES.includes(data.status) || FAILED_STATUSES.includes(data.pod_status)) {
          clearInterval(pollTimer.current);
          setPolling(false);
          setError(msg);
          message.error(msg);
          return;
        }

        if (attemptsRef.current >= POLL_MAX_ATTEMPTS) {
          clearInterval(pollTimer.current);
          setPolling(false);
          setError('启动超时，请稍后在 Notebook 页面手动进入');
          message.warning('启动超时，请稍后在 Notebook 页面手动进入');
        }
      } catch (_err) {
        // Silently retry on network errors
      }
    }, POLL_INTERVAL);
  }, []);

  const handleSubmit = useCallback(async () => {
    try {
      const values = await form.validateFields();
      setLoading(true);
      setError(null);
      setStatusMsg('');
      const config: INotebookConfig = {
        name: values.name || undefined,
        image: values.image || model.notebook_image || '',
        python_version: values.python_version || '3.10',
        cuda_version: values.cuda_version || '11.8',
        cpu: values.cpu != null ? values.cpu : 2,
        memory: values.memory != null ? values.memory : 4,
        gpu: values.gpu != null ? values.gpu : 0,
        work_dir: values.work_dir || `/workspace/${model.name}`,
        mount_demo: values.mount_demo ?? true,
      };
      const taskInfo = await createNotebook(model.id, config);
      const notebookData = taskInfo.data || taskInfo;
      setTask(notebookData);
      message.success('Notebook 已创建，正在等待启动...');

      // Start polling for Running status
      if (notebookData.target_id) {
        startPolling(notebookData.target_id);
      }
    } catch (err: any) {
      if (err.errorFields) return;
      const msg = err.response?.data?.message || err.message || '创建失败';
      setError(msg);
      message.error(msg);
    } finally {
      setLoading(false);
    }
  }, [model, form, startPolling]);

  // Reset everything
  const handleReset = useCallback(() => {
    if (pollTimer.current) clearInterval(pollTimer.current);
    setTask(null);
    setError(null);
    setStatusMsg('');
    setPolling(false);
    openedRef.current = false;
  }, []);

  if (task && task.target_id) {
    const targetUrl =
      task.jupyter_url ||
      task.url ||
      `/frontend/dev/dev_online/notebook_shell?id=${task.target_id}`;
    const openUrl = normalizeOpenUrl(targetUrl);
    return (
      <div style={{ padding: 20 }}>
        <Alert
          type={error ? 'error' : polling ? 'info' : 'success'}
          message={error ? '启动失败' : polling ? '等待 Notebook 启动' : 'Notebook 已就绪'}
          description={error || statusMsg || 'JupyterLab 已在新标签页打开'}
          showIcon
          style={{ marginBottom: 16 }}
        />
        <Descriptions size="small" column={1} style={{ marginBottom: 16 }}>
          <Descriptions.Item label="Notebook ID">{task.target_id}</Descriptions.Item>
          <Descriptions.Item label="名称">{task.target_name}</Descriptions.Item>
          {statusMsg && <Descriptions.Item label="状态">{statusMsg}</Descriptions.Item>}
        </Descriptions>
        {polling && (
          <div style={{ marginBottom: 16 }}>
            <Spin indicator={<LoadingOutlined spin />} /> 正在等待 Pod 启动，通常需要 1-3 分钟...
          </div>
        )}
        <Button
          type="primary"
          icon={<LinkOutlined />}
          onClick={() => {
            console.log('[ModelDevelopPanel] manual-open url:', openUrl);
            window.open(openUrl, '_blank');
          }}
          style={{ marginRight: 8 }}
        >
          打开 JupyterLab
        </Button>
        <Button onClick={handleReset} style={{ marginLeft: 8 }}>返回配置</Button>
      </div>
    );
  }

  return (
    <div style={{ maxWidth: 520 }}>
      <p style={{ color: '#666', marginBottom: 16, fontSize: 13 }}>
        基于模型 "{model.display_name}" 创建在线开发环境
      </p>
      <Form form={form} layout="vertical" initialValues={{
        image: model.notebook_image || '',
        python_version: '3.10',
        cuda_version: '11.8', cpu: 2, memory: 4, gpu: 0,
        work_dir: `/workspace/${model.name}`, mount_demo: true,
      }}>
        <Form.Item label="开发环境名称" name="name" rules={[{ required: true }]}>
          <Input placeholder="自动生成或输入自定义名称" />
        </Form.Item>
        <Form.Item label="基础镜像" name="image">
          <Input placeholder={model.notebook_image || '请输入镜像地址'} />
        </Form.Item>
        <div style={{ display: 'flex', gap: 12 }}>
          <Form.Item label="Python" name="python_version"><Input style={{ width: 100 }} /></Form.Item>
          <Form.Item label="CUDA" name="cuda_version"><Input style={{ width: 100 }} /></Form.Item>
        </div>
        <div style={{ display: 'flex', gap: 12 }}>
          <Form.Item label="CPU" name="cpu"><InputNumber min={1} max={64} /></Form.Item>
          <Form.Item label="内存 (GB)" name="memory"><InputNumber min={1} max={512} /></Form.Item>
          <Form.Item label="GPU" name="gpu"><InputNumber min={0} max={8} /></Form.Item>
        </div>
        <Form.Item label="工作目录" name="work_dir"><Input /></Form.Item>
        <Form.Item label="挂载 Demo Notebook" name="mount_demo" valuePropName="checked">
          <Switch />
        </Form.Item>
      </Form>
      {error && <Alert message={error} type="error" showIcon style={{ marginBottom: 16 }} />}
      <Button icon={<CodeOutlined />} onClick={handleSubmit} loading={loading}
        style={{ color: '#15803d', background: '#dcfce7', border: '1px solid #bbf7d0', borderRadius: 4, fontWeight: 500 }}>
        创建开发环境
      </Button>
    </div>
  );
};

export default ModelDevelopPanel;
