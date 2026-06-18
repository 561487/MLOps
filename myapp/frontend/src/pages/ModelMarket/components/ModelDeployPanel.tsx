// 一键部署面板 — 创建/卸载真实 InferenceService，支持部署后状态切换

import React, { useState, useCallback, useEffect } from 'react';
import { Form, Input, InputNumber, Button, Select, Alert, Descriptions, message, Modal, Tag } from 'antd';
import { CloudUploadOutlined, LinkOutlined, DeleteOutlined } from '@ant-design/icons';
import { deployModel, getModelServices, unloadService } from '../api';
import { IModelMarketItem, IDeployConfig, IDeployResponse } from '../types';
import axios from '../../../api/index';

interface Props {
  model: IModelMarketItem;
  onServiceStateChange?: () => void;
}
interface IVersion {
  version: string; model_path: string; source: string; label: string;
  image?: string; command?: string; working_dir?: string;
  deployable?: boolean; reason?: string;
}
interface IActiveService {
  market_service_id: number;
  service_id: number;
  service_name: string;
  model_version: string;
  model_path: string;
  service_status: string;
  ready: boolean;
  endpoint: string;
  api_url: string;
  url: string;
}

const ModelDeployPanel: React.FC<Props> = ({ model, onServiceStateChange }) => {
  const [loading, setLoading] = useState(false);
  const [unloading, setUnloading] = useState(false);
  const [task, setTask] = useState<IDeployResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [versions, setVersions] = useState<IVersion[]>([]);
  const [activeService, setActiveService] = useState<IActiveService | null>(null);
  const [checkingServices, setCheckingServices] = useState(true);
  const [form] = Form.useForm();

  // Load model versions
  useEffect(() => {
    axios.get(`/model_market/api/models/${model.id}/versions`).then((res: any) => {
      const data = res.data || res;
      const inner = data.data || data;
      const vList = inner.versions || [];
      setVersions(vList);
      if (vList.length > 0) {
        form.setFieldsValue({ model_version: vList[0].version });
      }
    }).catch(() => {});
  }, [model.id, form]);

  // Check for existing active service on init and refresh
  const checkActiveService = useCallback(async () => {
    try {
      const result = await getModelServices(model.id, { active_only: true });
      const svcList = result?.services || [];
      if (svcList.length > 0) {
        setActiveService(svcList[0]);
      } else {
        setActiveService(null);
      }
    } catch {
      setActiveService(null);
    } finally {
      setCheckingServices(false);
    }
  }, [model.id]);

  useEffect(() => { checkActiveService(); }, [checkActiveService]);

  const handleSubmit = useCallback(async () => {
    try {
      const values = await form.validateFields();
      setLoading(true);
      setError(null);

      const selectedVersion = versions.find((v: IVersion) => v.version === values.model_version);
      const isFinetune = (values.model_version || '').startsWith('finetune');

      // Validate finetune deployability
      if (isFinetune) {
        if (selectedVersion?.deployable === false) {
          message.error(selectedVersion?.reason || '微调模型未找到可部署权重文件，请开启自动注册模型或手动注册模型后再部署。');
          setLoading(false);
          return;
        }
        if (!selectedVersion?.model_path || !selectedVersion.model_path.endsWith('.pt')) {
          message.error('微调模型未找到可部署权重文件，请开启自动注册模型或手动注册模型后再部署。');
          setLoading(false);
          return;
        }
      }

      // Use version metadata directly (not model defaults)
      const modelPath = selectedVersion?.model_path || (isFinetune ? '' : '/yolov8/yolov8n.pt');
      const deployImage = values.image || selectedVersion?.image || model.inference_image || 'ccr.ccs.tencentyun.com/cube-studio/yolov8:20250801';
      const deployCommand = selectedVersion?.command || 'python server.py';
      const deployWorkdir = selectedVersion?.working_dir || '/yolov8';

      if (isFinetune && !modelPath) {
        message.error('微调模型未找到可部署权重文件，请开启自动注册模型或手动注册模型后再部署。');
        setLoading(false);
        return;
      }

      const config: IDeployConfig = {
        name: values.name || `${model.name}-svc`,
        image: deployImage,
        cpu: values.cpu != null ? values.cpu : 2,
        memory: values.memory != null ? values.memory : 8,
        gpu: values.gpu != null ? values.gpu : 0,
        replicas: values.replicas || 1,
        env_vars: {},
        inference_entry: values.inference_entry || 'predict.py',
        expose_type: values.expose_type || 'gateway',
        enable_experience: values.enable_experience ?? true,
        model_version: values.model_version,
        model_path: modelPath,
        command: deployCommand as any,
        working_dir: deployWorkdir as any,
      };
      const taskInfo = await deployModel(model.id, config);
      console.log('deploy response:', taskInfo);
      if (taskInfo.code && taskInfo.code !== 0) {
        throw new Error(taskInfo.message || '部署失败');
      }
      const deployData = taskInfo.data || taskInfo;
      setTask(deployData);
      message.success('推理服务已创建，正在跳转...');

      // Refresh active service state
      await checkActiveService();
      if (onServiceStateChange) onServiceStateChange();

      // Auto-redirect to service page
      const targetUrl = deployData.url;
      if (targetUrl) {
        const finalUrl = targetUrl.startsWith('http')
          ? targetUrl
          : `${window.location.origin}${targetUrl}`;
        setTimeout(() => { window.open(finalUrl, '_blank'); }, 800);
      }
    } catch (err: any) {
      if (err.errorFields) return;
      const msg = err.response?.data?.message || err.message || '部署失败';
      setError(msg);
      message.error(msg);
    } finally {
      setLoading(false);
    }
  }, [model, form, versions, checkActiveService, onServiceStateChange]);

  // Unload handler
  const handleUnload = useCallback(() => {
    if (!activeService) return;
    Modal.confirm({
      title: '确认卸载推理服务',
      content: '卸载后将停止当前推理服务，卸载完成后可以重新部署。是否继续？',
      okText: '确认卸载',
      okType: 'danger',
      cancelText: '取消',
      onOk: async () => {
        setUnloading(true);
        try {
          const res = await unloadService(activeService.market_service_id);
          if (res.code && res.code !== 0) {
            throw new Error(res.message || '卸载失败');
          }
          message.success('服务已卸载');
          // Clear local state immediately
          setActiveService(null);
          setTask(null);
          // Re-check from backend to confirm no active services remain
          await checkActiveService();
          if (onServiceStateChange) onServiceStateChange();
        } catch (err: any) {
          message.error(err?.response?.data?.message || err?.message || '卸载失败');
        } finally {
          setUnloading(false);
        }
      },
    });
  }, [activeService, checkActiveService, onServiceStateChange]);

  // Navigate to service page
  const goToService = useCallback(() => {
    if (activeService) {
      const serviceUrl = `/frontend/service/model_market_group/service/${activeService.market_service_id}`;
      window.open(serviceUrl, '_blank');
    }
  }, [activeService]);

  // Loading state while checking services
  if (checkingServices) {
    return <div style={{ padding: 20, textAlign: 'center', color: '#999' }}>检查服务状态...</div>;
  }

  // --- ACTIVE SERVICE VIEW ---
  if (activeService) {
    const statusColor = activeService.ready ? '#52c41a' : activeService.service_status === 'Failed' ? '#ff4d4f' : '#faad14';
    const isDeploying = !activeService.ready && ['created', 'Pending', 'Deploying', 'deploying'].includes(activeService.service_status);
    const alertType = activeService.ready ? 'success' : isDeploying ? 'info' : 'warning';
    const alertTitle = activeService.ready
      ? '该模型版本已部署推理服务'
      : isDeploying
        ? '推理服务正在部署中'
        : '推理服务状态异常';
    const alertDesc = activeService.ready
      ? '您可以前往推理服务页面进行推理体验，或卸载后重新部署。'
      : isDeploying
        ? '服务正在部署或未就绪，请稍后刷新。若长时间未就绪，请卸载后重新部署。'
        : '服务状态异常，建议卸载后重新部署。';
    return (
      <div style={{ padding: 16 }}>
        <Alert
          type={alertType}
          message={alertTitle}
          description={alertDesc}
          showIcon
          style={{ marginBottom: 16 }}
        />
        <Descriptions size="small" column={{ xs: 1, sm: 2 }} bordered style={{ marginBottom: 16 }}>
          <Descriptions.Item label="市场服务 ID">{activeService.market_service_id}</Descriptions.Item>
          <Descriptions.Item label="推理服务 ID">{activeService.service_id}</Descriptions.Item>
          <Descriptions.Item label="服务名称">{activeService.service_name}</Descriptions.Item>
          <Descriptions.Item label="模型版本">{activeService.model_version || '-'}</Descriptions.Item>
          <Descriptions.Item label="服务状态">
            <Tag color={statusColor}>{activeService.service_status}</Tag>
          </Descriptions.Item>
          <Descriptions.Item label="Endpoint"><code style={{ fontSize: 11 }}>{activeService.endpoint || '—'}</code></Descriptions.Item>
          <Descriptions.Item label="API 地址" span={2}><code style={{ fontSize: 11 }}>{activeService.api_url}</code></Descriptions.Item>
        </Descriptions>
        <div style={{ display: 'flex', gap: 8 }}>
          <Button type="primary" icon={<LinkOutlined />} onClick={goToService} disabled={!activeService.ready}>
            {activeService.ready ? '前往推理服务页面' : '服务未就绪'}
          </Button>
          <Button danger icon={<DeleteOutlined />} loading={unloading} onClick={handleUnload}>
            一键卸载
          </Button>
        </div>
        {isDeploying && (
          <div style={{ marginTop: 8, color: '#faad14', fontSize: 12 }}>
            提示：Pod 已运行但推理服务尚未就绪。请稍后刷新页面查看状态更新。
          </div>
        )}
      </div>
    );
  }

  // --- TASK SUCCESS (just deployed, no active service yet — fallback) ---
  if (task && task.target_id) {
    const serviceUrl = task.url
      ? (task.url.startsWith('http') ? task.url : `${window.location.origin}${task.url}`)
      : `${window.location.origin}/service/inferenceservice/inferenceservice_manager`;
    return (
      <div style={{ padding: 20 }}>
        <Alert
          type="success"
          message="推理服务已创建"
          description="服务正在部署中。您可以查看服务状态或稍后体验。"
          showIcon
          style={{ marginBottom: 16 }}
        />
        <Descriptions size="small" column={1} style={{ marginBottom: 16 }}>
          <Descriptions.Item label="服务 ID">{task.target_id}</Descriptions.Item>
          <Descriptions.Item label="服务名称">{task.target_name}</Descriptions.Item>
          <Descriptions.Item label="模型版本">{task.model_version}</Descriptions.Item>
          {task.api_url && <Descriptions.Item label="API"><code>{task.api_url}</code></Descriptions.Item>}
        </Descriptions>
        <Button type="primary" icon={<LinkOutlined />} onClick={() => window.open(serviceUrl, '_blank')} style={{ marginRight: 8 }}>
          前往服务页面
        </Button>
        <Button onClick={() => { setTask(null); checkActiveService(); if (onServiceStateChange) onServiceStateChange(); }} style={{ marginLeft: 8 }}>返回配置</Button>
      </div>
    );
  }

  // --- DEPLOY FORM (no active service) ---
  return (
    <div style={{ maxWidth: 520 }}>
      <p style={{ color: '#666', marginBottom: 16, fontSize: 13 }}>将 "{model.display_name}" 部署为推理服务</p>
      <Form form={form} layout="vertical" initialValues={{
        name: `${model.name}-svc`, image: model.inference_image || '', cpu: 2, memory: 8, gpu: 0,
        replicas: 1, inference_entry: 'predict.py', expose_type: 'gateway', enable_experience: true,
      }}>
        <Form.Item label="服务名称" name="name" rules={[{ required: true }]}><Input /></Form.Item>
        <Form.Item label="镜像" name="image"><Input placeholder={model.inference_image || '请输入镜像地址'} /></Form.Item>
        {versions.length > 0 && (
          <Form.Item label="模型版本" name="model_version">
            <Select>
              {versions.map((v: IVersion) => (
                <Select.Option
                  key={v.version}
                  value={v.version}
                  disabled={v.deployable === false}
                >
                  {v.label}{v.deployable === false ? ' (不可部署)' : v.source === 'finetune' ? ' (微调)' : ' (基础)'}
                </Select.Option>
              ))}
            </Select>
          </Form.Item>
        )}
        <div style={{ display: 'flex', gap: 12 }}>
          <Form.Item label="CPU" name="cpu"><InputNumber min={1} /></Form.Item>
          <Form.Item label="内存 (GB)" name="memory"><InputNumber min={1} /></Form.Item>
          <Form.Item label="GPU" name="gpu"><InputNumber min={0} /></Form.Item>
          <Form.Item label="副本数" name="replicas"><InputNumber min={1} max={10} /></Form.Item>
        </div>
        <Form.Item label="推理入口" name="inference_entry"><Input placeholder="predict.py" /></Form.Item>
        <Form.Item label="暴露方式" name="expose_type">
          <Select><Select.Option value="gateway">网关访问</Select.Option><Select.Option value="internal">内部访问</Select.Option></Select>
        </Form.Item>
      </Form>
      {error && <Alert message={error} type="error" showIcon style={{ marginBottom: 16 }} />}
      <Button icon={<CloudUploadOutlined />} onClick={handleSubmit} loading={loading}
        style={{ color: '#6d28d9', background: '#ede9fe', border: '1px solid #ddd6fe', borderRadius: 4, fontWeight: 500 }}>
        确认部署
      </Button>
    </div>
  );
};

export default ModelDeployPanel;
