import axios from 'axios';
import React, { useEffect, useState } from 'react';
import { Alert, Button, Card, Checkbox, Form, Input, InputNumber, message, Space, Switch, Table, Tag, Typography } from 'antd';

const API = '/api/notification';
const eventOptions = [
  { label: 'Workflow 成功', value: 'workflow.succeeded' },
  { label: 'Workflow 失败', value: 'workflow.failed' },
];

const NotificationSettings: React.FC = () => {
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [deliveries, setDeliveries] = useState<any[]>([]);
  const load = async () => {
    const [configRes, deliveryRes] = await Promise.all([axios.get(`${API}/dingtalk`), axios.get(`${API}/deliveries`)]);
    const config = configRes.data.result;
    form.setFieldsValue({ ...config, webhook: '', secret: '' });
    setDeliveries(deliveryRes.data.result || []);
  };
  useEffect(() => { load().catch((e) => message.error(e?.response?.data?.error || '加载通知配置失败')); }, []);
  const save = async () => {
    const values = await form.validateFields();
    setLoading(true);
    try { await axios.put(`${API}/dingtalk`, values); message.success('配置已保存'); await load(); }
    catch (e: any) { message.error(e?.response?.data?.error || '保存失败'); }
    finally { setLoading(false); }
  };
  const test = async () => {
    setLoading(true);
    try { const res = await axios.post(`${API}/dingtalk/test`); message.success(res.data.message); await load(); }
    catch (e: any) { message.error(e?.response?.data?.error || '测试发送失败'); }
    finally { setLoading(false); }
  };
  return <div style={{ padding: 24 }}>
    <Typography.Title level={3}>钉钉任务通知</Typography.Title>
    <Alert showIcon type="info" message="该渠道仅用于 MLOps 主动发送任务状态，不接入灵犀助手或大模型。" style={{ marginBottom: 16 }} />
    <Card title="渠道配置" style={{ marginBottom: 16 }}>
      <Form form={form} layout="vertical" initialValues={{ enabled: false, eventTypes: eventOptions.map(i => i.value), pendingTimeout: 300, silenceSeconds: 1800, resourceMonitorEnabled: false, cpuThreshold: 90, memoryThreshold: 90, gpuThreshold: 95, gpuMemoryThreshold: 90, gpuTemperatureThreshold: 85, thresholdDuration: 120, completionSummary: true }}>
        <Form.Item name="enabled" label="启用通知" valuePropName="checked"><Switch /></Form.Item>
        <Form.Item name="name" label="渠道名称" rules={[{ required: true }]}><Input /></Form.Item>
        <Form.Item name="webhook" label="钉钉群机器人 Webhook" extra="留空表示保留已保存值；页面不会返回原文"><Input.Password placeholder="https://oapi.dingtalk.com/robot/send?..." /></Form.Item>
        <Form.Item name="secret" label="Webhook 加签密钥" extra="未启用加签可留空；留空不会清除已有密钥"><Input.Password placeholder="SEC..." /></Form.Item>
        <Form.Item name="eventTypes" label="通知事件" rules={[{ required: true }]}><Checkbox.Group options={eventOptions} /></Form.Item>
        <Space size="large"><Form.Item name="pendingTimeout" label="Pending 阈值（秒）"><InputNumber min={60} max={86400} /></Form.Item><Form.Item name="silenceSeconds" label="静默时间（秒）"><InputNumber min={0} max={604800} /></Form.Item></Space>
        <Card size="small" title="资源监控" style={{ marginBottom: 16 }}>
          <Space size="large" wrap>
            <Form.Item name="resourceMonitorEnabled" label="启用资源监控" valuePropName="checked"><Switch /></Form.Item>
            <Form.Item name="completionSummary" label="发送完成摘要" valuePropName="checked"><Switch /></Form.Item>
            <Form.Item name="thresholdDuration" label="持续时间（秒）"><InputNumber min={10} max={86400} /></Form.Item>
          </Space>
          <Space size="large" wrap>
            <Form.Item name="cpuThreshold" label="CPU 使用率阈值（%）"><InputNumber min={1} max={100} /></Form.Item>
            <Form.Item name="memoryThreshold" label="内存使用率阈值（%）"><InputNumber min={1} max={100} /></Form.Item>
            <Form.Item name="gpuThreshold" label="GPU 利用率阈值（%）"><InputNumber min={1} max={100} /></Form.Item>
            <Form.Item name="gpuMemoryThreshold" label="GPU 显存阈值（%）"><InputNumber min={1} max={100} /></Form.Item>
            <Form.Item name="gpuTemperatureThreshold" label="GPU 温度阈值（℃）"><InputNumber min={30} max={120} /></Form.Item>
          </Space>
        </Card>
        <Form.Item name="projectScope" label="项目范围"><Input placeholder="逗号分隔；空表示全部" /></Form.Item>
        <Form.Item name="clusterScope" label="集群范围"><Input placeholder="逗号分隔；空表示全部" /></Form.Item>
        <Form.Item name="namespaceScope" label="命名空间范围"><Input placeholder="逗号分隔；空表示全部" /></Form.Item>
        <Space><Button type="primary" loading={loading} onClick={save}>保存</Button><Button loading={loading} onClick={test}>测试发送</Button></Space>
      </Form>
    </Card>
    <Card title="最近发送记录">
      <Table rowKey="id" pagination={false} dataSource={deliveries} columns={[
        { title: '时间', dataIndex: 'createdAt' }, { title: '事件', dataIndex: 'eventType' },
        { title: '状态', dataIndex: 'status', render: (v) => <Tag color={v === 'SUCCESS' ? 'green' : 'red'}>{v}</Tag> },
        { title: '摘要', dataIndex: 'summary' }, { title: '错误', dataIndex: 'error' },
      ]} />
    </Card>
  </div>;
};
export default NotificationSettings;
