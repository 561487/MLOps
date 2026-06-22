// 任务状态展示组件 — 浅色风格

import React, { useState, useEffect, useRef } from 'react';
import { Button, Steps, Alert, Spin } from 'antd';
import { CheckCircleFilled, LoadingOutlined, ArrowLeftOutlined, LinkOutlined } from '@ant-design/icons';
import { getTaskStatus, getTaskLogs } from '../api';
import { ITaskInfo } from '../types';

interface Props {
  task: ITaskInfo;
  onBack: () => void;
  successMessage: string;
  successAction?: { label: string; url: string };
  children?: React.ReactNode;
}

const typeLabels: Record<string, string> = { notebook: '开发环境', finetune: '微调训练', deploy: '模型部署' };

const ModelTaskStatus: React.FC<Props> = ({ task: initTask, onBack, successMessage, successAction, children }) => {
  const [taskInfo, setTaskInfo] = useState<ITaskInfo>(initTask);
  const [logs, setLogs] = useState<string[]>([]);
  const [loadingLogs, setLoadingLogs] = useState(false);
  const intervalRef = useRef<any>(null);

  useEffect(() => {
    if (taskInfo.status === 'completed' || taskInfo.status === 'failed') return;
    intervalRef.current = setInterval(async () => {
      try {
        const updated = await getTaskStatus(taskInfo.id);
        setTaskInfo(updated);
        if (updated.status === 'completed' || updated.status === 'failed') {
          clearInterval(intervalRef.current);
        }
      } catch { /* silent */ }
    }, 3000);
    return () => clearInterval(intervalRef.current);
  }, [taskInfo.id, taskInfo.status]);

  const fetchLogs = async () => {
    setLoadingLogs(true);
    try { const r = await getTaskLogs(taskInfo.id); setLogs(r.logs || []); }
    catch { setLogs([taskInfo.message || '暂无日志']); }
    finally { setLoadingLogs(false); }
  };

  const statusColor = taskInfo.status === 'completed' ? '#52c41a' : taskInfo.status === 'failed' ? '#ff4d4f' : '#1890ff';

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
        <h4 style={{ margin: 0, fontSize: 15, color: '#333' }}>{typeLabels[taskInfo.type] || '任务'} 状态</h4>
        <Button icon={<ArrowLeftOutlined />} onClick={onBack} size="small">返回</Button>
      </div>

      <div style={{ background: '#fafafa', border: '1px solid #f0f0f0', borderRadius: 6, padding: 16, marginBottom: 16 }}>
        <div style={{ display: 'flex', gap: 12, marginBottom: 6, fontSize: 13 }}>
          <span style={{ color: '#888' }}>ID:</span><code>{taskInfo.id}</code>
          <span style={{ color: '#888', marginLeft: 16 }}>状态:</span>
          <span style={{ color: statusColor, fontWeight: 600 }}>● {taskInfo.status?.toUpperCase()}</span>
        </div>
        {taskInfo.message && <p style={{ color: '#666', fontSize: 13, margin: 0 }}>{taskInfo.message}</p>}
      </div>

      {taskInfo.status === 'completed' && (
        <Alert type="success" message={successMessage} showIcon style={{ marginBottom: 16 }}
          action={successAction ? <a href={successAction.url}><LinkOutlined /> {successAction.label}</a> : undefined} />
      )}
      {taskInfo.status === 'failed' && <Alert type="error" message="任务执行失败" description={taskInfo.message} showIcon style={{ marginBottom: 16 }} />}
      {children}

      <Button onClick={fetchLogs} loading={loadingLogs} size="small" style={{ marginTop: 8 }}>查看日志</Button>
      {logs.length > 0 && (
        <pre style={{ marginTop: 12, color: '#d4d4d4', background: '#1e1e1e', fontSize: 11, padding: 12, borderRadius: 4, maxHeight: 240, overflowY: 'auto', fontFamily: 'monospace' }}>
          {logs.join('\n')}
        </pre>
      )}
      {(taskInfo.status === 'running' || taskInfo.status === 'pending') && <div style={{ textAlign: 'center', padding: 20 }}><Spin /></div>}
    </div>
  );
};

export default ModelTaskStatus;
