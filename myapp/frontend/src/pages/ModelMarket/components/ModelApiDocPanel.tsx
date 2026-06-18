// API 文档 Tab — 清晰分区布局

import React from 'react';
import { Tabs, Table } from 'antd';
import { IModelMarketItem } from '../types';

interface Props { model: IModelMarketItem; }

const baseUrl = 'https://api.example.com/v1';

const codePreStyle: React.CSSProperties = {
  background: '#1e1e1e', color: '#d4d4d4', fontFamily: "'Fira Code', 'JetBrains Mono', monospace",
  fontSize: 12, padding: '12px 16px', borderRadius: 4, whiteSpace: 'pre-wrap', overflowX: 'auto',
  maxHeight: 300, overflowY: 'auto', margin: 0,
};

const CodeBlock: React.FC<{ title: string; code: string }> = ({ title, code }) => (
  <div style={{ marginBottom: 16 }}>
    <div style={{ fontSize: 12, color: '#888', marginBottom: 4, fontWeight: 500 }}>{title}</div>
    <pre style={codePreStyle}>{code}</pre>
  </div>
);

const ApiSection: React.FC<{
  method: string; url: string; desc: string;
  curl: string; python: string; response: any;
}> = ({ method, url, desc, curl, python, response }) => (
  <div style={{ background: '#fff', border: '1px solid #f0f0f0', borderRadius: 6, padding: 20, marginBottom: 16 }}>
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
      <span style={{ background: method === 'POST' ? '#1890ff' : '#52c41a', color: '#fff', fontFamily: 'monospace',
        fontSize: 11, fontWeight: 700, padding: '2px 8px', borderRadius: 3 }}>{method}</span>
      <code style={{ fontSize: 13, color: '#333' }}>{url}</code>
    </div>
    <p style={{ color: '#666', fontSize: 13, marginBottom: 12 }}>{desc}</p>
    <CodeBlock title="curl 示例" code={curl} />
    <CodeBlock title="Python 示例" code={python} />
    <CodeBlock title="响应示例" code={JSON.stringify(response, null, 2)} />
  </div>
);

const ModelApiDocPanel: React.FC<Props> = ({ model }) => {
  const inferUrl = `${baseUrl}/api/model-market/${model.id}/infer`;
  const notebookUrl = `${baseUrl}/api/model-market/${model.id}/notebook`;
  const finetuneUrl = `${baseUrl}/api/model-market/${model.id}/finetune`;
  const deployUrl = `${baseUrl}/api/model-market/${model.id}/deploy`;

  const tabs = [
    {
      key: 'infer', label: '推理 API',
      content: <ApiSection method="POST" url={inferUrl} desc="在线推理接口 — 上传输入数据并获取模型预测结果"
        curl={`curl -X POST "${inferUrl}" \\\n  -H "Content-Type: application/json" \\\n  -d '{"inputs": {"image": "base64..."}}'`}
        python={`import requests\n\nurl = "${inferUrl}"\ndata = {"inputs": {"image": "base64..."}}\nresp = requests.post(url, json=data)\nprint(resp.json())`}
        response={{ success: true, result: { predictions: [{ label: 'cat', score: 0.95 }] } }} />,
    },
  ];

  if (model.support_develop) tabs.push({
    key: 'notebook', label: '开发 API',
    content: <ApiSection method="POST" url={notebookUrl} desc="创建在线开发环境"
      curl={`curl -X POST "${notebookUrl}" \\\n  -H "Content-Type: application/json" \\\n  -d '{"name": "${model.name}-dev", "cpu": 4, "gpu": 1}'`}
      python={`import requests\n\nurl = "${notebookUrl}"\ndata = {"name": "${model.name}-dev", "cpu": 4}\nresp = requests.post(url, json=data)\nprint(resp.json())`}
      response={{ task_id: 'nb-xxx', status: 'created', message: 'Notebook created' }} />,
  });

  if (model.support_finetune) tabs.push({
    key: 'finetune', label: '微调 API',
    content: <ApiSection method="POST" url={finetuneUrl} desc="创建微调训练任务"
      curl={`curl -X POST "${finetuneUrl}" \\\n  -H "Content-Type: application/json" \\\n  -d '{"dataset_id": 1, "epochs": 10}'`}
      python={`import requests\n\nurl = "${finetuneUrl}"\ndata = {"dataset_id": 1, "epochs": 10}\nresp = requests.post(url, json=data)\nprint(resp.json())`}
      response={{ task_id: 'ft-xxx', status: 'created', message: 'Finetune task created' }} />,
  });

  if (model.support_deploy) tabs.push({
    key: 'deploy', label: '部署 API',
    content: <ApiSection method="POST" url={deployUrl} desc="部署推理服务"
      curl={`curl -X POST "${deployUrl}" \\\n  -H "Content-Type: application/json" \\\n  -d '{"name": "${model.name}-svc", "cpu": 2}'`}
      python={`import requests\n\nurl = "${deployUrl}"\ndata = {"name": "${model.name}-svc", "cpu": 2}\nresp = requests.post(url, json=data)\nprint(resp.json())`}
      response={{ task_id: 'deploy-xxx', status: 'created', service_name: `${model.name}-svc` }} />,
  });

  return (
    <Tabs tabPosition="top" items={tabs.map(t => ({ key: t.key, label: t.label, children: t.content }))} />
  );
};

export default ModelApiDocPanel;
