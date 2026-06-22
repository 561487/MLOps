// 模型介绍 Tab

import React from 'react';
import { Descriptions, Tag } from 'antd';
import { IModelMarketItem } from '../types';

interface Props { model: IModelMarketItem; }

const catLabels: Record<string, string> = {
  vision: '视觉', audio: '语音', nlp: '自然语言', multimodal: '多模态', llm: '大模型',
};

const ModelOverviewTab: React.FC<Props> = ({ model }) => {
  const pythonVer = model.python_version || '3.10';
  const cudaVer = model.cuda_version || '11.8';
  const primaryImage = model.image || model.inference_image || model.finetune_image || model.notebook_image || '未配置';

  return (
    <div style={{ padding: '8px 0' }}>
      <h3 style={{ fontSize: 15, fontWeight: 600, color: '#333', marginBottom: 12 }}>模型简介</h3>
      <p style={{ color: '#555', fontSize: 14, lineHeight: '24px', marginBottom: 24 }}>
        {model.description}
      </p>

      <h3 style={{ fontSize: 15, fontWeight: 600, color: '#333', marginBottom: 12 }}>技术规格</h3>
      <Descriptions size="small" column={{ xs: 1, sm: 2 }} bordered
        labelStyle={{ color: '#888', fontWeight: 500 }} contentStyle={{ color: '#333' }}>
        <Descriptions.Item label="模型分类">{catLabels[model.category] || model.category}</Descriptions.Item>
        <Descriptions.Item label="任务类型">{model.task_type || '—'}</Descriptions.Item>
        <Descriptions.Item label="框架">{model.framework || '—'}</Descriptions.Item>
        <Descriptions.Item label="Python 版本">{pythonVer}</Descriptions.Item>
        <Descriptions.Item label="CUDA 版本">{cudaVer}</Descriptions.Item>
        <Descriptions.Item label="Docker 镜像"><code style={{ fontSize: 11 }}>{primaryImage}</code></Descriptions.Item>
      </Descriptions>

      {/* Per-function images */}
      {(model.notebook_image || model.finetune_image || model.inference_image) && (
        <>
          <h3 style={{ fontSize: 15, fontWeight: 600, color: '#333', margin: '24px 0 12px' }}>使用环境</h3>
          <Descriptions size="small" column={1} bordered
            labelStyle={{ color: '#888', fontWeight: 500 }} contentStyle={{ color: '#333' }}>
            {model.notebook_image && <Descriptions.Item label="开发镜像"><code style={{ fontSize: 11 }}>{model.notebook_image}</code></Descriptions.Item>}
            {model.finetune_image && <Descriptions.Item label="微调镜像"><code style={{ fontSize: 11 }}>{model.finetune_image}</code></Descriptions.Item>}
            {model.inference_image && <Descriptions.Item label="推理镜像"><code style={{ fontSize: 11 }}>{model.inference_image}</code></Descriptions.Item>}
          </Descriptions>
        </>
      )}

      <h3 style={{ fontSize: 15, fontWeight: 600, color: '#333', margin: '24px 0 12px' }}>支持能力</h3>
      <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
        {[
          { ok: model.support_develop, label: '一键开发' },
          { ok: model.support_finetune, label: '一键微调' },
          { ok: model.support_deploy, label: '一键部署' },
        ].map(cap => (
          <Tag key={cap.label} color={cap.ok ? 'green' : 'default'}>{cap.label}: {cap.ok ? '支持' : '暂不支持'}</Tag>
        ))}
      </div>

      {model.input_schema && model.input_schema.length > 0 && (
        <>
          <h3 style={{ fontSize: 15, fontWeight: 600, color: '#333', margin: '24px 0 12px' }}>输入 / 输出</h3>
          <Descriptions size="small" column={1} bordered
            labelStyle={{ color: '#888' }} contentStyle={{ color: '#333' }}>
            <Descriptions.Item label="输入参数">
              {model.input_schema.map(s => (
                <div key={s.name} style={{ marginBottom: 4 }}>
                  <code>{s.name}</code> ({s.type}) — {s.label}
                  {s.required ? ' *必填' : ''}
                </div>
              ))}
            </Descriptions.Item>
            <Descriptions.Item label="输出格式">
              type: <Tag>{model.output_schema?.type || 'json'}</Tag>
              {model.output_schema?.fields && ` fields: [${model.output_schema.fields.join(', ')}]`}
            </Descriptions.Item>
          </Descriptions>
        </>
      )}
    </div>
  );
};

export default ModelOverviewTab;
