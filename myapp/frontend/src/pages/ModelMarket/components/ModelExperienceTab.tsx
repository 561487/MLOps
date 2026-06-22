// 在线体验 Tab — 根据 input/output schema 渲染

import React, { useState, useCallback } from 'react';
import { Button, Upload, Spin, message, Table } from 'antd';
import { UploadOutlined, SendOutlined } from '@ant-design/icons';
import { inferModel } from '../api';
import { IModelMarketItem, IInferResponse } from '../types';

interface Props { model: IModelMarketItem; }

const ModelExperienceTab: React.FC<Props> = ({ model }) => {
  const [fileList, setFileList] = useState<any[]>([]);
  const [textInput, setTextInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<IInferResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const isVision = model.category === 'vision';

  const handleInfer = useCallback(async () => {
    setLoading(true); setError(null); setResult(null);
    try {
      const fileObjs = fileList.filter(f => f.originFileObj).map(f => f.originFileObj as File);
      const inputs: Record<string, any> = {};
      if (model.input_schema) {
        model.input_schema.forEach(s => {
          if (s.type === 'text') inputs[s.name] = textInput || 'test input';
        });
      }
      // Preview
      if (fileObjs.length > 0) {
        const url = URL.createObjectURL(fileObjs[0]);
        setPreviewUrl(url);
      }
      const res = await inferModel(model.id, { inputs, files: fileObjs.length > 0 ? fileObjs : undefined });
      setResult(res);
    } catch (err: any) {
      setError(err.message || '推理请求失败');
      message.error('推理失败');
    } finally { setLoading(false); }
  }, [model, fileList, textInput]);

  const resultTitle = isVision ? '检测结果' : '推理结果';

  return (
    <div>
      {/* Input area */}
      <div style={{ marginBottom: 20 }}>
        {model.input_schema?.map(schema => (
          <div key={schema.name} style={{ marginBottom: 12 }}>
            <label style={{ display: 'block', marginBottom: 4, color: '#555', fontSize: 13, fontWeight: 500 }}>
              {schema.label}{schema.required && <span style={{ color: '#ff4d4f' }}> *</span>}
            </label>
            {schema.type === 'file' || schema.type === 'image' ? (
              <Upload listType="picture" fileList={fileList}
                onChange={({ fileList: fl }: any) => setFileList(fl)}
                beforeUpload={() => false} maxCount={1} accept={schema.accept}>
                <Button icon={<UploadOutlined />}>{schema.label}</Button>
              </Upload>
            ) : schema.type === 'text' ? (
              <textarea value={textInput} onChange={e => setTextInput(e.target.value)}
                placeholder={schema.placeholder || schema.label} rows={4}
                style={{ width: '100%', padding: '8px 12px', border: '1px solid #d9d9d9', borderRadius: 4, fontSize: 13, resize: 'vertical' }} />
            ) : null}
          </div>
        ))}
      </div>

      <Button icon={<SendOutlined />} onClick={handleInfer} loading={loading}
        style={{ color: '#0369a1', background: '#e0f2fe', border: '1px solid #bae6fd', borderRadius: 4 }}>
        运行推理
      </Button>

      {loading && <div style={{ marginTop: 20 }}><Spin /></div>}

      {error && <div style={{ marginTop: 16, padding: 12, background: '#fff2f0', border: '1px solid #ffccc7', borderRadius: 4, color: '#cf1322', fontSize: 13 }}>{error}</div>}

      {result && !loading && (
        <div style={{ marginTop: 20 }}>
          <h4 style={{ fontSize: 14, color: '#333', marginBottom: 12 }}>{resultTitle}</h4>

          {previewUrl && <img src={previewUrl} alt="preview" style={{ maxWidth: 300, maxHeight: 300, marginBottom: 12, border: '1px solid #f0f0f0', borderRadius: 4 }} />}

          {result.source && (
            <div style={{ fontSize: 12, color: '#888', marginBottom: 8 }}>
              source: {result.source}
              {result.source === 'backend_placeholder' && <span style={{ color: '#faad14', marginLeft: 8 }}>(backend placeholder)</span>}
            </div>
          )}

          {result.result?.predictions ? (
            <Table dataSource={result.result.predictions.map((p: any, i: number) => ({ ...p, key: i }))}
              columns={[
                { title: '类别', dataIndex: 'label', key: 'label' },
                { title: '置信度', dataIndex: 'score', key: 'score', render: (v: number) => v != null ? `${(v * 100).toFixed(1)}%` : '—' },
                { title: 'bbox', dataIndex: 'bbox', key: 'bbox', render: (v: number[]) => v ? `[${v.join(', ')}]` : '—' },
              ]}
              size="small" pagination={false} style={{ marginBottom: 12 }}
            />
          ) : result.result ? (
            <pre style={{ color: '#333', fontSize: 12, background: '#fafafa', padding: 12, borderRadius: 4, border: '1px solid #f0f0f0', whiteSpace: 'pre-wrap' }}>
              {JSON.stringify(result.result, null, 2)}
            </pre>
          ) : null}

          {/* Raw JSON toggle */}
          {result.result?.predictions && (
            <details style={{ marginTop: 8 }}>
              <summary style={{ cursor: 'pointer', color: '#888', fontSize: 12 }}>原始 JSON</summary>
              <pre style={{ color: '#555', fontSize: 11, background: '#fafafa', padding: 8, borderRadius: 4, border: '1px solid #f0f0f0', whiteSpace: 'pre-wrap', marginTop: 4 }}>
                {JSON.stringify(result, null, 2)}
              </pre>
            </details>
          )}
        </div>
      )}
    </div>
  );
};

export default ModelExperienceTab;
