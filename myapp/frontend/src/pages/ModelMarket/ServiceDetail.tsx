// 模型市场推理服务体验页 - 支持目标检测和语音识别
import React, { useEffect, useState, useCallback, useRef } from 'react';
import { useParams } from 'react-router-dom';
import {
  Tabs, Button, Upload, Descriptions, Alert, Spin,
  message, Card, Tag, Statistic, Row, Col, Modal, Select
} from 'antd';
import { CloudUploadOutlined, CopyOutlined, PlayCircleOutlined, DeleteOutlined } from '@ant-design/icons';
import axios from '../../api/index';

interface ServiceInfo {
  market_service_id?: number;
  service_id?: number;
  service_name?: string;
  model_id?: number;
  model_name?: string;
  display_name?: string;
  task_type?: string;
  demo_input_type?: string;
  demo_output_type?: string;
  model_version?: string;
  model_path?: string;
  status?: string;
  ready?: boolean;
  endpoint?: string;
  predict_url?: string;
  api_url?: string;
  api_example_url?: string;
  stats_url?: string;
  message?: string;
}

interface StatsInfo {
  call_count: number;
  success_count: number;
  failed_count: number;
  avg_latency_ms: number;
  last_called_at: string | null;
}

interface DetectionBox {
  x1: number; y1: number; x2: number; y2: number;
  label: string; score: number;
}

const ServiceDetail: React.FC = () => {
  const { marketServiceId } = useParams<{ marketServiceId: string }>();
  const [service, setService] = useState<ServiceInfo | null>(null);
  const [stats, setStats] = useState<StatsInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [predicting, setPredicting] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [rawDetections, setRawDetections] = useState<DetectionBox[]>([]);
  const [confidenceThreshold, setConfidenceThreshold] = useState<number>(0.5);
  const [latency, setLatency] = useState<number>(0);
  const [imageUrl, setImageUrl] = useState<string>('');
  const [fileObj, setFileObj] = useState<File | null>(null);
  const [audioLanguage, setAudioLanguage] = useState<string>('auto');
  const [apiExample, setApiExample] = useState<any>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const isAudio = service?.task_type === 'speech_recognition' || service?.demo_input_type === 'audio';

  const loadService = useCallback(async () => {
    try {
      const res = await (axios.get(`/model_market/api/services/${marketServiceId}/status`) as any);
      const data = (res.data || res).data || (res.data || res);
      setService(data);
    } catch (e) { /* ignore */ }
    setLoading(false);
  }, [marketServiceId]);

  const loadStats = useCallback(async () => {
    try {
      const res = await (axios.get(`/model_market/api/services/${marketServiceId}/stats`) as any);
      setStats((res.data || res).data || (res.data || res));
    } catch (e) { /* ignore */ }
  }, [marketServiceId]);

  const loadApiExample = useCallback(async () => {
    try {
      const res = await (axios.get(`/model_market/api/services/${marketServiceId}/api-example`) as any);
      setApiExample((res.data || res).data || (res.data || res));
    } catch (e) { /* ignore */ }
  }, [marketServiceId]);

  useEffect(() => {
    if (!marketServiceId) return;
    if (!predicting) {
      loadService();
      loadStats();
    }
    loadApiExample();
    const timer = predicting ? undefined : setInterval(() => { loadService(); loadStats(); }, 5000);
    return () => { if (timer) clearInterval(timer); };
  }, [marketServiceId, loadService, loadStats, loadApiExample, predicting]);

  // Extract detection boxes from result, handling nested data structures
  const extractDetections = useCallback((resultData: any): DetectionBox[] => {
    if (!resultData) return [];
    // Handle various nesting levels: result.labels, data.result.labels, result.result.labels, etc.
    let obj = resultData;
    // Unwrap 'result' key if present
    if (obj.result && typeof obj.result === 'object' && !Array.isArray(obj.result)) {
      obj = obj.result;
    }
    // Unwrap 'data' key if present
    if (obj.data && typeof obj.data === 'object' && !Array.isArray(obj.data)) {
      obj = obj.data;
    }
    // Try again after data unwrapping
    if (obj.result && typeof obj.result === 'object' && !Array.isArray(obj.result)) {
      obj = obj.result;
    }

    const labels: string[] = obj.labels || [];
    const scores: number[] = obj.scores || [];
    // The real service returns "xywhns" (normalized center xywh), backend also maps to "xywhs"
    const xywhs: number[][] = obj.xywhns || obj.xywhs || [];
    const origShape: number[] = obj.orig_shape || [];

    if (!labels.length || !xywhs.length) return [];

    const boxes: DetectionBox[] = [];
    for (let i = 0; i < Math.min(labels.length, xywhs.length); i++) {
      const [cx, cy, w, h] = xywhs[i];
      const score = scores[i] || 0;
      boxes.push({
        x1: cx - w / 2,  // normalized center → top-left
        y1: cy - h / 2,
        x2: cx + w / 2,
        y2: cy + h / 2,
        label: labels[i],
        score: score,
      });
    }
    return boxes;
  }, []);

  // Filtered detections by confidence threshold (derived state, computed early for drawBoxes)
  const filteredDetections = rawDetections.filter(d => d.score >= confidenceThreshold);

  // Draw detection boxes on canvas overlay
  const drawBoxes = useCallback(() => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img || !img.complete) return;

    const displayW = img.clientWidth;
    const displayH = img.clientHeight;
    if (!displayW || !displayH) return;

    canvas.width = displayW;
    canvas.height = displayH;
    canvas.style.width = displayW + 'px';
    canvas.style.height = displayH + 'px';

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, displayW, displayH);

    const colors = ['#FF4444', '#44FF44', '#4488FF', '#FFAA00', '#FF44FF',
                    '#44FFFF', '#FFFF44', '#FF8844', '#88FF44', '#4444FF'];

    filteredDetections.forEach((box, idx) => {
      // Convert normalized [0-1] coords to display pixels
      const x1 = box.x1 * displayW;
      const y1 = box.y1 * displayH;
      const x2 = box.x2 * displayW;
      const y2 = box.y2 * displayH;
      const w = x2 - x1;
      const h = y2 - y1;

      if (w <= 0 || h <= 0) return;

      const color = colors[idx % colors.length];

      // Draw rectangle
      ctx.strokeStyle = color;
      ctx.lineWidth = Math.max(2, displayW / 300);
      ctx.strokeRect(x1, y1, w, h);

      // Draw label background
      const text = `${box.label} ${(box.score * 100).toFixed(1)}%`;
      ctx.font = `${Math.max(12, displayW / 40)}px Arial`;
      const textMetrics = ctx.measureText(text);
      const textW = textMetrics.width;
      const textH = Math.max(14, displayW / 40) + 4;

      ctx.fillStyle = color;
      ctx.fillRect(x1, Math.max(0, y1 - textH), textW + 6, textH);

      // Draw label text
      ctx.fillStyle = '#FFFFFF';
      ctx.fillText(text, x1 + 3, Math.max(textH - 4, y1 - 2));
    });
  }, [filteredDetections]);

  // Re-draw when filtered detections change or image loads
  useEffect(() => {
    drawBoxes();
    // Also listen for window resize
    window.addEventListener('resize', drawBoxes);
    return () => window.removeEventListener('resize', drawBoxes);
  }, [drawBoxes]);

  const handlePredict = useCallback(async () => {
    if (!fileObj) { message.warning(isAudio ? '请先上传音频' : '请先上传图片'); return; }
    setPredicting(true);
    setResult(null);
    setRawDetections([]);
    try {
      const formData = new FormData();
      formData.append('file', fileObj);
      if (isAudio) {
        formData.append('language', audioLanguage);
        formData.append('timestamps', 'true');
      }
      const start = Date.now();
      const res = await (axios.post(`/model_market/api/services/${marketServiceId}/predict`, formData, {
        timeout: isAudio ? 300000 : 30000,
      }) as any);
      setLatency(Date.now() - start);
      const body = (res.data || res);
      const inner = body.data || body;
      if (body.code && body.code !== 0) {
        throw new Error(body.message || '推理失败');
      }
      // inner contains: { result, latency_ms, endpoint, predict_url, message }
      const predictionResult = inner.result || inner;
      setResult(predictionResult);
      setRawDetections(isAudio ? [] : extractDetections(inner));
      if (inner.message && (isAudio || inner.message.indexOf('未检测到目标') >= 0)) {
        message.info(inner.message);
      }
      loadStats();
    } catch (e: any) {
      message.error(e?.response?.data?.message || e?.message || '推理失败');
    } finally {
      setPredicting(false);
    }
  }, [marketServiceId, fileObj, loadStats, extractDetections, isAudio, audioLanguage]);

  const handleUpload = useCallback((file: File) => {
    setFileObj(file);
    setResult(null);
    setRawDetections([]);
    const reader = new FileReader();
    reader.onload = (e) => {
      const dataUrl = e.target?.result as string;
      setImageUrl(dataUrl);
    };
    reader.readAsDataURL(file);
    return false;
  }, []);

  const handleImageLoad = useCallback(() => {
    drawBoxes();
  }, [drawBoxes]);

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text).then(() => message.success('已复制'));
  };

  const [unloading, setUnloading] = useState(false);
  const handleUnload = useCallback(() => {
    Modal.confirm({
      title: '确认卸载推理服务',
      content: '卸载后将停止当前推理服务，卸载完成后可以重新部署。是否继续？',
      okText: '确认卸载',
      okType: 'danger',
      cancelText: '取消',
      onOk: async () => {
        setUnloading(true);
        try {
          const res = await (axios.post(`/model_market/api/services/${marketServiceId}/unload`, {}, { timeout: 30000 }) as any);
          const body = res.data || res;
          if (body.code && body.code !== 0) {
            throw new Error(body.message || '卸载失败');
          }
          message.success('服务已卸载');
          setTimeout(() => {
            const modelId = service?.model_id;
            window.location.href = modelId ? `/frontend/service/model_market_group/detail/${modelId}?tab=deploy` : '/frontend/service/model_market_group';
          }, 1000);
        } catch (err: any) {
          message.error(err?.response?.data?.message || err?.message || '卸载失败');
        } finally {
          setUnloading(false);
        }
      },
    });
  }, [marketServiceId, service]);

  const statusColor = service?.ready ? '#52c41a' : service?.status === 'Failed' ? '#ff4d4f' : '#faad14';
  const statusText = service?.ready ? 'Ready' : (service?.status || 'Unknown');

  const hasRawDetections = rawDetections.length > 0;
  const hasFilteredDetections = filteredDetections.length > 0;
  const modelVersionDisplay = service?.model_version && service.model_version !== '-'
    ? service.model_version
    : '-';

  return (
    <div style={{ padding: 24, maxWidth: 1000, margin: '0 auto' }}>
      {loading ? <Spin size="large" style={{ display: 'block', marginTop: 100 }} /> : (
        <>
          <Card style={{ marginBottom: 16 }}>
            <h2 style={{ margin: 0 }}>{service?.display_name || service?.model_name || '模型'}推理服务</h2>
            <Descriptions size="small" column={{ xs: 1, sm: 2 }} style={{ marginTop: 12 }}>
              <Descriptions.Item label="服务名称">{service?.service_name || '-'}</Descriptions.Item>
              <Descriptions.Item label="模型版本">{modelVersionDisplay}</Descriptions.Item>
              <Descriptions.Item label="服务状态">
                <Tag color={statusColor}>{statusText}</Tag>
                {!service?.ready && service?.message && <span style={{ color: '#999', marginLeft: 8 }}>{service.message}</span>}
              </Descriptions.Item>
              <Descriptions.Item label="市场服务 ID">{marketServiceId}</Descriptions.Item>
              {service?.service_id && <Descriptions.Item label="推理服务 ID">{service.service_id}</Descriptions.Item>}
            </Descriptions>
            <div style={{ marginTop: 12 }}>
              <Button danger icon={<DeleteOutlined />} loading={unloading} onClick={handleUnload}>
                卸载服务
              </Button>
            </div>
          </Card>

          <Row gutter={16} style={{ marginBottom: 16 }}>
            <Col xs={12} sm={6}><Card><Statistic title="总调用" value={stats?.call_count || 0} /></Card></Col>
            <Col xs={12} sm={6}><Card><Statistic title="成功" value={stats?.success_count || 0} valueStyle={{ color: '#52c41a' }} /></Card></Col>
            <Col xs={12} sm={6}><Card><Statistic title="失败" value={stats?.failed_count || 0} valueStyle={{ color: '#ff4d4f' }} /></Card></Col>
            <Col xs={12} sm={6}><Card><Statistic title="平均延迟(ms)" value={stats?.avg_latency_ms || 0} /></Card></Col>
          </Row>

          <Tabs defaultActiveKey="inference" items={[
            {
              key: 'inference', label: '推理',
              children: (
                <div>
                  <div style={{ marginBottom: 8, color: '#888', fontSize: 12 }}>
                    {isAudio
                      ? '支持 WAV、MP3、M4A、FLAC、OGG 和 WebM，单个文件不超过 100 MB、时长不超过 10 分钟。'
                      : '建议上传包含 person / car / dog / bus / bottle 等 COCO 常见目标的图片进行测试。'}
                  </div>
                  {isAudio && (
                    <div style={{ marginBottom: 12 }}>
                      <span style={{ marginRight: 8, color: '#666' }}>音频语言：</span>
                      <Select
                        value={audioLanguage}
                        onChange={setAudioLanguage}
                        style={{ width: 140 }}
                        options={[
                          { value: 'auto', label: '自动识别' },
                          { value: 'zh', label: '中文' },
                          { value: 'en', label: '英语' },
                          { value: 'ja', label: '日语' },
                          { value: 'ko', label: '韩语' },
                        ]}
                      />
                    </div>
                  )}
                  <Upload beforeUpload={handleUpload} showUploadList={false} accept={isAudio ? 'audio/*,.m4a' : 'image/*'}>
                    <Button icon={<CloudUploadOutlined />}>{isAudio ? '上传音频' : '上传图片'}</Button>
                  </Upload>

                  {isAudio && imageUrl && (
                    <div style={{ marginTop: 12 }}>
                      <audio controls src={imageUrl} style={{ width: '100%', maxWidth: 600 }} />
                    </div>
                  )}

                  {!isAudio && imageUrl && (
                    <div style={{ position: 'relative', display: 'inline-block', margin: '12px 0', maxWidth: '100%' }}>
                      <img
                        ref={imgRef}
                        src={imageUrl}
                        alt="uploaded"
                        style={{ maxWidth: 500, maxHeight: 400, display: 'block' }}
                        onLoad={handleImageLoad}
                      />
                      <canvas
                        ref={canvasRef}
                        style={{
                          position: 'absolute', top: 0, left: 0,
                          pointerEvents: 'none', maxWidth: '100%',
                        }}
                      />
                    </div>
                  )}

                  <div style={{ marginTop: 12 }}>
                    <Button type="primary" icon={<PlayCircleOutlined />} loading={predicting}
                      onClick={handlePredict} disabled={!service?.ready}>
                      {service?.ready ? '提交推理' : '服务未就绪'}
                    </Button>
                  </div>

                  {isAudio && result && (
                    <Card title="识别结果" style={{ marginTop: 12 }}>
                      <div style={{ fontSize: 16, lineHeight: 1.8, whiteSpace: 'pre-wrap' }}>
                        {result.text || '未识别到文本'}
                      </div>
                      <div style={{ marginTop: 8, color: '#888', fontSize: 12 }}>
                        语言：{result.language || audioLanguage}；音频时长：{result.duration_seconds ?? '-'} 秒；请求耗时：{latency} ms
                      </div>
                      {Array.isArray(result.segments) && result.segments.length > 0 && (
                        <div style={{ marginTop: 16 }}>
                          <strong>分段时间轴</strong>
                          {result.segments.map((segment: any, index: number) => (
                            <div key={index} style={{ padding: '6px 0', borderBottom: '1px solid #f0f0f0' }}>
                              <code style={{ marginRight: 8 }}>[{segment.start ?? '-'}s - {segment.end ?? '-'}s]</code>
                              {segment.text}
                            </div>
                          ))}
                        </div>
                      )}
                      {result.text && (
                        <Button size="small" icon={<CopyOutlined />} style={{ marginTop: 12 }} onClick={() => copyToClipboard(result.text)}>
                          复制文本
                        </Button>
                      )}
                    </Card>
                  )}

                  {/* Confidence threshold selector */}
                  {!isAudio && result && (
                    <div style={{ marginTop: 12, display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                      <span style={{ fontSize: 13, color: '#666' }}>置信度阈值：</span>
                      <Select
                        size="small"
                        value={confidenceThreshold}
                        onChange={(v: number) => setConfidenceThreshold(v)}
                        style={{ width: 100 }}
                        options={[
                          { value: 0.25, label: '0.25' },
                          { value: 0.5, label: '0.50' },
                          { value: 0.7, label: '0.70' },
                        ]}
                      />
                      {hasRawDetections && (
                        <span style={{ fontSize: 12, color: '#888' }}>
                          （原始 {rawDetections.length} 个，展示 {filteredDetections.length} 个）
                        </span>
                      )}
                    </div>
                  )}

                  {/* No raw detections at all */}
                  {!isAudio && result && !hasRawDetections && (
                    <Alert
                      type="info"
                      showIcon
                      style={{ marginTop: 12 }}
                      message="未检测到目标"
                      description="模型未能在该图片中检测到目标。请尝试上传包含 person / car / dog / bus / bottle 等 COCO 常见目标的图片进行测试。"
                    />
                  )}

                  {/* Has raw detections but all filtered out */}
                  {hasRawDetections && !hasFilteredDetections && (
                    <Alert
                      type="warning"
                      showIcon
                      style={{ marginTop: 12 }}
                      message="检测结果置信度较低，已按阈值过滤"
                      description={`原始检测到 ${rawDetections.length} 个目标，但置信度均低于 ${confidenceThreshold}。可降低阈值或查看下方 JSON 原始结果。`}
                    />
                  )}

                  {/* Has filtered detections — show boxes */}
                  {hasFilteredDetections && (
                    <Alert
                      type="success"
                      showIcon
                      style={{ marginTop: 12 }}
                      message={`检测到 ${filteredDetections.length} 个目标（原始 ${rawDetections.length} 个）`}
                      description={filteredDetections.map((d, i) => (
                        <span key={i} style={{ marginRight: 12 }}>
                          <span style={{
                            display: 'inline-block', width: 10, height: 10, marginRight: 4,
                            background: ['#FF4444', '#44FF44', '#4488FF', '#FFAA00', '#FF44FF',
                                         '#44FFFF', '#FFFF44', '#FF8844'][i % 8],
                          }} />
                          {d.label} ({(d.score * 100).toFixed(1)}%)
                        </span>
                      ))}
                    />
                  )}

                  {/* Low confidence note */}
                  {hasFilteredDetections && (
                    <div style={{ marginTop: 4, color: '#aaa', fontSize: 11 }}>
                      当前仅展示置信度 ≥ {confidenceThreshold} 的检测框，完整结果见下方 JSON。
                    </div>
                  )}

                  {/* Raw JSON result (always show when available) */}
                  {result && (
                    <Card title="推理结果 (JSON)" style={{ marginTop: 12 }}>
                      {latency > 0 && <p>耗时: {latency}ms</p>}
                      <pre style={{ background: '#f5f5f5', padding: 12, borderRadius: 4, overflow: 'auto', maxHeight: 500, fontSize: 12 }}>
                        {JSON.stringify(result, null, 2)}
                      </pre>
                    </Card>
                  )}
                </div>
              )
            },
            {
              key: 'api', label: '接口',
              children: (
                <div>
                  <Card title="API 调用" size="small" style={{ marginBottom: 12 }}>
                    <p><strong>Platform Endpoint:</strong> <code>{apiExample?.endpoint || '-'}</code></p>
                    <p><strong>Method:</strong> POST</p>
                    <p><strong>Content-Type:</strong> multipart/form-data</p>
                    {service?.predict_url && (
                      <p><strong>Real Predict URL:</strong> <code>{service.predict_url}</code></p>
                    )}
                  </Card>
                  {apiExample?.curl && (
                    <Card title="cURL" size="small" style={{ marginBottom: 12 }}
                      extra={<Button size="small" icon={<CopyOutlined />} onClick={() => copyToClipboard(apiExample.curl)}>复制</Button>}>
                      <pre style={{ background: '#f5f5f5', padding: 8, borderRadius: 4, overflow: 'auto', fontSize: 12 }}>
                        {apiExample.curl}
                      </pre>
                    </Card>
                  )}
                  {apiExample?.python && (
                    <Card title="Python" size="small"
                      extra={<Button size="small" icon={<CopyOutlined />} onClick={() => copyToClipboard(apiExample.python)}>复制</Button>}>
                      <pre style={{ background: '#f5f5f5', padding: 8, borderRadius: 4, overflow: 'auto', fontSize: 12 }}>
                        {apiExample.python}
                      </pre>
                    </Card>
                  )}
                </div>
              )
            },
          ]} />
        </>
      )}
    </div>
  );
};

export default ServiceDetail;
