// 模型详情页

import React, { useState, useEffect, useCallback } from 'react';
import { useParams, useNavigate, useSearchParams } from 'react-router-dom';
import { Tabs, Button, Spin, Descriptions, Tag, Breadcrumb, message } from 'antd';
import {
  CodeOutlined, ThunderboltOutlined,
  CloudUploadOutlined, ApiOutlined, HomeOutlined,
} from '@ant-design/icons';
import { getModelDetail, getModelServices } from './api';
import { IModelMarketItem } from './types';
import ModelOverviewTab from './components/ModelOverviewTab';
import ModelDevelopPanel from './components/ModelDevelopPanel';
import ModelFinetunePanel from './components/ModelFinetunePanel';
import ModelDeployPanel from './components/ModelDeployPanel';
import ModelApiDocPanel from './components/ModelApiDocPanel';
import { getModelTagColor } from './components/tagStyle';

const categoryLabels: Record<string, string> = {
  vision: '视觉', audio: '语音', nlp: '自然语言', multimodal: '多模态', llm: '大模型',
};

const categoryGradients: Record<string, string> = {
  vision: 'linear-gradient(135deg, #E0F2FE 0%, #F0F9FF 45%, #DBEAFE 100%)',
  audio: 'linear-gradient(135deg, #D1FAE5 0%, #ECFDF5 45%, #CCFBF1 100%)',
  nlp: 'linear-gradient(135deg, #FCE7F3 0%, #FFF1F2 45%, #FDE68A 100%)',
  multimodal: 'linear-gradient(135deg, #EDE9FE 0%, #F5F3FF 45%, #FAE8FF 100%)',
  llm: 'linear-gradient(135deg, #FEF3C7 0%, #FFF7ED 45%, #FFEDD5 100%)',
};

const categoryTextColors: Record<string, string> = {
  vision: '#075985', audio: '#047857', nlp: '#BE185D', multimodal: '#6D28D9', llm: '#B45309',
};

function hasValidCover(cover: string | undefined | null): boolean {
  return !!(cover && cover.length > 0);
}

const btnStyles: Record<string, React.CSSProperties> = {
  develop: { color: '#15803d', background: '#dcfce7', border: '1px solid #bbf7d0' },
  finetune: { color: '#b45309', background: '#fef3c7', border: '1px solid #fde68a' },
  deploy: { color: '#6d28d9', background: '#ede9fe', border: '1px solid #ddd6fe' },
};

/** Normalize a tag value to a canonical key for dedup */
const normalizeTagKey = (tag?: string) => {
  const value = String(tag || '').trim().toLowerCase();
  if (!value) return '';

  if (value === 'pytorch' || value === 'torch') return 'pytorch';
  if (value === 'vision' || value === '视觉') return 'vision';
  if (value === 'object_detection' || value === '目标检测') return 'object_detection';
  if (value === 'online') return 'online';
  if (value === 'yolo' || value === 'yolov8') return 'yolo';

  return value;
};

/** Display-friendly text for a tag */
const getDisplayTagText = (tag?: string) => {
  const value = String(tag || '').trim();
  const lower = value.toLowerCase();

  if (lower === 'pytorch') return 'PyTorch';
  if (lower === 'vision') return '视觉';
  if (lower === 'object_detection') return '目标检测';
  if (lower === 'yolo' || lower === 'yolov8') return 'YOLO';
  if (lower === 'online') return 'online';

  return value;
};

/** Build deduplicated, normalized tag list for model detail header */
const buildModelDetailTags = (model: IModelMarketItem) => {
  const rawTags: string[] = [];

  // Prefer model.tags — it already covers category / task_type / framework
  if (Array.isArray(model.tags) && model.tags.length > 0) {
    rawTags.push(...model.tags);
  }

  // Only fall back to individual fields when tags is empty
  if (rawTags.length === 0) {
    rawTags.push(
      model.task_type || '',
      model.framework || '',
      model.category || ''
    );
  }

  const seen = new Set<string>();
  const result: string[] = [];

  rawTags.forEach(tag => {
    const key = normalizeTagKey(tag);
    if (!key || seen.has(key)) return;

    seen.add(key);
    result.push(getDisplayTagText(tag));
  });

  return result;
};

const ModelDetail: React.FC = () => {
  const { modelId } = useParams<{ modelId: string }>();
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const [model, setModel] = useState<IModelMarketItem | null>(null);
  const [loading, setLoading] = useState(true);
  const [imgFailed, setImgFailed] = useState(false);
  const [activeTab, setActiveTab] = useState(searchParams.get('tab') || 'overview');
  const [hasActiveService, setHasActiveService] = useState(false);

  const fetchModel = useCallback(async () => {
    if (!modelId) return;
    setLoading(true);
    try {
      const data = await getModelDetail(modelId);
      setModel(data);
    } catch (err: any) {
      message.error('获取模型详情失败: ' + (err.message || '未知错误'));
    } finally {
      setLoading(false);
    }
  }, [modelId]);

  // Check if model has active deployment services
  const checkActiveServices = useCallback(async () => {
    if (!modelId) return;
    try {
      const result = await getModelServices(modelId, { active_only: true });
      const svcList = result?.services || [];
      setHasActiveService(svcList.length > 0);
    } catch {
      setHasActiveService(false);
    }
  }, [modelId]);

  useEffect(() => { fetchModel(); }, [fetchModel]);
  useEffect(() => { checkActiveServices(); }, [checkActiveServices]);

  const handleTabChange = (key: string) => {
    setActiveTab(key);
    setSearchParams({ tab: key });
  };

  // Handle deploy/unload button click
  const handleDeployButton = () => {
    if (hasActiveService) {
      // Go to deploy tab to show unload UI
      handleTabChange('deploy');
    } else {
      handleTabChange('deploy');
    }
  };

  // Callback from deploy panel when service state changes
  const onServiceStateChange = useCallback(() => {
    checkActiveServices();
  }, [checkActiveServices]);

  if (loading) return <div style={{ textAlign: 'center', padding: 60 }}><Spin size="large" /></div>;
  if (!model) return <div style={{ textAlign: 'center', padding: 60 }}>模型未找到</div>;

  const coverUrl = hasValidCover(model.cover) ? model.cover : '';
  const showCoverImg = coverUrl && !imgFailed;
  const pythonVer = model.python_version || '3.10';
  const cudaVer = model.cuda_version || '11.8';
  const imageVer = model.image || model.inference_image || model.finetune_image || model.notebook_image || '未配置';

  return (
    <div className="bg-w" style={{ padding: '20px 24px', minHeight: '100%' }}>
      {/* Breadcrumb */}
      <Breadcrumb style={{ marginBottom: 16, fontSize: 12 }}>
        <Breadcrumb.Item><a onClick={() => navigate('/service/model_market_group/model_market_vision')}><HomeOutlined /> 模型市场</a></Breadcrumb.Item>
        <Breadcrumb.Item><a onClick={() => navigate(`/service/model_market_group/model_market_${model.category}`)}>{categoryLabels[model.category]}</a></Breadcrumb.Item>
        <Breadcrumb.Item>{model.display_name}</Breadcrumb.Item>
      </Breadcrumb>

      {/* Header Card */}
      <div style={{ background: '#fff', border: '1px solid #f0f0f0', borderRadius: 6, padding: 24, marginBottom: 16, display: 'flex', gap: 20, flexWrap: 'wrap' }}>
        {/* Cover */}
        <div style={{ width: 200, height: 140, flexShrink: 0, borderRadius: 4, overflow: 'hidden', background: '#f5f5f5' }}>
          {showCoverImg ? (
            <img src={coverUrl} alt={model.display_name} style={{ width: '100%', height: '100%', objectFit: 'cover' }}
              onError={() => setImgFailed(true)} />
          ) : (
            <div style={{ height: '100%', background: categoryGradients[model.category] || categoryGradients.vision,
              display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 4 }}>
              <span style={{ fontSize: 15, color: categoryTextColors[model.category], fontWeight: 600 }}>{categoryLabels[model.category]}</span>
              <span style={{ fontSize: 11, color: categoryTextColors[model.category], opacity: 0.6 }}>AI Model</span>
            </div>
          )}
        </div>

        {/* Info */}
        <div style={{ flex: 1, minWidth: 280 }}>
          <h2 style={{ fontSize: 20, fontWeight: 600, margin: '0 0 6px', color: '#222' }}>{model.display_name}</h2>
          <p style={{ fontSize: 13, color: '#666', margin: '0 0 10px', lineHeight: '20px' }}>{model.description}</p>

          {/* Tags — deduplicated via buildModelDetailTags */}
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 10 }}>
            {buildModelDetailTags(model).map(tag => (
              <Tag key={normalizeTagKey(tag)} color={getModelTagColor(tag)} style={{ margin: 0, borderRadius: 3 }}>{tag}</Tag>
            ))}
          </div>

          {/* Specs */}
          <Descriptions size="small" column={3} style={{ marginBottom: 8 }}
            labelStyle={{ color: '#888', fontSize: 12 }} contentStyle={{ color: '#333', fontSize: 13 }}>
            <Descriptions.Item label="模型分类">{categoryLabels[model.category]}</Descriptions.Item>
            <Descriptions.Item label="任务类型">{model.task_type || '—'}</Descriptions.Item>
            <Descriptions.Item label="框架">{model.framework || '—'}</Descriptions.Item>
            <Descriptions.Item label="Python">{pythonVer || '—'}</Descriptions.Item>
            <Descriptions.Item label="CUDA">{cudaVer || '—'}</Descriptions.Item>
            <Descriptions.Item label="镜像"><code style={{ fontSize: 11 }}>{imageVer}</code></Descriptions.Item>
          </Descriptions>

          {/* Action Buttons */}
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            {model.support_develop && (
              <Button icon={<CodeOutlined />} style={{ ...btnStyles.develop, borderRadius: 4, fontWeight: 500 }}
                onClick={() => handleTabChange('develop')}>一键开发</Button>
            )}
            {model.support_finetune && (
              <Button icon={<ThunderboltOutlined />} style={{ ...btnStyles.finetune, borderRadius: 4, fontWeight: 500 }}
                onClick={() => handleTabChange('finetune')}>一键微调</Button>
            )}
            {model.support_deploy && (
              <Button icon={<CloudUploadOutlined />} style={{ ...btnStyles.deploy, borderRadius: 4, fontWeight: 500 }}
                onClick={handleDeployButton}>
                {hasActiveService ? '一键卸载' : '一键部署'}
              </Button>
            )}
            <Button icon={<ApiOutlined />} onClick={() => handleTabChange('api')}>API 文档</Button>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div style={{ background: '#fff', border: '1px solid #f0f0f0', borderRadius: 6, padding: '8px 20px 20px' }}>
        <Tabs activeKey={activeTab} onChange={handleTabChange}
          tabBarStyle={{ marginBottom: 0 }}>
          <Tabs.TabPane tab="模型介绍" key="overview">
            <ModelOverviewTab model={model} />
          </Tabs.TabPane>
          {model.support_develop && (
            <Tabs.TabPane tab="一键开发" key="develop">
              <ModelDevelopPanel model={model} />
            </Tabs.TabPane>
          )}
          {model.support_finetune && (
            <Tabs.TabPane tab="一键微调" key="finetune">
              <ModelFinetunePanel model={model} />
            </Tabs.TabPane>
          )}
          {model.support_deploy && (
            <Tabs.TabPane tab="一键部署" key="deploy">
              <ModelDeployPanel model={model} onServiceStateChange={onServiceStateChange} />
            </Tabs.TabPane>
          )}
          <Tabs.TabPane tab="API 文档" key="api">
            <ModelApiDocPanel model={model} />
          </Tabs.TabPane>
        </Tabs>
      </div>
    </div>
  );
};

export default ModelDetail;
