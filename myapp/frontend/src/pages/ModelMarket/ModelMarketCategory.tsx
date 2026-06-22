// 模型市场分类页

import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { Row, Col, Spin, Empty, Input, Button, Pagination, Tag, message } from 'antd';
import { SearchOutlined, CodeOutlined, ThunderboltOutlined, CloudUploadOutlined } from '@ant-design/icons';
import { getModelList } from './api';
import { IModelMarketItem } from './types';
import { getModelTagColor } from './components/tagStyle';

const nameToLabel: Record<string, string> = {
  model_market_vision: '视觉模型',
  model_market_audio: '语音模型',
  model_market_nlp: '自然语言模型',
  model_market_multimodal: '多模态模型',
  model_market_llm: '大模型',
};

const nameToCategory: Record<string, string> = {
  model_market_vision: 'vision',
  model_market_audio: 'audio',
  model_market_nlp: 'nlp',
  model_market_multimodal: 'multimodal',
  model_market_llm: 'llm',
};

const categoryGradients: Record<string, string> = {
  vision: 'linear-gradient(135deg, #E0F2FE 0%, #F0F9FF 45%, #DBEAFE 100%)',
  audio: 'linear-gradient(135deg, #D1FAE5 0%, #ECFDF5 45%, #CCFBF1 100%)',
  nlp: 'linear-gradient(135deg, #FCE7F3 0%, #FFF1F2 45%, #FDE68A 100%)',
  multimodal: 'linear-gradient(135deg, #EDE9FE 0%, #F5F3FF 45%, #FAE8FF 100%)',
  llm: 'linear-gradient(135deg, #FEF3C7 0%, #FFF7ED 45%, #FFEDD5 100%)',
};

// 浅色背景下使用深色文字
const categoryTextColors: Record<string, string> = {
  vision: '#075985',
  audio: '#047857',
  nlp: '#BE185D',
  multimodal: '#6D28D9',
  llm: '#B45309',
};

const categoryLabels: Record<string, string> = {
  vision: '视觉',
  audio: '语音',
  nlp: '自然语言',
  multimodal: '多模态',
  llm: '大模型',
};

function getCategoryGradient(category: string): string {
  return categoryGradients[category] || 'linear-gradient(135deg, #F1F5F9 0%, #F8FAFC 45%, #E0F2FE 100%)';
}

function getCategoryTextColor(category: string): string {
  return categoryTextColors[category] || '#334155';
}

// 标签颜色：根据字符串 hash 到固定颜色，保证同一标签颜色一致
const TAG_PALETTE = [
  { bg: '#E8F4FD', text: '#1677FF' },
  { bg: '#F0F5FF', text: '#2F54EB' },
  { bg: '#EDF9E8', text: '#389E0D' },
  { bg: '#E6FFFB', text: '#13C2C2' },
  { bg: '#FFF1F0', text: '#CF1322' },
  { bg: '#FFF2E8', text: '#D46B08' },
  { bg: '#F9F0FF', text: '#722ED1' },
  { bg: '#FCE4EC', text: '#C41D7F' },
  { bg: '#E0F7FA', text: '#00838F' },
  { bg: '#FFF9E6', text: '#AD6800' },
];

function hashTag(tag: string): number {
  let h = 0;
  for (let i = 0; i < tag.length; i++) {
    h = (h * 31 + tag.charCodeAt(i)) & 0x7fffffff;
  }
  return h;
}

function tagColor(tag: string): { bg: string; color: string } {
  const c = TAG_PALETTE[hashTag(tag) % TAG_PALETTE.length];
  return { bg: c.bg, color: c.text };
}

function hasValidCover(cover: string | undefined | null): boolean {
  return !!(cover && cover.length > 0);
}

// 模型封面组件 — 用 <img onError> 检测 404 并降级到 CSS 渐变
const ModelCover: React.FC<{ model: IModelMarketItem }> = ({ model }) => {
  const [imgFailed, setImgFailed] = useState(false);
  const coverUrl = hasValidCover(model.cover) ? model.cover : '';

  if (coverUrl && !imgFailed) {
    return (
      <div style={{ height: 140, flexShrink: 0, overflow: 'hidden', position: 'relative', background: '#f5f5f5' }}>
        <img
          src={coverUrl}
          alt={model.display_name}
          style={{ width: '100%', height: '100%', objectFit: 'cover' }}
          onError={() => setImgFailed(true)}
        />
      </div>
    );
  }

  const gradient = getCategoryGradient(model.category);
  const textColor = getCategoryTextColor(model.category);
  return (
    <div
      style={{
        height: 140,
        background: gradient,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        flexShrink: 0,
        gap: 4,
      }}
    >
      <span style={{ fontSize: 15, color: textColor, fontWeight: 600, opacity: 0.85 }}>
        {categoryLabels[model.category]}
      </span>
      <span style={{ fontSize: 11, color: textColor, fontWeight: 400, opacity: 0.55 }}>
        {model.category === 'vision' ? 'Computer Vision' :
         model.category === 'audio' ? 'Speech & Audio' :
         model.category === 'nlp' ? 'Natural Language' :
         model.category === 'multimodal' ? 'Multimodal AI' :
         model.category === 'llm' ? 'Large Language Model' : 'AI Model'}
      </span>
    </div>
  );
};

// 按钮颜色配置
const btnStyles: Record<string, React.CSSProperties> = {
  develop: {
    color: '#15803d', background: '#dcfce7', border: '1px solid #bbf7d0',
  },
  finetune: {
    color: '#b45309', background: '#fef3c7', border: '1px solid #fde68a',
  },
  deploy: {
    color: '#6d28d9', background: '#ede9fe', border: '1px solid #ddd6fe',
  },
};

const PAGE_SIZE = 12;

const ModelMarketCategory: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const [models, setModels] = useState<IModelMarketItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [keyword, setKeyword] = useState('');

  const pathSegments = location.pathname.split('/');
  const menuName = pathSegments[pathSegments.length - 1] || 'model_market_vision';
  const pageTitle = nameToLabel[menuName] || menuName;
  const categoryKey = nameToCategory[menuName] || '';

  const fetchModels = useCallback(async (p?: number, kw?: string) => {
    setLoading(true);
    try {
      const res = await getModelList({
        category: categoryKey || undefined,
        keyword: kw || keyword || undefined,
        page: p || 1,
        page_size: PAGE_SIZE,
      });
      const data = res.data || [];
      setModels(data);
      setTotal(res.total || data.length);
    } catch (err: any) {
      message.error('获取模型列表失败: ' + (err.message || '未知错误'));
      setModels([]);
      setTotal(0);
    } finally {
      setLoading(false);
    }
  }, [categoryKey]);

  // 分类或页码变化时重新拉取
  useEffect(() => {
    setPage(1);
    setKeyword('');
    fetchModels(1, '');
  }, [menuName]);

  const handleSearch = (kw?: string) => {
    const q = kw ?? keyword;
    setPage(1);
    fetchModels(1, q);
  };

  const handlePageChange = (p: number) => {
    setPage(p);
    fetchModels(p, keyword);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const handleEnterDetail = (id: number) => {
    navigate(`/service/model_market_group/detail/${id}`);
  };

  const handleDevelop = (id: number) => {
    navigate(`/service/model_market_group/detail/${id}?tab=develop`);
  };

  const handleFinetune = (id: number) => {
    navigate(`/service/model_market_group/detail/${id}?tab=finetune`);
  };

  const handleDeploy = (id: number) => {
    navigate(`/service/model_market_group/detail/${id}?tab=deploy`);
  };

  return (
    <div className="bg-w" style={{ padding: '20px 24px', minHeight: '100%' }}>
      {/* Title + count */}
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 12, marginBottom: 16 }}>
        <h2 style={{ fontSize: 18, fontWeight: 600, color: '#333', margin: 0 }}>
          {pageTitle}
        </h2>
        <span style={{ fontSize: 13, color: '#999' }}>共 {total} 个模型</span>
      </div>

      {/* Search */}
      <div style={{ marginBottom: 20 }}>
        <Input.Search
          placeholder="搜索模型名称、标签、描述..."
          value={keyword}
          onChange={(e) => setKeyword(e.target.value)}
          onSearch={(v) => handleSearch(v)}
          style={{ maxWidth: 480 }}
          prefix={<SearchOutlined />}
          allowClear
          enterButton="搜索"
        />
      </div>

      {/* Cards */}
      <Spin spinning={loading}>
        {models.length > 0 ? (
          <Row gutter={[16, 20]}>
            {models.map((model) => (

                <Col key={model.id} xs={24} sm={12} md={8} lg={6} xl={6}>
                  <div
                    style={{
                      background: '#fff',
                      border: '1px solid #f0f0f0',
                      borderRadius: 6,
                      overflow: 'hidden',
                      cursor: 'pointer',
                      transition: 'box-shadow 0.2s',
                      height: '100%',
                      display: 'flex',
                      flexDirection: 'column',
                    }}
                    onMouseEnter={(e) => (e.currentTarget.style.boxShadow = '0 2px 12px rgba(0,0,0,0.08)')}
                    onMouseLeave={(e) => (e.currentTarget.style.boxShadow = 'none')}
                    onClick={() => handleEnterDetail(model.id)}
                  >
                    <ModelCover model={model} />

                    {/* Body */}
                    <div style={{ padding: '14px 14px 12px', display: 'flex', flexDirection: 'column', flex: 1 }}>
                      {/* Title */}
                      <div style={{ fontWeight: 600, fontSize: 14, color: '#222', marginBottom: 8, lineHeight: '20px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {model.display_name}
                      </div>

                      {/* Description */}
                      <div style={{
                        fontSize: 12, color: '#666', marginBottom: 10,
                        display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical',
                        overflow: 'hidden', lineHeight: '18px', minHeight: 36, flex: 1,
                      }}>
                        {model.description}
                      </div>

                      {/* Tags — unified colors */}
                      {model.tags && model.tags.length > 0 && (
                        <div style={{ marginBottom: 12, display: 'flex', flexWrap: 'wrap', gap: 5 }}>
                          {model.tags.slice(0, 4).map((tag: string) => (
                            <Tag key={tag} color={getModelTagColor(tag)} style={{ margin: 0, borderRadius: 3, fontSize: 11, lineHeight: '20px' }}>
                              {tag}
                            </Tag>
                          ))}
                        </div>
                      )}

                      {/* Action buttons — soft colors */}
                      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginTop: 'auto' }}>
                        {model.support_develop && (
                          <Button size="small" icon={<CodeOutlined />}
                            style={{ ...btnStyles.develop, borderRadius: 4, fontWeight: 500 }}
                            onClick={(e) => { e.stopPropagation(); handleDevelop(model.id); }}>
                            开发
                          </Button>
                        )}
                        {model.support_finetune && (
                          <Button size="small" icon={<ThunderboltOutlined />}
                            style={{ ...btnStyles.finetune, borderRadius: 4, fontWeight: 500 }}
                            onClick={(e) => { e.stopPropagation(); handleFinetune(model.id); }}>
                            微调
                          </Button>
                        )}
                        {model.support_deploy && (
                          <Button size="small" icon={<CloudUploadOutlined />}
                            style={{ ...btnStyles.deploy, borderRadius: 4, fontWeight: 500 }}
                            onClick={(e) => { e.stopPropagation(); handleDeploy(model.id); }}>
                            部署
                          </Button>
                        )}
                      </div>
                    </div>
                  </div>
                </Col>
              ))}
            </Row>
          ) : (
            !loading && <Empty description="该分类暂无模型" />
          )}
        </Spin>

      {/* Pagination */}
      {total > PAGE_SIZE && (
        <div style={{ marginTop: 24, textAlign: 'center' }}>
          <Pagination
            current={page}
            total={total}
            pageSize={PAGE_SIZE}
            onChange={handlePageChange}
            showSizeChanger={false}
            showTotal={(t) => `共 ${t} 个模型`}
          />
        </div>
      )}
    </div>
  );
};

export default ModelMarketCategory;
