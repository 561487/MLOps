import React, { useState } from 'react';
import { Input, Tag, Tooltip, Button, Empty, Spin } from 'antd';
import { SearchOutlined, EyeOutlined } from '@ant-design/icons';
import type { ServiceItem } from '../types';

interface ServiceListProps {
  services: ServiceItem[];
  loading: boolean;
  selectedServiceId: number | null;
  onSelect: (s: ServiceItem) => void;
}

const MODEL_STATUS: Record<string, { color: string; label: string }> = {
  online:  { color: '#52c41a', label: '在线' },
  debug:   { color: '#1890ff', label: '调试' },
  test:    { color: '#faad14', label: '测试' },
  offline: { color: '#d9d9d9', label: '已停止' },
};

const ServiceList: React.FC<ServiceListProps> = ({ services, loading, selectedServiceId, onSelect }) => {
  const [search, setSearch] = useState('');
  const filtered = services.filter(s => {
    if (!search) return true;
    const q = search.toLowerCase();
    return (s.label || s.service_name || '').toLowerCase().includes(q)
      || (s.model_name || '').toLowerCase().includes(q);
  });

  return (
    <div style={{ background: '#fff', borderRadius: 4, padding: 12, height: '100%', display: 'flex', flexDirection: 'column' }}>
      <div style={{ marginBottom: 8, fontWeight: 600, fontSize: 14 }}>
        可监控的 vLLM 服务（{services.length}）
      </div>
      <Input size="small" placeholder="搜索服务名称或模型名称" prefix={<SearchOutlined />}
        value={search} onChange={e => setSearch(e.target.value)} allowClear style={{ marginBottom: 8 }} />
      <div style={{ flex: 1, overflow: 'auto', minHeight: 0 }}>
        <Spin spinning={loading}>
          {filtered.length === 0 && !loading ? (
            <Empty description="暂无可监控的 vLLM 推理服务" image={Empty.PRESENTED_IMAGE_SIMPLE}>
              <span style={{ color: '#999', fontSize: 12 }}>当前版本仅支持普通 vLLM 推理服务</span>
            </Empty>
          ) : (
            filtered.map(svc => {
              const sel = svc.service_id === selectedServiceId;
              const statusCfg = MODEL_STATUS[svc.model_status] || { color: '#d9d9d9', label: svc.model_status };
              return (
                <div key={svc.service_id}
                  onClick={() => onSelect(svc)}
                  style={{
                    border: `1px solid ${sel ? '#1890ff' : '#f0f0f0'}`, borderRadius: 4,
                    padding: '8px 12px', marginBottom: 8, cursor: 'pointer',
                    backgroundColor: sel ? '#e6f7ff' : '#fff',
                  }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 4 }}>
                    <Tooltip title={(svc.label || svc.service_name).length > 28 ? svc.label || svc.service_name : undefined}>
                      <span style={{ fontWeight: 600, fontSize: 14, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: 200 }}>
                        {svc.label || svc.service_name}
                      </span>
                    </Tooltip>
                    <Tag color={statusCfg.color} style={{ margin: 0 }}>
                      {statusCfg.label}
                    </Tag>
                  </div>
                  {svc.model_name && (
                    <div style={{ fontSize: 12, color: '#666', marginBottom: 2 }}>
                      <Tooltip title={svc.model_name.length > 40 ? svc.model_name : undefined}>
                        <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', display: 'inline-block', maxWidth: 260 }}>
                          模型：{svc.model_name}
                        </span>
                      </Tooltip>
                    </div>
                  )}
                  <div style={{ fontSize: 12, color: '#666', marginBottom: 2 }}>
                    服务：{svc.service_name}
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <span style={{ fontSize: 12, color: '#999' }}>
                      引擎：{svc.service_type}
                    </span>
                    <Button type="link" size="small" icon={<EyeOutlined />}
                      onClick={e => { e.stopPropagation(); onSelect(svc); }}>查看监控</Button>
                  </div>
                </div>
              );
            })
          )}
        </Spin>
      </div>
    </div>
  );
};

export default ServiceList;
