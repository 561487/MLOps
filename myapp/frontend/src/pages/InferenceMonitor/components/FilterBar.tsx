import React from 'react';
import { Row, Col, Select, Switch, Button, Tooltip } from 'antd';
import { ReloadOutlined } from '@ant-design/icons';
import type { TimeRange, StatusFilter } from '../types';
import { TIME_RANGE_OPTIONS } from '../types';

export type EngineFilter = 'all' | 'vllm' | 'sglang';

interface FilterBarProps {
  statusFilter: StatusFilter;
  onStatusChange: (s: StatusFilter) => void;
  timeRange: TimeRange;
  onTimeRangeChange: (r: TimeRange) => void;
  autoRefresh: boolean;
  onAutoRefreshChange: (a: boolean) => void;
  onRefresh: () => void;
  loading: boolean;
  engineFilter: EngineFilter;
  onEngineChange: (e: EngineFilter) => void;
}

const FilterBar: React.FC<FilterBarProps> = ({
  statusFilter, onStatusChange, timeRange, onTimeRangeChange,
  autoRefresh, onAutoRefreshChange, onRefresh, loading,
  engineFilter, onEngineChange,
}) => (
  <div style={{ background: '#fff', padding: '8px 16px', borderRadius: 4 }}>
    <Row gutter={[12, 8]} align="middle">
      <Col>
        <span style={{ fontSize: 13, color: 'rgba(0,0,0,0.65)', marginRight: 4 }}>引擎：</span>
        <Select size="small" value={engineFilter} onChange={onEngineChange} style={{ width: 100 }}
          options={[
            { label: '全部', value: 'all' },
            { label: 'vLLM', value: 'vllm' },
            { label: 'SGLang', value: 'sglang' },
          ]} />
      </Col>
      <Col>
        <span style={{ fontSize: 13, color: 'rgba(0,0,0,0.65)', marginRight: 4 }}>状态：</span>
        <Select size="small" value={statusFilter} onChange={onStatusChange} style={{ width: 100 }}
          options={[{ label: '全部', value: 'all' }, { label: '运行中', value: 'online' }, { label: '已停止', value: 'offline' }]} />
      </Col>
      <Col>
        <span style={{ fontSize: 13, color: 'rgba(0,0,0,0.65)', marginRight: 4 }}>时间范围：</span>
        <Select size="small" value={timeRange} onChange={onTimeRangeChange} style={{ width: 140 }} options={TIME_RANGE_OPTIONS} />
      </Col>
      <Col>
        <span style={{ fontSize: 13, color: 'rgba(0,0,0,0.65)', marginRight: 4 }}>自动刷新：</span>
        <Switch size="small" checked={autoRefresh} onChange={onAutoRefreshChange} />
        <span style={{ marginLeft: 8, color: '#999', fontSize: 12 }}>{autoRefresh ? '30s' : '关闭'}</span>
      </Col>
      <Col>
        <Button icon={<ReloadOutlined />} onClick={onRefresh} loading={loading} size="small">刷新</Button>
      </Col>
    </Row>
  </div>
);

export default FilterBar;
