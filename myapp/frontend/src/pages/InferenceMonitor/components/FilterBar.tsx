import React from 'react';
import { Row, Col, Select, Switch, Button, Tooltip } from 'antd';
import { ReloadOutlined } from '@ant-design/icons';
import type { TimeRange, StatusFilter } from '../types';
import { TIME_RANGE_OPTIONS } from '../types';

interface FilterBarProps {
  statusFilter: StatusFilter;
  onStatusChange: (s: StatusFilter) => void;
  timeRange: TimeRange;
  onTimeRangeChange: (r: TimeRange) => void;
  autoRefresh: boolean;
  onAutoRefreshChange: (a: boolean) => void;
  onRefresh: () => void;
  loading: boolean;
}

const FilterBar: React.FC<FilterBarProps> = ({ statusFilter, onStatusChange, timeRange, onTimeRangeChange, autoRefresh, onAutoRefreshChange, onRefresh, loading }) => (
  <div style={{ background: '#fff', padding: '8px 16px', borderRadius: 4 }}>
    <Row gutter={[12, 8]} align="middle">
      <Col>
        <span style={{ fontSize: 13, color: 'rgba(0,0,0,0.65)', marginRight: 4 }}>引擎：</span>
        <Tooltip title="当前版本仅支持 vLLM">
          <Select size="small" value="vllm" disabled style={{ width: 90 }}
            options={[{ label: 'vLLM', value: 'vllm' }]} />
        </Tooltip>
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
