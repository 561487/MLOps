import React from 'react';
import { Row, Col, Card, Statistic, Tag, Empty } from 'antd';
import type { SummaryResult } from '../types';
import { MONITOR_STATUS_LABELS, MONITOR_STATUS_COLORS } from '../types';

interface MetricCardsProps { summary: SummaryResult | null; }

const CARDS = [
  { key: 'qps', title: '完成请求 QPS' },
  { key: 'running_requests', title: '运行请求数' },
  { key: 'waiting_requests', title: '等待请求数' },
  { key: 'ttft_p95', title: 'TTFT P95' },
  { key: 'itl_p95', title: 'ITL P95' },
];

const MetricCards: React.FC<MetricCardsProps> = ({ summary }) => {
  if (!summary) return <div style={{ marginBottom: 12 }}><Empty description="暂无数据" /></div>;

  const { monitor_status, model_status, ready, metrics } = summary;

  return (
    <div style={{ marginBottom: 12 }}>
      <Row gutter={[12, 12]}>
        {CARDS.map(({ key, title }) => {
          const m = metrics?.[key];
          if (!m) return (
            <Col xs={12} sm={8} md={6} lg={4} key={key}>
              <Card size="small"><Statistic title={title} value="-" /></Card>
            </Col>
          );
          if (m.status === 'no_data') return (
            <Col xs={12} sm={8} md={6} lg={4} key={key}>
              <Card size="small"><Statistic title={title} value="暂无请求数据" valueStyle={{ fontSize: 14, color: '#faad14' }} /></Card>
            </Col>
          );
          if (m.status === 'query_failed') return (
            <Col xs={12} sm={8} md={6} lg={4} key={key}>
              <Card size="small"><Statistic title={title} value="监控不可用" valueStyle={{ fontSize: 14, color: '#ff4d4f' }} /></Card>
            </Col>
          );
          return (
            <Col xs={12} sm={8} md={6} lg={4} key={key}>
              <Card size="small">
                <Statistic title={title} value={m.value != null ? m.value : undefined}
                  suffix={m.unit} precision={m.value != null ? 1 : undefined} valueStyle={{ fontSize: 18 }} />
              </Card>
            </Col>
          );
        })}
      </Row>
    </div>
  );
};

export default MetricCards;
