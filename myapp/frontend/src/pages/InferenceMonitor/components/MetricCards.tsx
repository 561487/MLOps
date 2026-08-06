import React from 'react';
import { Row, Col, Card, Statistic, Tag, Empty } from 'antd';
import type { SummaryResult } from '../types';

interface MetricCardsProps { summary: SummaryResult | null; }

const CARDS = [
  { key: 'qps', title: '完成请求 QPS' },
  { key: 'running_requests', title: '运行请求数' },
  { key: 'waiting_requests', title: '等待请求数' },
  { key: 'ttft_p95', title: 'TTFT P95' },
  { key: 'itl_p95', title: 'ITL P95' },
];

// 延迟类指标无当前请求样本时显示 —，不显示 0 或"暂无请求数据"
const LATENCY_KEYS = new Set(['ttft_p95', 'itl_p95']);

const MetricCards: React.FC<MetricCardsProps> = ({ summary }) => {
  if (!summary) return <div style={{ marginBottom: 12 }}><Empty description="暂无数据" /></div>;

  const { metrics } = summary;

  return (
    <div style={{ marginBottom: 12 }}>
      <div style={{ fontSize: 12, color: '#999', marginBottom: 6 }}>
        当前状态
      </div>
      <Row gutter={[12, 12]}>
        {CARDS.map(({ key, title }) => {
          const m = metrics?.[key];
          if (!m) return (
            <Col xs={12} sm={8} md={6} lg={4} key={key}>
              <Card size="small"><Statistic title={title} value="-" /></Card>
            </Col>
          );
          // unsupported：当前引擎版本不提供该指标
          if (m.status === 'unsupported') return (
            <Col xs={12} sm={8} md={6} lg={4} key={key}>
              <Card size="small">
                <Statistic title={title} value="当前版本不支持"
                  valueStyle={{ fontSize: 14, color: '#d9d9d9' }}
                  suffix={<span style={{ fontSize: 11, color: '#bfbfbf' }} title={m.message}>{m.message ? 'ⓘ' : ''}</span>} />
              </Card>
            </Col>
          );
          // no_data：监控正常但近期无请求；延迟指标显示 —，速率/并发显示 0
          if (m.status === 'no_data') return (
            <Col xs={12} sm={8} md={6} lg={4} key={key}>
              <Card size="small">
                {LATENCY_KEYS.has(key)
                  ? <Statistic title={title} value="—" valueStyle={{ fontSize: 18, color: '#bfbfbf' }} />
                  : <Statistic title={title} value={0} suffix={m.unit}
                      precision={0} valueStyle={{ fontSize: 18 }} />}
              </Card>
            </Col>
          );
          // query_failed：Prometheus 查询异常
          if (m.status === 'query_failed') return (
            <Col xs={12} sm={8} md={6} lg={4} key={key}>
              <Card size="small"><Statistic title={title} value="查询失败" valueStyle={{ fontSize: 14, color: '#ff4d4f' }} /></Card>
            </Col>
          );
          // normal：正常数值（null/undefined 不渲染为 0）
          if (m.status === 'normal') {
            const displayValue = (m.value != null && Number.isFinite(Number(m.value)))
              ? Number(m.value)
              : undefined;
            return (
              <Col xs={12} sm={8} md={6} lg={4} key={key}>
                <Card size="small">
                  <Statistic title={title}
                    value={displayValue}
                    suffix={m.unit}
                    precision={displayValue != null ? 1 : undefined}
                    valueStyle={{ fontSize: 18 }} />
                </Card>
              </Col>
            );
          }
          // 其他未知状态，安全回退
          return (
            <Col xs={12} sm={8} md={6} lg={4} key={key}>
              <Card size="small"><Statistic title={title} value="-" /></Card>
            </Col>
          );
        })}
      </Row>
    </div>
  );
};

export default MetricCards;
