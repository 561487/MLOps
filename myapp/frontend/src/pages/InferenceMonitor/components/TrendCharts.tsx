import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Card, Row, Col, Spin, Empty } from 'antd';
import moment from 'moment';
import EchartCore from '../../../components/EchartCore/EchartCore';
import { fetchTimeseries } from '../api';
import type { TimeRange, MetricPoint } from '../types';
import type { EChartsOption } from 'echarts';

interface TrendChartsProps {
  serviceId: number;
  timeRange: TimeRange;
  enabled: boolean;
  disabledReason?: string;
  /** 父组件刷新时递增，触发趋势数据重新加载。不作为 React key 使用。 */
  refreshToken?: number;
  /** 手动刷新时绕过服务端缓存 */
  forceRefresh?: boolean;
}

// 指标元数据：名称、单位、精度
const METRIC_META: Record<string, { label: string; unit: string; precision: number }> = {
  ttft_p50:  { label: 'TTFT P50',  unit: 'ms',         precision: 2 },
  ttft_p95:  { label: 'TTFT P95',  unit: 'ms',         precision: 2 },
  ttft_p99:  { label: 'TTFT P99',  unit: 'ms',         precision: 2 },
  itl_p50:   { label: 'ITL P50',   unit: 'ms/token',   precision: 2 },
  itl_p95:   { label: 'ITL P95',   unit: 'ms/token',   precision: 2 },
  qps:       { label: '完成请求 QPS', unit: 'req/s',   precision: 3 },
  running_requests:        { label: '运行请求数', unit: 'requests', precision: 0 },
  waiting_requests:        { label: '等待请求数', unit: 'requests', precision: 0 },
  input_tokens_per_second:  { label: '输入 Token/s', unit: 'tokens/s', precision: 2 },
  output_tokens_per_second: { label: '输出 Token/s', unit: 'tokens/s', precision: 2 },
};

// 图表分组定义
const CHART_GROUPS = [
  { title: 'TTFT 延迟', metrics: ['ttft_p50', 'ttft_p95', 'ttft_p99'] },
  { title: 'ITL 延迟',  metrics: ['itl_p50', 'itl_p95'] },
  // 请求吞吐使用双 Y 轴
  { title: '请求吞吐', metrics: ['qps', 'running_requests'] as const, dualAxis: true },
  { title: 'Token 吞吐', metrics: ['input_tokens_per_second', 'output_tokens_per_second'] },
];

/** 根据时间范围格式化横坐标时间标签 */
function formatAxisTime(value: number, range: TimeRange): string {
  const m = moment(value);
  if (!m.isValid()) return '';
  if (range === '5m' || range === '15m') return m.format('HH:mm:ss');
  if (range === '1h' || range === '6h')   return m.format('HH:mm');
  return m.format('MM-DD HH:mm');
}

/** 按精度格式化数值 */
function formatValue(value: unknown, precision: number): string {
  const n = Number(value);
  if (!Number.isFinite(n)) return '-';
  return n.toFixed(precision);
}

/** 构建单个图表的 ECharts option */
function buildChartOption(
  seriesData: Record<string, MetricPoint[]>,
  metrics: string[],
  range: TimeRange,
  dualAxis?: boolean,
): EChartsOption {
  // 过滤无效时间点，构建 [[timestamp_ms, value], ...] 格式
  const buildSeriesData = (metricName: string): [number, number][] => {
    const points = seriesData[metricName] || [];
    return points
      .filter(p => Number.isFinite(p.timestamp_ms) && p.timestamp_ms > 0)
      .map(p => [p.timestamp_ms, p.value] as [number, number]);
  };

  const isDual = dualAxis && metrics.length === 2;
  const meta0 = METRIC_META[metrics[0]] || { label: metrics[0], unit: '', precision: 2 };
  const meta1 = metrics.length > 1 ? (METRIC_META[metrics[1]] || { label: metrics[1], unit: '', precision: 2 }) : null;

  const baseGrid = { top: 16, right: isDual ? 64 : 24, bottom: 52, left: 56, containLabel: true };

  const baseXAxis = {
    type: 'time' as const,
    axisLabel: {
      fontSize: 10,
      margin: 12,
      hideOverlap: true,
      formatter: (value: number) => formatAxisTime(value, range),
    },
  };

  const series: any[] = metrics.map((m, idx) => {
    const meta = METRIC_META[m] || { label: m, unit: '', precision: 2 };
    return {
      name: meta.label,
      type: 'line',
      data: buildSeriesData(m),
      smooth: true,
      connectNulls: true,
      symbol: 'none',
      lineStyle: { width: 2 },
      yAxisIndex: isDual ? idx : 0,
    };
  });

  if (isDual && meta1) {
    return {
      tooltip: {
        trigger: 'axis',
        formatter: (params: any) => {
          const items = Array.isArray(params) ? params : [params];
          const ts = items[0]?.value?.[0];
          const timeStr = ts ? moment(ts).format('YYYY-MM-DD HH:mm:ss') : '-';
          let body = `<div style="font-weight:600;margin-bottom:4px">${timeStr}</div>`;
          for (const item of items) {
            const name = item.seriesName || '';
            const val = item.value?.[1];
            // 根据 seriesName 查 meta
            const metaEntry = Object.values(METRIC_META).find(v => v.label === name);
            const prec = metaEntry?.precision ?? 2;
            const unit = metaEntry?.unit ?? '';
            const formatted = formatValue(val, prec);
            const color = item.color || '#333';
            body += `<div><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${color};margin-right:6px"></span>${name}: ${formatted} ${unit}</div>`;
          }
          return body;
        },
      },
      legend: {
        bottom: 8, left: 'center', itemWidth: 14, itemHeight: 8, itemGap: 18,
        textStyle: { fontSize: 12 },
      },
      grid: baseGrid,
      xAxis: baseXAxis,
      yAxis: [
        { type: 'value' as const, name: meta0.unit, position: 'left', axisLabel: { fontSize: 10, margin: 8 }, splitLine: { show: true, lineStyle: { type: 'dashed', color: '#f0f0f0' } } },
        { type: 'value' as const, name: meta1.unit, position: 'right', axisLabel: { fontSize: 10, margin: 8 }, splitLine: { show: false } },
      ],
      series,
      animation: false,
    };
  }

  return {
    tooltip: {
      trigger: 'axis',
      formatter: (params: any) => {
        const items = Array.isArray(params) ? params : [params];
        const ts = items[0]?.value?.[0];
        const timeStr = ts ? moment(ts).format('YYYY-MM-DD HH:mm:ss') : '-';
        let body = `<div style="font-weight:600;margin-bottom:4px">${timeStr}</div>`;
        for (const item of items) {
          const name = item.seriesName || '';
          const val = item.value?.[1];
          const metaEntry = Object.values(METRIC_META).find(v => v.label === name);
          const prec = metaEntry?.precision ?? 2;
          const unit = metaEntry?.unit ?? '';
          const formatted = formatValue(val, prec);
          const color = item.color || '#333';
          body += `<div><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${color};margin-right:6px"></span>${name}: ${formatted} ${unit}</div>`;
        }
        return body;
      },
    },
    legend: {
      bottom: 8, left: 'center', itemWidth: 14, itemHeight: 8, itemGap: 18,
      textStyle: { fontSize: 12 },
    },
    grid: baseGrid,
    xAxis: baseXAxis,
    yAxis: {
      type: 'value' as const,
      name: metrics.length === 1 ? meta0.unit : undefined,
      axisLabel: { fontSize: 10, margin: 8 },
      splitLine: { show: true, lineStyle: { type: 'dashed', color: '#f0f0f0' } },
    },
    series,
    animation: false,
  };
}

const TrendCharts: React.FC<TrendChartsProps> = ({ serviceId, timeRange, enabled, disabledReason, refreshToken, forceRefresh }) => {
  const [data, setData] = useState<Record<string, MetricPoint[]>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // 请求序号：防止旧请求覆盖新服务数据
  const requestIdRef = useRef(0);

  // serviceId / timeRange / enabled 变化时立即清空旧数据
  useEffect(() => {
    setData({});
    setError(null);
  }, [serviceId, timeRange, enabled]);

  const load = useCallback(async () => {
    // 已禁用或无效 serviceId 则不发请求
    if (!enabled || !serviceId) {
      setLoading(false);
      return;
    }

    const requestId = ++requestIdRef.current;
    setLoading(true);
    setError(null);

    try {
      const allMetrics = CHART_GROUPS.flatMap(g => g.metrics);
      const r = await fetchTimeseries(
        serviceId,
        allMetrics,
        timeRange,
        { force: forceRefresh === true },
      );

      // 请求返回时检查是否仍是当前最新请求
      if (requestId !== requestIdRef.current) return;

      setData(r.data || {});
    } catch {
      // 请求失败时检查是否仍是当前最新请求
      if (requestId !== requestIdRef.current) return;

      // 清空数据，不保留旧服务曲线
      setData({});
      setError('数据加载失败，请稍后重试');
    } finally {
      if (requestId === requestIdRef.current) {
        setLoading(false);
      }
    }
  }, [serviceId, timeRange, enabled, refreshToken, forceRefresh]);

  useEffect(() => {
    load();
    // 组件卸载或依赖变化时使旧请求失效
    return () => {
      requestIdRef.current += 1;
    };
  }, [load]);

  // 状态 1：服务已停止或未启用监控
  if (!enabled) {
    return (
      <div style={{ marginTop: 12, background: '#fff', borderRadius: 4, padding: 24 }}>
        <Empty
          description={
            disabledReason === 'service_stopped'
              ? '服务已停止，暂无监控数据'
              : '该服务尚未启用推理监控'
          }
        />
      </div>
    );
  }

  // 状态 2：加载中（仅首次加载显示 Spin）
  if (loading && !Object.values(data).some(a => a.length > 0)) {
    return (
      <div style={{ marginTop: 12, background: '#fff', borderRadius: 4, padding: 24 }}>
        <Spin spinning>
          <div style={{ height: 200 }} />
        </Spin>
      </div>
    );
  }

  // 状态 3：请求出错
  if (error && !Object.values(data).some(a => a.length > 0)) {
    return (
      <div style={{ marginTop: 12, background: '#fff', borderRadius: 4, padding: 24 }}>
        <Empty description={error} />
      </div>
    );
  }

  // 状态 4：无数据
  const hasData = Object.values(data).some(a => a.length > 0);
  if (!hasData) {
    return (
      <div style={{ marginTop: 12, background: '#fff', borderRadius: 4, padding: 24 }}>
        <Empty description="暂无趋势数据" />
      </div>
    );
  }

  // 状态 5：正常渲染图表
  return (
    <Spin spinning={loading}>
      <Row gutter={[12, 12]} style={{ marginTop: 12 }}>
        {CHART_GROUPS.map(g => (
          <Col span={12} key={g.title}>
            <Card size="small" title={<span style={{ fontSize: 13 }}>{g.title}</span>}>
              <EchartCore
                option={buildChartOption(data, g.metrics as string[], timeRange, (g as any).dualAxis)}
                style={{ height: 260 }}
              />
            </Card>
          </Col>
        ))}
      </Row>
    </Spin>
  );
};

export default TrendCharts;
