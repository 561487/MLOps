import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Card, Spin, Empty } from 'antd';
import EchartCore from '../../../components/EchartCore/EchartCore';
import { fetchTimeseries } from '../api';
import { buildChartOption } from '../chartUtils';
import type { TimeRange, MetricPoint } from '../types';

interface TrendChartsProps {
  serviceId: number;
  timeRange: TimeRange;
  enabled: boolean;
  disabledReason?: string;
  refreshToken?: number;
  forceRefresh?: boolean;
  unsupportedMetrics?: Set<string>;
}

// vLLM 和 SGLang 共用同一配置
// 注意：运行请求数/等待请求数是 Gauge（桶内最大观测值），不是吞吐量，
// 因此「请求负载」图使用双 Y 轴：左轴 req/s，右轴 requests
const CHART_GROUPS = [
  { title: 'TTFT 延迟', metrics: ['ttft_p50', 'ttft_p95', 'ttft_p99'] as readonly string[], yUnit: 'ms' },
  { title: 'ITL 延迟',  metrics: ['itl_p50', 'itl_p95'] as readonly string[], yUnit: 'ms/token' },
  {
    title: '请求负载',
    metrics: ['qps', 'running_requests', 'waiting_requests'] as readonly string[],
    yUnit: 'req/s',
    rightYUnit: 'requests',
    rightMetrics: ['running_requests', 'waiting_requests'] as readonly string[],
    hint: '并发/排队：桶内峰值',
  },
  { title: 'Token 吞吐', metrics: ['input_tokens_per_second', 'output_tokens_per_second'] as readonly string[], yUnit: 'tokens/s' },
];

/** 原子响应状态 — 数据与窗口在一次更新中同时替换 */
interface ResponseState {
  range: string | null;
  queryStartMs: number | null;
  queryEndMs: number | null;
  displayStepSeconds: number;
  calculationWindowSeconds: number;
  data: Record<string, MetricPoint[]>;
}

const EMPTY_RESPONSE: ResponseState = {
  range: null,
  queryStartMs: null,
  queryEndMs: null,
  displayStepSeconds: 30,
  calculationWindowSeconds: 120,
  data: {},
};

const TrendCharts: React.FC<TrendChartsProps> = ({
  serviceId, timeRange, enabled, disabledReason,
  refreshToken, forceRefresh, unsupportedMetrics,
}) => {
  const [response, setResponse] = useState<ResponseState>(EMPTY_RESPONSE);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // 单调递增请求序号
  const requestSeqRef = useRef(0);
  // AbortController
  const abortRef = useRef<AbortController | null>(null);

  // 依赖变化时清空旧数据
  useEffect(() => {
    setResponse(EMPTY_RESPONSE);
    setError(null);
  }, [serviceId, timeRange, enabled]);

  const load = useCallback(async () => {
    if (!enabled || !serviceId) {
      setLoading(false);
      return;
    }

    // 取消旧的未完成请求
    if (abortRef.current) {
      abortRef.current.abort();
    }
    const controller = new AbortController();
    abortRef.current = controller;

    const seq = ++requestSeqRef.current;
    setLoading(true);
    setError(null);

    try {
      const allMetrics = CHART_GROUPS.flatMap(g => g.metrics);
      const r = await fetchTimeseries(serviceId, allMetrics, timeRange, {
        force: forceRefresh === true,
        signal: controller.signal,
      });

      if (seq !== requestSeqRef.current) return;

      // 原子替换整个 response state — 不会出现 data/axis 混合状态
      // 优先使用 display_step_seconds，回退到 step_seconds（向后兼容）
      const displayStep = (r.display_step_seconds != null && Number.isFinite(r.display_step_seconds) && r.display_step_seconds > 0)
        ? r.display_step_seconds
        : (r.step_seconds != null && Number.isFinite(r.step_seconds) && r.step_seconds > 0)
          ? r.step_seconds : 30;
      setResponse({
        range: r.range || null,
        queryStartMs: Number.isFinite(r.query_start_ms) ? r.query_start_ms : null,
        queryEndMs: Number.isFinite(r.query_end_ms) ? r.query_end_ms : null,
        displayStepSeconds: displayStep,
        calculationWindowSeconds: (r.calculation_window_seconds != null && Number.isFinite(r.calculation_window_seconds) && r.calculation_window_seconds > 0)
          ? r.calculation_window_seconds : 120,
        data: r.data || {},
      });
    } catch (e: any) {
      if (e?.name === 'CanceledError' || e?.code === 'ERR_CANCELED') return;
      if (seq !== requestSeqRef.current) return;
      setResponse(EMPTY_RESPONSE);
      setError('数据加载失败，请稍后重试');
    } finally {
      if (seq === requestSeqRef.current) {
        setLoading(false);
      }
    }
  }, [serviceId, timeRange, enabled, refreshToken, forceRefresh]);

  useEffect(() => { load(); }, [load]);

  // 卸载时取消请求
  useEffect(() => {
    return () => {
      requestSeqRef.current += 1;
      if (abortRef.current) {
        abortRef.current.abort();
        abortRef.current = null;
      }
    };
  }, []);

  // 状态 1：disabled
  if (!enabled) {
    return (
      <div style={{ marginTop: 12, background: '#fff', borderRadius: 4, padding: 24 }}>
        <Empty description={disabledReason === 'service_stopped' ? '服务已停止，暂无监控数据' : '该服务尚未启用推理监控'} />
      </div>
    );
  }

  // 状态 2：loading（首次）
  if (loading && response.queryStartMs == null) {
    return (
      <div style={{ marginTop: 12, background: '#fff', borderRadius: 4, padding: 24 }}>
        <Spin spinning><div style={{ height: 200 }} /></Spin>
      </div>
    );
  }

  // 状态 3：API error（且无有效窗口）
  if (error && response.queryStartMs == null) {
    return (
      <div style={{ marginTop: 12, background: '#fff', borderRadius: 4, padding: 24 }}>
        <Empty description={error} />
      </div>
    );
  }

  // 状态 4：API 成功但无有效查询窗口 → 不应出现（后端保证返回窗口），兜底
  if (response.queryStartMs == null || response.queryEndMs == null) {
    return (
      <div style={{ marginTop: 12, background: '#fff', borderRadius: 4, padding: 24 }}>
        <Empty description="暂无趋势数据" />
      </div>
    );
  }

  // 状态 5：有有效查询窗口 → 始终渲染四张标准坐标图
  // 无数据时图表仍渲染完整坐标轴（xAxis 范围来自查询窗口），不再叠加空态文案
  const { queryStartMs, queryEndMs, displayStepSeconds, data } = response;

  function isGroupUnsupported(metrics: readonly string[]): boolean {
    if (!unsupportedMetrics || unsupportedMetrics.size === 0) return false;
    return metrics.every(m => unsupportedMetrics.has(m));
  }

  return (
    <Spin spinning={loading}>
      <div className="trend-charts-grid">
        {CHART_GROUPS.map(g => {
          const groupUnsupported = isGroupUnsupported(g.metrics);
          return (
            <Card key={g.title} size="small" className="trend-chart-card"
              title={<span style={{ fontSize: 13 }}>{g.title}
                {'hint' in g && g.hint
                  ? <span style={{ fontSize: 11, color: '#999', marginLeft: 8, whiteSpace: 'nowrap' }}>{g.hint}</span>
                  : null}
              </span>}>
              <div className="trend-chart-content">
                <EchartCore
                  option={buildChartOption({
                    group: { ...g, unsupported: groupUnsupported },
                    data,
                    timeRange,
                    queryStartMs,
                    queryEndMs,
                    stepSeconds: displayStepSeconds,
                    displayStepSeconds,
                  })}
                  style={{ width: '100%', height: '100%' }}
                  preserveLegendSelection
                  legendStateKey={`${serviceId}:${g.title}`}
                />
              </div>
            </Card>
          );
        })}
      </div>
    </Spin>
  );
};

export default TrendCharts;
