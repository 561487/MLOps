/** 指标数据点 */
export interface MetricPoint {
  timestamp_ms: number;
  value: number | null;
}

/** 单个指标响应结构 */
export interface MetricValue {
  value: number | null;
  status: string;
  unit: string;
  message: string;
}

/** 服务列表项 */
export interface ServiceItem {
  service_id: number;
  service_name: string;
  label: string;
  model_name: string;
  model_status: string;
  service_type: string;
  project_id: number;
  project_name: string;
  namespace: string;
  ready: boolean | null;
  up_status: number | null;
}

/** 摘要接口响应 */
export interface SummaryResult {
  service_id: number;
  service_name: string;
  label: string;
  model_name: string;
  model_status: string;
  service_type: string;
  namespace: string;
  ready: boolean;
  monitor_status: string;
  metrics: Record<string, MetricValue>;
}

/** 时间序列响应 */
export interface TimeseriesResult {
  data: Record<string, MetricPoint[]>;
}

/** 时间范围选项 */
export type TimeRange = '5m' | '15m' | '1h' | '6h' | '24h' | '3d';

/** 第一阶段支持的引擎 */
export type EngineType = 'vllm';

/** 运行状态筛选 */
export type StatusFilter = 'all' | 'online' | 'offline';

/** API 通用响应 */
export interface ApiResponse<T> {
  status: number;
  message: string;
  result: T;
}

export const METRIC_UNITS: Record<string, string> = {
  ttft_p50: 'ms',
  ttft_p95: 'ms',
  ttft_p99: 'ms',
  itl_p50: 'ms/token',
  itl_p95: 'ms/token',
  qps: 'req/s',
  running_requests: 'requests',
  waiting_requests: 'requests',
  input_tokens_per_second: 'tokens/s',
  output_tokens_per_second: 'tokens/s',
};

export const METRIC_LABELS: Record<string, string> = {
  ttft_p50: 'TTFT P50',
  ttft_p95: 'TTFT P95',
  ttft_p99: 'TTFT P99',
  itl_p50: 'ITL P50',
  itl_p95: 'ITL P95',
  qps: '完成请求 QPS',
  running_requests: '运行请求数',
  waiting_requests: '等待请求数',
  input_tokens_per_second: '输入 Token/s',
  output_tokens_per_second: '输出 Token/s',
};

export const TIME_RANGE_OPTIONS: { label: string; value: TimeRange }[] = [
  { label: '最近 5 分钟', value: '5m' },
  { label: '最近 15 分钟', value: '15m' },
  { label: '最近 1 小时', value: '1h' },
  { label: '最近 6 小时', value: '6h' },
  { label: '最近 24 小时', value: '24h' },
  { label: '最近 3 天', value: '3d' },
];

export const MONITOR_STATUS_LABELS: Record<string, string> = {
  normal: '正常',
  no_data: '暂无请求数据',
  not_configured: '未接入监控',
  scrape_failed: '采集异常',
  query_failed: '监控服务不可用',
  service_stopped: '服务已停止',
  unsupported: '不适用',
};

export const MONITOR_STATUS_COLORS: Record<string, string> = {
  normal: '#52c41a',
  no_data: '#faad14',
  not_configured: '#d9d9d9',
  scrape_failed: '#ff4d4f',
  query_failed: '#ff4d4f',
  service_stopped: '#d9d9d9',
  unsupported: '#d9d9d9',
};
