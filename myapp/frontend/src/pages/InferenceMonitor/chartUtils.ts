/**
 * 推理监控图表公共工具函数。
 * 所有 vLLM、SGLang 及未来新增推理服务共用。
 * 不依赖 service_id、服务名称或模型名称。
 *
 * 核心约定：
 * - 所有时间戳统一为毫秒
 * - xAxis 由后端 queryStartMs/queryEndMs 控制，前端不重新计算
 * - series.data 只包含真实 Prometheus 数据点，不注入虚拟 null 点
 * - connectNulls = false
 * - 禁止创建 __timeline__ 等虚假 series
 */
import moment from 'moment';
import type { TimeRange, MetricPoint } from './types';
import type { EChartsOption } from 'echarts';

// ====== 指标元数据 ======

export interface MetricMeta {
  label: string;
  unit: string;
  precision: number;
}

export const METRIC_META: Record<string, MetricMeta> = {
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

// ====== 数据规范化 ======

/**
 * 将后端 MetricPoint[] 规范化为 ECharts 可用格式 [timestampMs, number | null]。
 * - 过滤无效时间戳（<=0, NaN, Infinity）
 * - 过滤查询窗口外数据
 * - NaN/Infinity 值 → null
 * - 相同时间戳去重（保留最后一个）
 * - 严格升序排列
 */
export function normalizePoints(
  rawPoints: MetricPoint[],
  queryStartMs: number,
  queryEndMs: number,
): [number, number | null][] {
  if (!rawPoints || rawPoints.length === 0) return [];

  // 去重 + 规范化
  const seen = new Map<number, number | null>();
  for (const p of rawPoints) {
    if (!Number.isFinite(p.timestamp_ms) || p.timestamp_ms <= 0) continue;
    if (p.timestamp_ms < queryStartMs || p.timestamp_ms > queryEndMs) continue;
    const valid = (p.value != null && Number.isFinite(p.value)) ? p.value : null;
    seen.set(p.timestamp_ms, valid);
  }

  // 升序排列
  return Array.from(seen.entries())
    .sort(([a], [b]) => a - b)
    .map(([ts, v]) => [ts, v] as [number, number | null]);
}

// ====== 断线标记 ======

/**
 * 在稀疏数据点之间插入 null 断点，防止 ECharts 跨长时间区间连线。
 *
 * 当相邻两个点之间的时间间隔超过 gapThresholdMs 时，在两点之间插入一个 null 标记。
 * 只插入单个 null 点作为断线信号，不生成全窗口稠密 timeline。
 *
 * @param points  已排序、去重的 [timestampMs, value|null] 数组
 * @param stepSeconds  查询窗口步长（秒），用于计算期望间隔
 * @param scrapeIntervalSeconds  Prometheus 抓取间隔（秒），默认 15
 */
export function insertGapBreaks(
  points: [number, number | null][],
  stepSeconds: number,
  scrapeIntervalSeconds: number = 15,
): [number, number | null][] {
  if (points.length < 2) return points;

  const expectedIntervalMs = stepSeconds * 1000;
  const gapThresholdMs = Math.max(
    expectedIntervalMs * 2.5,
    scrapeIntervalSeconds * 1000 * 3,
  );

  const result: [number, number | null][] = [];

  for (let i = 0; i < points.length; i++) {
    result.push(points[i]);

    if (i < points.length - 1) {
      const currentTs = points[i][0];
      const nextTs = points[i + 1][0];
      const gap = nextTs - currentTs;

      if (gap > gapThresholdMs) {
        // 在两点之间插入一个 null 断点，位置取当前点 + 一个期望间隔
        const breakTs = currentTs + expectedIntervalMs;
        if (breakTs < nextTs) {
          result.push([breakTs, null]);
        }
      }
    }
  }

  return result;
}

// ====== 孤立单桶样本的短线展开 ======

/** ECharts 数据项：value 为 [timestampMs, value|null]；bucketTs 记录统计桶原始时间 */
export interface ChartDataItem {
  value: [number, number | null];
  /** 单桶展开时记录桶原始中心时间，tooltip 仍显示该时间 */
  bucketTs?: number;
}

/**
 * 将孤立的单桶样本（前后均无相邻有效点的单点）在渲染层展开为一条短水平线：
 * [t - step/2, v] → [t + step/2, v]，两端使用同一个真实值。
 *
 * 这只是单个统计桶的可视化展开，不生成新的监控值；
 * bucketTs 保留桶原始时间供 tooltip 显示。不使用圆点代替。
 */
export function expandIsolatedBuckets(
  points: [number, number | null][],
  stepSeconds: number,
): ChartDataItem[] {
  const halfMs = (stepSeconds * 1000) / 2;
  const result: ChartDataItem[] = [];

  const isValid = (p: [number, number | null] | undefined) =>
    !!p && p[1] != null && Number.isFinite(p[1]);

  for (let i = 0; i < points.length; i++) {
    const [ts, v] = points[i];
    if (v == null || !Number.isFinite(v)) {
      result.push({ value: [ts, null] });
      continue;
    }
    const prevValid = isValid(points[i - 1]);
    const nextValid = isValid(points[i + 1]);
    if (!prevValid && !nextValid) {
      // 孤立单桶 → 展开为短水平线
      result.push({ value: [ts - halfMs, v], bucketTs: ts });
      result.push({ value: [ts + halfMs, v], bucketTs: ts });
    } else {
      result.push({ value: [ts, v] });
    }
  }
  return result;
}

// ====== 时间格式化 ======

export function formatTooltipTime(timestampMs: number, _range: TimeRange): string {
  const m = moment(timestampMs);
  if (!m.isValid()) return '-';
  return m.format('YYYY-MM-DD HH:mm:ss');
}

export function formatAxisTime(value: number, range: TimeRange): string {
  const m = moment(value);
  if (!m.isValid()) return '';
  if (range === '5m' || range === '15m' || range === '1h' || range === '6h') {
    return m.format('HH:mm');
  }
  return m.format('MM-DD HH:mm');
}

// ====== 固定 X 轴刻度 ======
// 每个 range 使用确定的 tick 间隔（毫秒）与格式，不交给 ECharts 自动推导，
// 保证同一 range 重复刷新、切换 service 后刻度规则稳定。
export const XAXIS_TICK_CONFIG: Record<TimeRange, { intervalMs: number }> = {
  '5m':  { intervalMs: 60 * 1000 },          // 1 分钟
  '15m': { intervalMs: 3 * 60 * 1000 },      // 3 分钟
  '1h':  { intervalMs: 10 * 60 * 1000 },     // 10 分钟
  '6h':  { intervalMs: 60 * 60 * 1000 },     // 1 小时
  '24h': { intervalMs: 4 * 3600 * 1000 },    // 4 小时
  '3d':  { intervalMs: 12 * 3600 * 1000 },   // 12 小时
};

// ====== 数值格式化 ======

export function formatMetricValue(value: unknown, precision: number): string {
  const n = Number(value);
  if (!Number.isFinite(n)) return '-';
  return n.toFixed(precision);
}

// ====== 稳定纵轴计算 ======

export function calculateStableYAxisRange(
  seriesDataList: (ChartDataItem[] | undefined)[],
): { min: number; max: number } {
  const values: number[] = [];
  for (const pts of seriesDataList) {
    if (!pts) continue;
    for (const item of pts) {
      const v = item?.value?.[1];
      if (v !== null && v !== undefined && Number.isFinite(v)) {
        values.push(v);
      }
    }
  }
  if (values.length === 0) return { min: 0, max: 1 };
  const rawMax = Math.max(0, ...values);
  if (rawMax === 0) return { min: 0, max: 1 };
  return { min: 0, max: calculateNiceAxisMax(rawMax) };
}

export function calculateNiceAxisMax(rawMax: number): number {
  if (rawMax <= 0) return 1;
  const padded = rawMax * 1.15;
  if (padded <= 1) return Math.ceil(padded * 10) / 10;
  const magnitude = Math.pow(10, Math.floor(Math.log10(padded)));
  const normalized = padded / magnitude;
  let nice: number;
  if (normalized <= 1) nice = 1;
  else if (normalized <= 2) nice = 2;
  else if (normalized <= 5) nice = 5;
  else nice = 10;
  return nice * magnitude;
}

// ====== 公共 ECharts Option 构建 ======

export interface ChartGroupConfig {
  title: string;
  metrics: readonly string[];
  unsupported?: boolean;
  /** 左 Y 轴单位（显示为轴名称） */
  yUnit?: string;
  /** 右 Y 轴单位；设置后启用双 Y 轴 */
  rightYUnit?: string;
  /** 绑定到右 Y 轴的指标 key */
  rightMetrics?: readonly string[];
}

export interface BuildOptionsInput {
  group: ChartGroupConfig;
  data: Record<string, MetricPoint[]>;
  timeRange: TimeRange;
  queryStartMs: number;
  queryEndMs: number;
  /** display step in seconds — for gap detection only, NOT for calculation */
  stepSeconds?: number;
  /** @deprecated alias for stepSeconds */
  displayStepSeconds?: number;
}

/** 公共横轴配置：min/max 来自查询窗口，tick 间隔按 range 固定 */
function buildCommonXAxis(
  timeRange: TimeRange,
  queryStartMs: number,
  queryEndMs: number,
): any {
  const tick = XAXIS_TICK_CONFIG[timeRange];
  return {
    type: 'time' as const,
    min: queryStartMs,
    max: queryEndMs,
    interval: tick?.intervalMs,
    axisLabel: {
      fontSize: 10,
      margin: 12,
      hideOverlap: true,
      formatter: (value: number) => formatAxisTime(value, timeRange),
    },
    axisTick: { alignWithLabel: false },
    boundaryGap: false,
  };
}

/** 公共 Tooltip formatter：seriesName 是展示标签，需经 labelToMeta 反查指标元数据 */
function buildTooltipFormatter(
  timeRange: TimeRange,
  metricNames: readonly string[],
): (params: any) => string {
  const labelToMeta = new Map<string, MetricMeta>();
  for (const m of metricNames) {
    const meta = METRIC_META[m];
    if (meta) labelToMeta.set(meta.label, meta);
  }
  return (params: any) => {
    const items = Array.isArray(params) ? params : [params];
    if (items.length === 0) return '';

    // 单桶展开的点携带 bucketTs（统计桶原始时间），优先于渲染用的时间边界
    const firstData = items[0]?.data;
    const axisTs = Number(
      (firstData && typeof firstData === 'object' && !Array.isArray(firstData) && firstData.bucketTs)
        ?? items[0]?.axisValue,
    );
    const timeStr = Number.isFinite(axisTs) ? formatTooltipTime(axisTs, timeRange) : '-';

    let body = `<div style="font-weight:600;margin-bottom:4px">${timeStr}</div>`;
    let hasData = false;

    for (const item of items) {
      const name = item.seriesName || '';
      if (name === '__timeline__') continue;

      const meta = labelToMeta.get(name) || METRIC_META[name];
      const label = meta?.label || name;
      const unit = meta?.unit || '';
      const prec = meta?.precision ?? 2;

      const val = item.value?.[1];
      const displayVal = (val != null && Number.isFinite(Number(val)))
        ? formatMetricValue(val, prec)
        : '—';
      const color = item.color || '#333';

      body += `<div><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${color};margin-right:6px"></span>${label}: ${displayVal} ${unit}</div>`;
      if (val != null && Number.isFinite(Number(val))) hasData = true;
    }

    if (!hasData) {
      body += '<div style="color:#999;margin-top:4px">该时间点无监控样本</div>';
    }

    return body;
  };
}

/** 构建单个图表组的 ECharts option */
export function buildChartOption(input: BuildOptionsInput): EChartsOption {
  const { group, data, timeRange, queryStartMs, queryEndMs } = input;
  const metrics = group.metrics;
  const rightMetrics = new Set(group.rightMetrics ?? []);
  const hasRightAxis = rightMetrics.size > 0;

  // 规范化每个指标：去重、排序、过滤、NaN→null + 插入断线标记 + 孤立单桶短线展开
  const stepS = input.stepSeconds ?? 60;
  const normalizedSeries = metrics.map((m) => {
    const rawPoints = data[m] || [];
    const pts = normalizePoints(rawPoints, queryStartMs, queryEndMs);
    // 在稀疏数据之间插入 null 断点，防止跨长时间区间连线
    const withGapBreaks = insertGapBreaks(pts, stepS, 15);
    // 孤立单桶样本展开为短水平线（纯渲染层，不产生新数据）
    const items = expandIsolatedBuckets(withGapBreaks, stepS);
    const meta = METRIC_META[m] || { label: m, unit: '', precision: 2 };
    return { name: meta.label, data: items, metricKey: m };
  });

  // xAxis：始终使用后端查询窗口，不从 series 有效点推导
  const xAxis = buildCommonXAxis(timeRange, queryStartMs, queryEndMs);

  // yAxis：稳定的纵轴范围（双轴时分别计算）
  const leftRange = calculateStableYAxisRange(
    normalizedSeries.filter(s => !rightMetrics.has(s.metricKey)).map(s => s.data),
  );
  const rightRange = hasRightAxis
    ? calculateStableYAxisRange(
        normalizedSeries.filter(s => rightMetrics.has(s.metricKey)).map(s => s.data),
      )
    : null;

  const makeAxis = (range: { min: number; max: number }, unit?: string, showSplit = true): any => ({
    type: 'value' as const,
    name: unit || '',
    nameTextStyle: { fontSize: 10, color: '#999' },
    min: range.min,
    max: range.max,
    axisLine: { show: true },
    axisTick: { show: true },
    axisLabel: { show: true, fontSize: 10, margin: 8 },
    splitLine: { show: showSplit, lineStyle: { type: 'dashed', color: '#f0f0f0' } },
  });

  const yAxis: any = hasRightAxis
    ? [makeAxis(leftRange, group.yUnit), makeAxis(rightRange!, group.rightYUnit, false)]
    : makeAxis(leftRange, group.yUnit);

  // ECharts series：始终为每个真实指标创建 series（即使全为空或全为 0）
  // 数据为空时 series.data=[]，坐标轴由 xAxis.min/max 保证
  // 所有曲线只显示线条，永不显示数据圆点；孤立单桶已由渲染层展开为短线
  const chartSeries: any[] = normalizedSeries.map((s) => ({
    name: s.name,
    type: 'line',
    data: s.data,
    smooth: false,
    connectNulls: false,
    showSymbol: false,
    symbol: 'none',
    symbolSize: 0,
    lineStyle: { width: 2 },
    yAxisIndex: hasRightAxis && rightMetrics.has(s.metricKey) ? 1 : 0,
  }));

  const legendNames = metrics.map(m => (METRIC_META[m] || { label: m }).label);

  const grid = { top: 24, right: hasRightAxis ? 56 : 24, bottom: 52, left: 56, containLabel: true };

  return {
    tooltip: {
      trigger: 'axis',
      triggerOn: 'mousemove|click',
      axisPointer: { type: 'line', snap: true },
      formatter: buildTooltipFormatter(timeRange, metrics as readonly string[]),
    },
    legend: {
      bottom: 8, left: 'center', itemWidth: 14, itemHeight: 8, itemGap: 18,
      textStyle: { fontSize: 12 },
      data: legendNames.map(name => ({ name })),
    },
    grid,
    xAxis,
    yAxis,
    series: chartSeries,
    animation: false,
  };
}

/** 测试专用导出：tooltip formatter（生产代码通过 buildChartOption 内部使用） */
export const buildTooltipFormatterForTest = buildTooltipFormatter;
