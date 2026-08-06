/**
 * 推理监控图表工具测试（峰值语义 v3）。
 *
 * 覆盖（需求十三·前端）：
 * - 所有 series 的 showSymbol=false 且 symbol='none'（无圆点）
 * - 单桶数据渲染为短线（两个端点 + bucketTs），而不是圆点
 * - 5m~3d 的 X 轴 interval 固定映射
 * - X 轴 min/max 来自 query window
 * - 同一 range 重复构建 option 刻度规则相同
 * - 无数据仍渲染坐标轴且无空态文案
 * - 延迟 null 不绘制成 0
 * - 请求负载双 Y 轴、三 series 恒存在
 * - Tooltip 单位正确；单桶 tooltip 显示桶原始时间
 */
import {
  buildChartOption,
  normalizePoints,
  insertGapBreaks,
  expandIsolatedBuckets,
  formatAxisTime,
  buildTooltipFormatterForTest,
  XAXIS_TICK_CONFIG,
  METRIC_META,
} from '../chartUtils';

const START = 1785980340000;
const END = 1785983940000;

const LOAD_GROUP = {
  title: '请求负载',
  metrics: ['qps', 'running_requests', 'waiting_requests'] as readonly string[],
  yUnit: 'req/s',
  rightYUnit: 'requests',
  rightMetrics: ['running_requests', 'waiting_requests'] as readonly string[],
};

const TTFT_GROUP = {
  title: 'TTFT 延迟',
  metrics: ['ttft_p50', 'ttft_p95', 'ttft_p99'] as readonly string[],
  yUnit: 'ms',
};

describe('无圆点渲染', () => {
  it('所有 series 的 showSymbol=false 且 symbol=none、symbolSize=0', () => {
    const option: any = buildChartOption({
      group: TTFT_GROUP,
      data: { ttft_p95: [{ timestamp_ms: START + 60000, value: 200 }] },
      timeRange: '1h', queryStartMs: START, queryEndMs: END, stepSeconds: 30,
    });
    expect(option.series).toHaveLength(3);
    for (const s of option.series) {
      expect(s.showSymbol).toBe(false);
      expect(s.symbol).toBe('none');
      expect(s.symbolSize).toBe(0);
      expect(s.connectNulls).toBe(false);
    }
    expect(JSON.stringify(option)).not.toContain('circle');
  });

  it('孤立单桶样本展开为短水平线而非圆点', () => {
    const t = START + 60000;
    const option: any = buildChartOption({
      group: TTFT_GROUP,
      data: { itl_like_single: [], ttft_p50: [{ timestamp_ms: t, value: 150 }] },
      timeRange: '1h', queryStartMs: START, queryEndMs: END, stepSeconds: 30,
    });
    const s = option.series.find((x: any) => x.name === 'TTFT P50');
    // 单桶 → 两个端点，同一真实值，携带桶原始时间
    expect(s.data).toHaveLength(2);
    expect(s.data[0].value).toEqual([t - 15000, 150]);
    expect(s.data[1].value).toEqual([t + 15000, 150]);
    expect(s.data[0].bucketTs).toBe(t);
    expect(s.data[1].bucketTs).toBe(t);
  });

  it('连续多点不做展开', () => {
    const pts: [number, number | null][] = [
      [START, 1], [START + 30000, 2], [START + 60000, 3],
    ];
    const out = expandIsolatedBuckets(pts, 30);
    expect(out).toHaveLength(3);
    expect(out.every(d => d.bucketTs === undefined)).toBe(true);
  });

  it('被 null 隔开的两个孤点各自展开', () => {
    const pts: [number, number | null][] = [
      [START, 1], [START + 30000, null], [START + 60000, 2],
    ];
    const out = expandIsolatedBuckets(pts, 30);
    const valid = out.filter(d => d.value[1] !== null);
    expect(valid).toHaveLength(4); // 两个孤点 × 两端
  });
});

describe('固定 X 轴刻度', () => {
  it('六个 range 的 interval 固定映射', () => {
    expect(XAXIS_TICK_CONFIG['5m'].intervalMs).toBe(60000);
    expect(XAXIS_TICK_CONFIG['15m'].intervalMs).toBe(180000);
    expect(XAXIS_TICK_CONFIG['1h'].intervalMs).toBe(600000);
    expect(XAXIS_TICK_CONFIG['6h'].intervalMs).toBe(3600000);
    expect(XAXIS_TICK_CONFIG['24h'].intervalMs).toBe(14400000);
    expect(XAXIS_TICK_CONFIG['3d'].intervalMs).toBe(43200000);
  });

  it('option.xAxis 携带固定 interval，min/max 来自查询窗口', () => {
    for (const range of ['5m', '15m', '1h', '6h', '24h', '3d'] as const) {
      const option: any = buildChartOption({
        group: TTFT_GROUP, data: {}, timeRange: range,
        queryStartMs: START, queryEndMs: END, stepSeconds: 15,
      });
      expect(option.xAxis.interval).toBe(XAXIS_TICK_CONFIG[range].intervalMs);
      expect(option.xAxis.min).toBe(START);
      expect(option.xAxis.max).toBe(END);
      expect(option.xAxis.type).toBe('time');
    }
  });

  it('同一 range 重复构建 option 刻度规则一致', () => {
    const build = () => buildChartOption({
      group: TTFT_GROUP, data: {}, timeRange: '24h',
      queryStartMs: START, queryEndMs: END, stepSeconds: 300,
    }) as any;
    const a = build();
    const b = build();
    expect(a.xAxis.interval).toBe(b.xAxis.interval);
    expect(a.xAxis.axisLabel.formatter(START)).toBe(b.xAxis.axisLabel.formatter(START));
  });

  it('时间格式随 range 固定', () => {
    expect(formatAxisTime(START, '5m')).toMatch(/^\d{2}:\d{2}$/);
    expect(formatAxisTime(START, '15m')).toMatch(/^\d{2}:\d{2}$/);
    expect(formatAxisTime(START, '1h')).toMatch(/^\d{2}:\d{2}$/);
    expect(formatAxisTime(START, '6h')).toMatch(/^\d{2}:\d{2}$/);
    expect(formatAxisTime(START, '24h')).toMatch(/^\d{2}-\d{2} \d{2}:\d{2}$/);
    expect(formatAxisTime(START, '3d')).toMatch(/^\d{2}-\d{2} \d{2}:\d{2}$/);
  });
});

describe('空数据与 null 语义', () => {
  it('无数据仍渲染坐标轴且无空态文案', () => {
    const option: any = buildChartOption({
      group: TTFT_GROUP, data: {}, timeRange: '24h',
      queryStartMs: START, queryEndMs: END, stepSeconds: 300,
    });
    const s = JSON.stringify(option);
    expect(s).not.toContain('暂无请求数据');
    expect(option.graphic).toBeUndefined();
    expect(option.xAxis).toBeDefined();
    expect(option.yAxis).toBeDefined();
    expect(option.tooltip).toBeDefined();
    expect(option.series).toHaveLength(3);
  });

  it('延迟 null 不绘制成 0', () => {
    const option: any = buildChartOption({
      group: TTFT_GROUP,
      data: { ttft_p95: [
        { timestamp_ms: START + 30000, value: null },
        { timestamp_ms: START + 60000, value: 200 },
        { timestamp_ms: START + 90000, value: 210 },
      ] },
      timeRange: '5m', queryStartMs: START, queryEndMs: END, stepSeconds: 15,
    });
    const p95 = option.series.find((x: any) => x.name === 'TTFT P95');
    const values = p95.data.map((d: any) => d.value[1]);
    expect(values).toContain(null);
    expect(values).not.toContain(0);
  });

  it('normalizePoints 过滤窗口外与非法点', () => {
    const pts = normalizePoints([
      { timestamp_ms: START - 1000, value: 5 },
      { timestamp_ms: START + 1000, value: NaN },
      { timestamp_ms: START + 2000, value: 3 },
      { timestamp_ms: 0, value: 9 },
    ], START, END);
    expect(pts).toEqual([[START + 1000, null], [START + 2000, 3]]);
  });
});

describe('请求负载双 Y 轴', () => {
  it('qps 左轴、running/waiting 右轴，三 series 恒存在', () => {
    const option: any = buildChartOption({
      group: LOAD_GROUP, data: {}, timeRange: '24h',
      queryStartMs: START, queryEndMs: END, stepSeconds: 300,
    });
    expect(option.yAxis).toHaveLength(2);
    expect(option.yAxis[0].name).toBe('req/s');
    expect(option.yAxis[1].name).toBe('requests');
    const byName = Object.fromEntries(option.series.map((s: any) => [s.name, s]));
    expect(byName['完成请求 QPS'].yAxisIndex).toBe(0);
    expect(byName['运行请求数'].yAxisIndex).toBe(1);
    expect(byName['等待请求数'].yAxisIndex).toBe(1);
    expect(option.legend.data.map((d: any) => d.name))
      .toEqual(['完成请求 QPS', '运行请求数', '等待请求数']);
  });

  it('running/waiting 全 0 时 series 仍保留', () => {
    const zeros = [
      { timestamp_ms: START + 1000, value: 0 },
      { timestamp_ms: START + 16000, value: 0 },
    ];
    const option: any = buildChartOption({
      group: LOAD_GROUP,
      data: { qps: zeros, running_requests: zeros, waiting_requests: zeros },
      timeRange: '5m', queryStartMs: START, queryEndMs: END, stepSeconds: 15,
    });
    expect(option.series).toHaveLength(3);
    for (const s of option.series) expect(s.data.length).toBeGreaterThan(0);
  });
});

describe('tooltip', () => {
  it('单位与精度正确', () => {
    const fmt = buildTooltipFormatterForTest('5m', LOAD_GROUP.metrics);
    const out = fmt([{
      axisValue: START + 1000,
      seriesName: '完成请求 QPS', value: [START + 1000, 0.5], color: '#111',
    }, {
      axisValue: START + 1000,
      seriesName: '运行请求数', value: [START + 1000, 3], color: '#222',
    }]);
    expect(out).toContain('req/s');
    expect(out).toContain('requests');
    expect(out).toContain('0.500');
  });

  it('单桶展开点的 tooltip 显示桶原始时间', () => {
    const t = START + 60000;
    const fmt = buildTooltipFormatterForTest('1h', TTFT_GROUP.metrics);
    const out = fmt([{
      axisValue: t - 15000,
      data: { value: [t - 15000, 200], bucketTs: t },
      seriesName: 'TTFT P95', value: [t - 15000, 200], color: '#111',
    }]);
    expect(out).toContain('200.00 ms');
    // header 时间应为桶原始时间 t，而非线段端点 t-15000
    const headerMatch = out.match(/(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})/);
    expect(headerMatch).toBeTruthy();
    const expected = new Date(t);
    expect(headerMatch![1]).toBe(
      `${expected.getFullYear()}-${String(expected.getMonth() + 1).padStart(2, '0')}-${String(expected.getDate()).padStart(2, '0')} ` +
      `${String(expected.getHours()).padStart(2, '0')}:${String(expected.getMinutes()).padStart(2, '0')}:${String(expected.getSeconds()).padStart(2, '0')}`,
    );
  });
});

describe('断线插入', () => {
  it('间隔超过阈值时插入 null 断点，正常间隔不插入', () => {
    const pts: [number, number | null][] = [
      [START, 1], [START + 15000, 2], [START + 15000 * 10, 3],
    ];
    const out = insertGapBreaks(pts, 15, 15);
    expect(out.filter(([, v]) => v === null).length).toBe(1);
    const dense: [number, number | null][] = [
      [START, 1], [START + 15000, 2], [START + 30000, 3],
    ];
    expect(insertGapBreaks(dense, 15, 15)).toHaveLength(3);
  });
});

describe('指标元数据', () => {
  it('所有指标都有中文标签和单位', () => {
    for (const [key, meta] of Object.entries(METRIC_META)) {
      expect(meta.label.length).toBeGreaterThan(0);
      expect(meta.unit.length).toBeGreaterThan(0);
      expect(key).toBe(key.toLowerCase());
    }
  });
});
