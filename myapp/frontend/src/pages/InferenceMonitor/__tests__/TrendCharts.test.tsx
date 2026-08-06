/**
 * TrendCharts 组件测试（需求十二·前端部分）。
 *
 * 覆盖：
 * 1. 24h 有历史点时不能进入 empty 状态（无"暂无请求数据"文字）
 * 2. 无数据时仍渲染坐标轴（EchartCore 收到带 xAxis.min/max 的 option）
 * 3. 切换窗口时旧请求不能覆盖新请求（sequence + AbortController）
 * 4. 请求负载图三条 series 始终存在
 */
import React from 'react';
import { render, screen, waitFor, act } from '@testing-library/react';
import TrendCharts from '../components/TrendCharts';
import { fetchTimeseries } from '../api';

// Mock EchartCore：捕获每次渲染的 option
const capturedOptions: any[] = [];
jest.mock('../../../components/EchartCore/EchartCore', () => ({
  __esModule: true,
  default: (props: any) => {
    capturedOptions.push(props.option);
    return <div data-testid="echart" />;
  },
}));

jest.mock('../api', () => ({
  fetchTimeseries: jest.fn(),
}));

const mockedFetch = fetchTimeseries as jest.MockedFunction<typeof fetchTimeseries>;

const WINDOW = {
  query_start_ms: 1785980340000,
  query_end_ms: 1785983940000,
  display_step_seconds: 30,
  calculation_window_seconds: 120,
};

function makeResult(data: any, overrides: any = {}) {
  return {
    range: '1h',
    ...WINDOW,
    step_seconds: 30,
    data,
    ...overrides,
  } as any;
}

beforeEach(() => {
  capturedOptions.length = 0;
  mockedFetch.mockReset();
});

describe('TrendCharts', () => {
  it('有数据时渲染四张图且无"暂无请求数据"文字', async () => {
    mockedFetch.mockResolvedValue(makeResult({
      ttft_p95: [{ timestamp_ms: WINDOW.query_start_ms + 60000, value: 200 }],
      qps: [{ timestamp_ms: WINDOW.query_start_ms + 60000, value: 0.3 }],
    }));
    render(<TrendCharts serviceId={76} timeRange="1h" enabled />);
    await waitFor(() => expect(screen.getAllByTestId('echart')).toHaveLength(4));
    expect(screen.queryByText('暂无请求数据')).toBeNull();
    expect(screen.queryByText('暂无趋势数据')).toBeNull();
    expect(screen.getByText('请求负载')).toBeTruthy();
    // 简洁提示文案
    expect(screen.getByText('并发/排队：桶内峰值')).toBeTruthy();
    expect(screen.queryByText(/不是吞吐量/)).toBeNull();
    expect(screen.queryByText(/Gauge 桶内最大观测值/)).toBeNull();
    // 每张图 option 的 xAxis 使用查询窗口
    for (const opt of capturedOptions) {
      expect(opt.xAxis.min).toBe(WINDOW.query_start_ms);
      expect(opt.xAxis.max).toBe(WINDOW.query_end_ms);
    }
    // 请求负载图（第三张）双 Y 轴 + 三 series
    const loadOpt = capturedOptions[2];
    expect(loadOpt.series).toHaveLength(3);
    expect(loadOpt.yAxis).toHaveLength(2);
  });

  it('全部无数据时仍渲染坐标轴，不出现空态文案', async () => {
    mockedFetch.mockResolvedValue(makeResult({}));
    render(<TrendCharts serviceId={76} timeRange="24h" enabled />);
    await waitFor(() => expect(screen.getAllByTestId('echart')).toHaveLength(4));
    expect(screen.queryByText('暂无请求数据')).toBeNull();
    for (const opt of capturedOptions) {
      expect(opt.xAxis.min).toBe(WINDOW.query_start_ms);
      expect(opt.xAxis.max).toBe(WINDOW.query_end_ms);
      expect(opt.yAxis).toBeDefined();
      expect(opt.graphic).toBeUndefined();
    }
  });

  it('切换窗口时旧请求后返回不能覆盖新请求', async () => {
    // 旧请求（5m）晚返回，新请求（24h）先返回
    let resolveOld: (v: any) => void = () => {};
    const oldPromise = new Promise((res) => { resolveOld = res; });
    const newWindow = { query_start_ms: 1785897300000, query_end_ms: 1785983700000 };

    mockedFetch
      .mockImplementationOnce(() => oldPromise as any)                       // 5m 请求挂起
      .mockResolvedValueOnce(makeResult({                                    // 24h 请求立即返回
        qps: [{ timestamp_ms: 1785980700000, value: 0.1 }],
      }, { range: '24h', ...newWindow, display_step_seconds: 300 }));

    const { rerender } = render(<TrendCharts serviceId={76} timeRange="5m" enabled />);
    // 立即切到 24h
    rerender(<TrendCharts serviceId={76} timeRange="24h" enabled />);
    await waitFor(() => expect(capturedOptions.length).toBeGreaterThan(0));
    expect(capturedOptions[0].xAxis.min).toBe(newWindow.query_start_ms);

    // 旧 5m 请求随后返回 — 必须被丢弃
    const before = capturedOptions.length;
    await act(async () => {
      resolveOld(makeResult({}, { range: '5m' }));
    });
    await waitFor(() => expect(capturedOptions.length).toBe(before));
    expect(capturedOptions[capturedOptions.length - 1].xAxis.min).toBe(newWindow.query_start_ms);
  });

  it('旧请求被取消时带 AbortSignal', async () => {
    mockedFetch.mockResolvedValue(makeResult({}));
    const { rerender } = render(<TrendCharts serviceId={76} timeRange="5m" enabled />);
    rerender(<TrendCharts serviceId={76} timeRange="15m" enabled />);
    await waitFor(() => expect(mockedFetch).toHaveBeenCalledTimes(2));
    const firstSignal = mockedFetch.mock.calls[0][3]?.signal;
    expect(firstSignal?.aborted).toBe(true);
  });
});
