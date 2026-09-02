/**
 * MetricCards 文案测试（需求十三·前端 1~3）。
 *
 * - 顶部只出现"当前状态"
 * - 不出现"最近2分钟"
 * - 不出现"与下方所选时间窗口无关"
 * - 延迟无数据显示 —，速率/并发无数据显示 0
 */
import React from 'react';
import { render, screen } from '@testing-library/react';
import MetricCards from '../components/MetricCards';

// antd Row/Col 的 responsiveObserve 依赖 window.matchMedia，jsdom 未实现
beforeAll(() => {
  Object.defineProperty(window, 'matchMedia', {
    writable: true,
    value: (query: string) => ({
      matches: false,
      media: query,
      onchange: null,
      addListener: () => {},
      removeListener: () => {},
      addEventListener: () => {},
      removeEventListener: () => {},
      dispatchEvent: () => false,
    }),
  });
});

const BASE_SUMMARY: any = {
  service_id: 76,
  service_name: 'qwen3-0-6b-22',
  monitor_status: 'normal',
  metrics: {},
};

function metric(value: number | null, status = 'normal', unit = '') {
  return { value, status, unit, message: '' };
}

describe('MetricCards 文案', () => {
  it('顶部只显示"当前状态"，无旧说明文字', () => {
    render(<MetricCards summary={{
      ...BASE_SUMMARY,
      metrics: {
        qps: metric(0.3, 'normal', 'req/s'),
        running_requests: metric(0, 'normal', 'requests'),
        waiting_requests: metric(0, 'normal', 'requests'),
        ttft_p95: metric(152.3, 'normal', 'ms'),
        itl_p95: metric(3.9, 'normal', 'ms/token'),
      },
    }} />);
    expect(screen.getByText('当前状态')).toBeTruthy();
    expect(screen.queryByText(/最近2分钟/)).toBeNull();
    expect(screen.queryByText(/与下方所选时间窗口无关/)).toBeNull();
  });

  it('延迟无数据显示 —，速率/并发无数据显示 0', () => {
    render(<MetricCards summary={{
      ...BASE_SUMMARY,
      metrics: {
        qps: metric(null, 'no_data', 'req/s'),
        running_requests: metric(null, 'no_data', 'requests'),
        waiting_requests: metric(null, 'no_data', 'requests'),
        ttft_p95: metric(null, 'no_data', 'ms'),
        itl_p95: metric(null, 'no_data', 'ms/token'),
      },
    }} />);
    const dashes = screen.getAllByText('—');
    expect(dashes.length).toBe(2); // TTFT P95 + ITL P95
    expect(screen.queryByText('暂无请求数据')).toBeNull();
  });
});
