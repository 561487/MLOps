import type { ServiceItem } from './types';

/**
 * 用户界面统一的监控显示状态（仅 4 种）。
 * 后端完整状态码保持不变，仅前端聚合展示。
 */
export type MonitorDisplayStatus =
  | 'stopped'        // 服务已停止
  | 'normal'         // 监控正常（含 no_data：近期无请求但仍正常采集）
  | 'not_configured' // 未接入监控
  | 'error';         // 监控异常（Prometheus/Target 问题）

/** 统一状态映射：根据服务运行状态 + 后端 monitor_status 推导用户可见状态 */
export function getMonitorDisplayStatus(
  service: ServiceItem,
  monitorStatus?: string | null,
): MonitorDisplayStatus {
  // 已停止服务优先级最高
  if (service.model_status !== 'online') {
    return 'stopped';
  }

  switch (monitorStatus) {
    // 正常和无数据 → 监控正常
    case 'normal':
    case 'no_data':
      return 'normal';

    // 未配置
    case 'not_configured':
    case 'target_missing':
      return 'not_configured';

    // 各类异常
    case 'target_down':
    case 'scrape_failed':
    case 'query_failed':
      return 'error';

    // 默认（monitorStatus 为 null/undefined 时）
    default:
      return 'normal';
  }
}

/** 四种状态的展示配置 */
export const DISPLAY_STATUS_CONFIG: Record<MonitorDisplayStatus, { label: string; color: string }> = {
  stopped:        { label: '服务已停止',  color: '#d9d9d9' },
  normal:         { label: '监控正常',    color: '#52c41a' },
  not_configured: { label: '未接入监控',  color: '#d9d9d9' },
  error:          { label: '监控异常',    color: '#ff4d4f' },
};
