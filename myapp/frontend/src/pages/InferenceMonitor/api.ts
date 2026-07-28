import axios from '../../api';
import type {
  ApiResponse,
  ServiceItem,
  SummaryResult,
  TimeseriesResult,
  TimeRange,
} from './types';

const BASE = '/inference_monitor/api';

/** 请求可选参数 */
export interface FetchOptions {
  /** 手动刷新时绕过服务端缓存 */
  force?: boolean;
}

export async function fetchServices(params: {
  project_id?: number;
  engine?: string;
  status?: string;
}): Promise<ServiceItem[]> {
  const res = await axios.get<ApiResponse<ServiceItem[]>>(`${BASE}/services`, { params });
  return res.data.result || [];
}

export async function fetchSummary(
  serviceId: number,
  options?: FetchOptions,
): Promise<SummaryResult> {
  const params: Record<string, string | number> = { service_id: serviceId };
  if (options?.force) {
    params.force_refresh = 1;
  }
  const res = await axios.get<ApiResponse<SummaryResult>>(`${BASE}/summary`, { params });
  return res.data.result;
}

export async function fetchTimeseries(
  serviceId: number,
  metrics: string[],
  range: TimeRange,
  options?: FetchOptions,
): Promise<TimeseriesResult> {
  const params: Record<string, string | number> = {
    service_id: serviceId,
    metrics: metrics.join(','),
    range,
  };
  if (options?.force) {
    params.force_refresh = 1;
  }
  const res = await axios.get<ApiResponse<TimeseriesResult>>(`${BASE}/timeseries`, { params });
  return res.data.result;
}

export async function fetchTargetStatus(serviceId: number): Promise<any> {
  const res = await axios.get<ApiResponse<any>>(`${BASE}/target_status`, {
    params: { service_id: serviceId },
  });
  return res.data.result;
}
