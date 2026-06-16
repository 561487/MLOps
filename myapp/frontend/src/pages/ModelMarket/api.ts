
// 模型市场 - API 服务层 (使用项目已有的 axios 实例，自动携带认证 cookie)

import {
  IModelMarketListParams,
  IModelMarketListResponse,
  IModelCategory,
  IInferRequest,
  IInferResponse,
  INotebookConfig,
  IDeployConfig,
  IFinetuneConfig,
  ITaskInfo,
  INotebookResponse,
  IPipelineResponse,
  IDeployResponse,
} from './types';
import axios from '../../api/index';

// Per-request timeout (seconds) — much shorter than the global 600s default
const REQUEST_TIMEOUT = 30000; // 30s

function buildQuery(params: Record<string, any>): string {
  const searchParams = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== '') {
      searchParams.append(key, String(value));
    }
  });
  const qs = searchParams.toString();
  return qs ? `?${qs}` : '';
}

// ===== 模型列表 =====

export async function getModelList(
  params?: IModelMarketListParams
): Promise<IModelMarketListResponse> {
  const query = buildQuery(params || {});
  const res = await (axios.get(`/model_market/api/models${query}`, { timeout: REQUEST_TIMEOUT }) as any);
  // Backend: { code:0, message:"success", data:{ items:[...], total:N } }
  const body = res.data || res;
  if (body && body.data && body.data.items) {
    return { data: body.data.items, total: body.data.total, page: params?.page || 1, page_size: params?.page_size || 12 } as any;
  }
  return body;
}

export async function getModelCategories(): Promise<IModelCategory[]> {
  const res = await (axios.get('/model_market/api/models', { timeout: REQUEST_TIMEOUT }) as any);
  return res.data || res;
}

export async function getModelDetail(id: number | string): Promise<any> {
  const res = await (axios.get(`/model_market/api/models/${id}`, { timeout: REQUEST_TIMEOUT }) as any);
  const body = res.data || res;
  if (body && body.data) return body.data;
  return body;
}

// ===== 在线体验 =====

export async function inferModel(
  id: number | string,
  data: IInferRequest
): Promise<IInferResponse> {
  if (data.files && data.files.length > 0) {
    const formData = new FormData();
    Object.entries(data.inputs).forEach(([key, value]) => {
      formData.append(key, typeof value === 'string' ? value : JSON.stringify(value));
    });
    data.files.forEach((file) => {
      formData.append('files', file);
    });

    const res = await (axios.post(`/model_market/api/models/${id}/experience`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 60000, // longer timeout for file upload
    }) as any);
    return res.data || res;
  }

  const res = await (axios.post(`/model_market/api/models/${id}/experience`, data, { timeout: REQUEST_TIMEOUT }) as any);
  return res.data || res;
}

// ===== 一键开发 =====

export async function createNotebook(
  id: number | string,
  config: INotebookConfig
): Promise<INotebookResponse> {
  const res = await (axios.post(`/model_market/api/models/${id}/develop`, config, { timeout: REQUEST_TIMEOUT }) as any);
  return res.data || res;
}

export async function getNotebookStatus(notebookId: number | string): Promise<any> {
  const res = await (axios.get(`/model_market/api/notebooks/${notebookId}/status`, { timeout: 15000 }) as any);
  return res.data || res;
}

// ===== 一键微调 =====

export async function createFinetune(
  id: number | string,
  config: IFinetuneConfig
): Promise<IPipelineResponse> {
  const res = await (axios.post(`/model_market/api/models/${id}/finetune`, config, { timeout: REQUEST_TIMEOUT }) as any);
  return res.data || res;
}

// ===== 一键部署 =====

export async function deployModel(
  id: number | string,
  config: IDeployConfig
): Promise<IDeployResponse> {
  const res = await (axios.post(`/model_market/api/models/${id}/deploy`, config, { timeout: REQUEST_TIMEOUT }) as any);
  return res.data || res;
}

export async function getModelServices(
  modelId: number | string,
  params?: { active_only?: boolean; model_version?: string; service_name?: string }
): Promise<any> {
  const query = buildQuery(params || {});
  const res = await (axios.get(`/model_market/api/models/${modelId}/services${query}`, { timeout: REQUEST_TIMEOUT }) as any);
  const body = res.data || res;
  if (body && body.data) return body.data;
  return body;
}

export async function unloadService(marketServiceId: number | string): Promise<any> {
  const res = await (axios.post(`/model_market/api/services/${marketServiceId}/unload`, {}, { timeout: REQUEST_TIMEOUT }) as any);
  return res.data || res;
}

// ===== 任务状态 =====

export async function getTaskStatus(taskId: string): Promise<ITaskInfo> {
  const res = await (axios.get(`/model_market/api/actions/${taskId}`, { timeout: REQUEST_TIMEOUT }) as any);
  return res.data || res;
}

export async function getTaskLogs(taskId: string): Promise<{ logs: string[] }> {
  const res = await (axios.get(`/model_market/api/actions/${taskId}/logs`, { timeout: REQUEST_TIMEOUT }) as any);
  return res.data || res;
}
