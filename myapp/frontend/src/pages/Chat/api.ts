/**
 * Chat API 服务层
 * ==================
 * 封装所有 /api/v2/chat/* 接口调用
 */

import axios from '../../api/index';
import type {
  IAgentItem,
  IAgentDetail,
  ISession,
  IChatRequest,
  ICredentialsUpdate,
} from './types';

const BASE = '/api/v2/chat';

// ===== 智能体 =====

/** 获取智能体列表 */
export async function getAgents(category?: string): Promise<IAgentItem[]> {
  const params: any = {};
  if (category) params.category = category;
  const res = await axios.get(`${BASE}/agents`, { params });
  return res.data?.result || [];
}

/** 获取智能体详情（含凭证） */
export async function getAgentDetail(name: string): Promise<IAgentDetail | null> {
  const res = await axios.get(`${BASE}/agents/${name}`);
  return res.data?.result || null;
}

/** 更新智能体配置 */
export async function updateAgentConfig(
  name: string,
  data: ICredentialsUpdate
): Promise<boolean> {
  const res = await axios.put(`${BASE}/agents/${name}/config`, data);
  return res.data?.status === 0;
}

// ===== 会话 =====

/** 创建新会话 */
export async function createSession(
  agentName: string
): Promise<ISession | null> {
  const res = await axios.post(`${BASE}/sessions`, { agent_name: agentName });
  return res.data?.result || null;
}

/** 获取会话列表 */
export async function getSessions(agentName: string): Promise<ISession[]> {
  const res = await axios.get(`${BASE}/sessions`, {
    params: { agent: agentName },
  });
  return res.data?.result || [];
}

/** 删除会话 */
export async function deleteSession(sessionId: string, agentName: string): Promise<boolean> {
  const res = await axios.delete(`${BASE}/sessions/${sessionId}`, {
    params: { agent: agentName },
  });
  return res.data?.status === 0;
}

/** 更新会话标题 */
export async function updateSessionTitle(
  sessionId: string,
  agentName: string,
  title: string,
): Promise<boolean> {
  const res = await axios.patch(`${BASE}/sessions/${sessionId}`, { title }, {
    params: { agent: agentName },
  });
  return res.data?.status === 0;
}

// ===== 聊天历史 =====

/** 获取会话历史消息 */
export async function getHistory(
  sessionId: string,
  agentName: string,
  before?: number,
  limit = 20
): Promise<Array<{ role: string; content: string }>> {
  const params: any = { limit, agent: agentName };
  if (before) params.before = before;
  const res = await axios.get(`${BASE}/history/${sessionId}`, { params });
  return res.data?.result?.messages || [];
}

// ===== SSE 流式对话（使用 fetch，Axios 不支持流式） =====

/**
 * SSE 流式对话
 *
 * 使用原生 fetch + ReadableStream 实现 SSE 消费。
 * Axios 不支持流式响应，所以这里独立实现。
 *
 * @param agentName  智能体名称
 * @param body       请求体 { session_id, search_text, stream: true }
 * @param onChunk    每次收到增量文本时回调
 * @param onDone     流结束时回调（携带完整文本）
 * @param onError    出错时回调
 * @returns          用于取消请求的 AbortController
 */
export function chatSSE(
  agentName: string,
  body: IChatRequest,
  onChunk: (text: string) => void,
  onDone: (fullText: string) => void,
  onError: (err: Error) => void
): AbortController {
  const controller = new AbortController();
  const baseUrl = (axios.defaults as any).baseURL || '';

  fetch(`${baseUrl}${BASE}/chat/${agentName}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-Requested-With': 'XMLHttpRequest',
    },
    body: JSON.stringify({ ...body, stream: true }),
    signal: controller.signal,
    credentials: 'include',
    })
      .then(async (response) => {
        if (!response.ok) {
          let errMsg = `HTTP ${response.status}`;
          try {
            const body = await response.json();
            if (body.error) errMsg = body.error;
          } catch {}
          throw new Error(errMsg);
        }
        const reader = response.body?.getReader();
        if (!reader) {
          throw new Error('No response body');
        }

        const decoder = new TextDecoder();
        let fullText = '';
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });

          // 解析 SSE 格式: "TQJXQKT0POF6P4D:..." 分隔符
          // （与现有后端 chatgpt() 的返回格式兼容）
          const parts = buffer.split('TQJXQKT0POF6P4D:');
          buffer = parts.pop() || '';

          for (const part of parts) {
            if (!part.trim()) continue;
            try {
              const chunk = JSON.parse(part);
              if (chunk.result?.[0]?.text) {
                const text = chunk.result[0].text;
                // 只推送增量部分
                if (text.length > fullText.length) {
                  const delta = text.slice(fullText.length);
                  fullText = text;
                  onChunk(delta);
                }
              }
            } catch {
              // 跳过无法解析的片段
            }
          }
        }

        onDone(fullText);
      })
      .catch((err) => {
        if (err.name !== 'AbortError') {
          onError(err);
        }
      });

  return controller;
}
