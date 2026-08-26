import { useCallback, useEffect, useRef, useState } from 'react';
import Cookies from 'js-cookie';

export interface AssistantMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
}

export interface AssistantSession {
  session_id: string;
  title: string;
  created_on: string;
  changed_on: string;
}

const BASE_URL = '/api/v2/assistant';
const CHAT_URL = `${BASE_URL}/chat`;

// 后端接口未就绪时置为 true，使用本地 mock 流式回复；后端就绪后改为 false
const USE_MOCK = false;

// 输入框字符上限
export const INPUT_MAX_CHARS = 2000;

let uid = 0;
const genId = () => `msg_${Date.now()}_${uid++}`;

interface SessionInit {
  sid: string;
  messages: AssistantMessage[];
  sessions: AssistantSession[];
}

const WELCOME = '你好！我是平台助手，可以帮你解答平台操作问题（模型微调、数据集、流水线、镜像等）。你可以点击下方快捷问题，或直接输入你的问题。';

const MOCK_REPLY = `这是一个 **mock 回复**（后端接口尚未接入）。

后端就绪后，这里将返回 LLM 的流式回答。你可以这样使用平台：

1. **模型微调**：进入流水线页面，选择 llama_factory 任务模板，配置模型、数据集和 LoRA 参数后提交。
2. **数据集**：支持 ShareGPT / 标准格式，需配套 dataset_info.json。
3. **镜像规范**：Tag 格式为 <framework-version>-py<python>-cu<cuda>-r<revision>。

> 提示：回答将以流式逐字显示。`;

// ===== 用户名（仅用于本地 fallback session_id，不再绑用户存 localStorage） =====
const getUsername = (): string => {
  try {
    return Cookies.get('myapp_username') || 'kubeflow';
  } catch {
    return 'kubeflow';
  }
};

/**
 * 从后端拉取当前用户的会话列表，返回最近一段会话的消息。
 * - 有历史会话 → 取最近一段，加载其消息
 * - 没有历史会话 → POST /session/new 创建新会话，返回欢迎语
 * - 后端不可用 → 降级到本地 fallback session_id
 */
const initSessionFromBackend = async (): Promise<SessionInit> => {
  // 1. 拉用户会话列表
  try {
    const resp = await fetch(`${BASE_URL}/sessions`, { credentials: 'include' });
    if (resp.ok) {
      const data = await resp.json();
      const sessions: AssistantSession[] = (data?.data || []).map((s: any) => ({
        session_id: s.session_id,
        title: s.title || '',
        created_on: s.created_on || '',
        changed_on: s.changed_on || '',
      }));
      if (sessions.length > 0) {
        const sid = sessions[0].session_id;
        // 加载该会话的消息
        const histResp = await fetch(`${BASE_URL}/history/${sid}`, { credentials: 'include' });
        let messages: AssistantMessage[] = [];
        if (histResp.ok) {
          const histData = await histResp.json();
          const hist: any[] = histData?.data || [];
          messages = hist.map((m) => ({
            id: genId(),
            role: m.role as 'user' | 'assistant',
            content: m.content || '',
          }));
        }
        if (messages.length === 0) {
          messages = [{ id: genId(), role: 'assistant', content: WELCOME }];
        }
        return { sid, messages, sessions };
      }
    }
  } catch {
    // 网络失败，继续走创建新会话
  }

  // 2. 没有历史会话 → 创建新会话
  try {
    const resp = await fetch(`${BASE_URL}/session/new`, {
      method: 'POST',
      credentials: 'include',
    });
    if (resp.ok) {
      const data = await resp.json();
      return {
        sid: data.session_id,
        messages: [{ id: genId(), role: 'assistant', content: WELCOME }],
        sessions: [],
      };
    }
  } catch {
    // 后端不可用，降级
  }

  // 3. 降级：本地生成 session_id
  const sid = `assistant_${getUsername()}_${Date.now()}`;
  return {
    sid,
    messages: [{ id: genId(), role: 'assistant', content: WELCOME }],
    sessions: [],
  };
};

/** 调后端创建新会话，返回 session_id（失败时本地生成） */
const createNewSession = async (): Promise<string> => {
  try {
    const resp = await fetch(`${BASE_URL}/session/new`, {
      method: 'POST',
      credentials: 'include',
    });
    if (resp.ok) {
      const data = await resp.json();
      return data.session_id;
    }
  } catch {}
  return `assistant_${getUsername()}_${Date.now()}`;
};

/** 重新拉取会话列表 */
const fetchSessions = async (): Promise<AssistantSession[]> => {
  try {
    const resp = await fetch(`${BASE_URL}/sessions`, { credentials: 'include' });
    if (resp.ok) {
      const data = await resp.json();
      return (data?.data || []).map((s: any) => ({
        session_id: s.session_id,
        title: s.title || '',
        created_on: s.created_on || '',
        changed_on: s.changed_on || '',
      }));
    }
  } catch {}
  return [];
};

export function useAssistant() {
  // 初始用欢迎语占位，useEffect 里异步从后端加载真实历史
  const [messages, setMessages] = useState<AssistantMessage[]>(() => [
    { id: genId(), role: 'assistant', content: WELCOME },
  ]);
  const [sessions, setSessions] = useState<AssistantSession[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [initialized, setInitialized] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const mockTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const sessionIdRef = useRef<string>('');

  // 首次挂载：从后端加载历史会话
  useEffect(() => {
    let cancelled = false;
    (async () => {
      const { sid, messages: hist, sessions: ss } = await initSessionFromBackend();
      if (cancelled) return;
      sessionIdRef.current = sid;
      setCurrentSessionId(sid);
      setMessages(hist);
      setSessions(ss);
      setInitialized(true);
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // 向最后一条 assistant 消息追加内容
  const appendToLast = useCallback((text: string) => {
    setMessages((prev) => {
      const next = [...prev];
      for (let i = next.length - 1; i >= 0; i--) {
        if (next[i].role === 'assistant') {
          next[i] = { ...next[i], content: next[i].content + text };
          break;
        }
      }
      return next;
    });
  }, []);

  const clearMockTimer = () => {
    if (mockTimerRef.current) {
      clearInterval(mockTimerRef.current);
      mockTimerRef.current = null;
    }
  };

  // mock 流式回复：逐字吐出（用 [\s\S] 替代 s flag，兼容 ES2017）
  const runMock = useCallback((onDone: () => void) => {
    const chunks = MOCK_REPLY.match(/[\s\S]{1,4}/g) || [];
    let idx = 0;
    clearMockTimer();
    mockTimerRef.current = setInterval(() => {
      if (idx < chunks.length) {
        appendToLast(chunks[idx]);
        idx++;
      } else {
        clearMockTimer();
        onDone();
      }
    }, 30);
  }, [appendToLast]);

  // 真实 SSE 请求
  const runSSE = useCallback(async (text: string, onDone: () => void, onError: (msg: string) => void) => {
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      const resp = await fetch(CHAT_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, session_id: sessionIdRef.current }),
        signal: controller.signal,
        credentials: 'include',
      });
      if (!resp.ok || !resp.body) {
        throw new Error(`HTTP ${resp.status}`);
      }
      const reader = resp.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed.startsWith('data:')) continue;
          const payload = trimmed.slice(5).trim();
          if (!payload) continue;
          try {
            const data = JSON.parse(payload);
            if (data.done) {
              onDone();
              return;
            }
            if (typeof data.text === 'string') {
              appendToLast(data.text);
            }
          } catch {
            appendToLast(payload);
          }
        }
      }
      onDone();
    } catch (e: any) {
      if (e?.name === 'AbortError') {
        onDone();
        return;
      }
      onError(e?.message || '请求失败');
    }
  }, [appendToLast]);

  const send = useCallback((text: string) => {
    const content = text.trim();
    if (!content || loading) return;
    setMessages((prev) => [
      ...prev,
      { id: genId(), role: 'user', content },
      { id: genId(), role: 'assistant', content: '' },
    ]);
    setLoading(true);

    const onDone = () => setLoading(false);
    const onError = (msg: string) => {
      appendToLast(`\n\n> 请求失败：${msg}`);
      setLoading(false);
    };

    if (USE_MOCK) {
      runMock(onDone);
    } else {
      runSSE(content, onDone, onError);
    }
  }, [loading, runMock, runSSE, appendToLast]);

  // 停止当前回复
  const stop = useCallback(() => {
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
    }
    clearMockTimer();
    setLoading(false);
  }, []);

  // 重新生成最后一条回复
  const regenerate = useCallback(() => {
    if (loading) return;
    let lastUserIdx = -1;
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === 'user') {
        lastUserIdx = i;
        break;
      }
    }
    if (lastUserIdx < 0) return;
    const userText = messages[lastUserIdx].content;
    setMessages((prev) => [
      ...prev.slice(0, lastUserIdx + 1),
      { id: genId(), role: 'assistant', content: '' },
    ]);
    setLoading(true);
    const onDone = () => setLoading(false);
    const onError = (msg: string) => {
      appendToLast(`\n\n> 请求失败：${msg}`);
      setLoading(false);
    };
    if (USE_MOCK) {
      runMock(onDone);
    } else {
      runSSE(userText, onDone, onError);
    }
  }, [loading, messages, runMock, runSSE, appendToLast]);

  // 清空对话：调后端 DELETE，再创建新会话
  const clearHistory = useCallback(async () => {
    const oldSid = sessionIdRef.current;
    try {
      await fetch(`${BASE_URL}/history/${oldSid}`, {
        method: 'DELETE',
        credentials: 'include',
      });
    } catch {}
    const newSid = await createNewSession();
    sessionIdRef.current = newSid;
    setCurrentSessionId(newSid);
    setMessages([{ id: genId(), role: 'assistant', content: WELCOME }]);
    setSessions(await fetchSessions());
  }, []);

  // 切换到指定会话（用于侧边栏点击历史会话）
  const switchSession = useCallback(async (sid: string) => {
    if (sid === sessionIdRef.current) return;
    sessionIdRef.current = sid;
    setCurrentSessionId(sid);
    try {
      const resp = await fetch(`${BASE_URL}/history/${sid}`, { credentials: 'include' });
      if (resp.ok) {
        const data = await resp.json();
        const hist: any[] = data?.data || [];
        if (hist.length > 0) {
          setMessages(hist.map((m) => ({
            id: genId(),
            role: m.role as 'user' | 'assistant',
            content: m.content || '',
          })));
          return;
        }
      }
    } catch {}
    setMessages([{ id: genId(), role: 'assistant', content: WELCOME }]);
  }, []);

  // 新建会话（不删旧的，用于侧边栏"新建对话"按钮）
  const newSession = useCallback(async () => {
    const newSid = await createNewSession();
    sessionIdRef.current = newSid;
    setCurrentSessionId(newSid);
    setMessages([{ id: genId(), role: 'assistant', content: WELCOME }]);
    setSessions(await fetchSessions());
  }, []);

  return {
    messages,
    sessions,
    currentSessionId,
    loading,
    initialized,
    send,
    stop,
    clearHistory,
    regenerate,
    switchSession,
    newSession,
  };
}
