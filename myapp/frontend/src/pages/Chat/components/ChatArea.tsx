/**
 * ChatArea - 聊天主区域（v2.0）
 * ===========================================================
 * 功能：
 *   - 会话列表（新建/切换/删除）
 *   - 消息列表（Markdown 渲染 + <think> 折叠块）
 *   - SSE 流式接收
 *   - 会话上下文自动管理
 */

import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  Input,
  Button,
  Avatar,
  Typography,
  Spin,
  Empty,
  message as antdMsg,
  Popconfirm,
} from 'antd';
import {
  SendOutlined,
  RobotOutlined,
  UserOutlined,
  ClearOutlined,
  SettingOutlined,
  LoadingOutlined,
  PlusOutlined,
  DeleteOutlined,
  HistoryOutlined,
} from '@ant-design/icons';
import { chatSSE, createSession, getHistory, getSessions, deleteSession, updateSessionTitle } from '../api';
import type { IAgentItem, IMessage, ISession } from '../types';

const { Text } = Typography;
const { TextArea } = Input;

/* ================================================================
   工具函数
   ================================================================ */

/** 去除 HTML 标签，保留纯文本 */
function stripHtml(html: string): string {
  if (!html) return '';
  const tmp = document.createElement('div');
  tmp.innerHTML = html;
  return tmp.textContent || tmp.innerText || '';
}

/* ================================================================
   Props
   ================================================================ */

interface ChatAreaProps {
  agent: IAgentItem | null;
  onOpenSettings?: () => void;
}

/* ================================================================
   Markdown + Think Block 渲染（纯函数，可提取到独立文件）
   ================================================================ */

function simpleMarkdown(text: string): string {
  if (!text) return '';

  const parts: string[] = [];
  let lastIndex = 0;
  let inThink = false;
  let thinkBuffer = '';

  for (const match of Array.from(text.matchAll(/<think>|<\/think>/gi))) {
    const token = match[0];
    const start = match.index ?? 0;
    const segment = text.slice(lastIndex, start);

    if (inThink) {
      thinkBuffer += segment;
    } else {
      parts.push(formatTextSegment(segment));
    }

    if (token.toLowerCase() === '<think>') {
      inThink = true;
      thinkBuffer = '';
    } else if (token.toLowerCase() === '</think>') {
      inThink = false;
      const innerHtml = escapeAndInlineMarkdown(thinkBuffer);
      parts.push(renderThinkBlock(innerHtml, false));
      thinkBuffer = '';
    }

    lastIndex = (match.index ?? 0) + token.length;
  }

  const tail = text.slice(lastIndex);
  if (inThink) {
    thinkBuffer += tail;
    parts.push(renderThinkBlock(escapeAndInlineMarkdown(thinkBuffer), true));
  } else {
    parts.push(formatTextSegment(tail));
  }

  return parts.join('');
}

function renderThinkBlock(content: string, streaming: boolean): string {
  const statusText = streaming ? '思考中' : '思考过程';
  const dotHtml = streaming ? '<span class="think-dot"></span>' : '';
  const openAttr = streaming ? ' open' : '';

  return (
    '<details class="think-block' +
    (streaming ? ' think-block-streaming' : ' think-block-complete') +
    '"' +
    openAttr +
    '>' +
    '<summary class="think-summary">' +
    dotHtml +
    '<span>' +
    statusText +
    '</span>' +
    '</summary>' +
    '<div class="think-body">' +
    (content || '<span class="think-body-empty">暂无内容</span>') +
    '</div>' +
    '</details>'
  );
}

function formatTextSegment(text: string): string {
  if (!text) return '';

  return text
    .split('\n')
    .map((line) => {
      const trimmed = line.trim();

      if (/^#{1,6}\s+/.test(trimmed)) {
        const level = trimmed.match(/^#{1,6}/)?.[0].length ?? 1;
        const content = trimmed.replace(/^#{1,6}\s+/, '');
        return '<h' + level + '>' + escapeHtml(content) + '</h' + level + '>';
      }

      if (/^([-*_])(?:\s*){2,}\s*$/.test(trimmed) || /^-{3,}\s*$/.test(trimmed)) {
        return '<div class="md-divider"></div>';
      }

      return escapeAndInlineMarkdown(line);
    })
    .join('<br/>');
}

function escapeHtml(txt: string): string {
  if (!txt) return '';
  return txt.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function escapeAndInlineMarkdown(txt: string): string {
  if (!txt) return '';
  return escapeHtml(txt)
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\n/g, '<br/>');
}

/* ================================================================
   Component
   ================================================================ */

const ChatArea: React.FC<ChatAreaProps> = ({ agent, onOpenSettings }) => {
  /* ---------- state ---------- */
  const [messages, setMessages] = useState<IMessage[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [streaming, setStreaming] = useState(false);
  const [streamingText, setStreamingText] = useState('');
  const [session, setSession] = useState<ISession | null>(null);
  const [sessions, setSessions] = useState<ISession[]>([]);
  const [loadingSessions, setLoadingSessions] = useState(false);
  const [loadingHistory, setLoadingHistory] = useState(false);

  /* ---------- refs ---------- */
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const messagesCacheRef = useRef<Record<string, IMessage[]>>({});
  const sessionsCacheRef = useRef<Record<string, ISession[]>>({});
  const sessionCreatedRef = useRef(false);

  /* ---------- 切换AI助手：重置 + 加载会话 ---------- */
  useEffect(() => {
    setMessages([]);
    setStreamingText('');
    setSession(null);
    setInputValue('');
    setSessions([]);
    sessionCreatedRef.current = false;

    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
    }

    if (agent?.name) {
      loadSessions(agent.name);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [agent?.name]);

  /* ---------- 自动滚动 ---------- */
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streamingText]);

  /* ---------- 加载会话列表 ---------- */
  const loadSessions = useCallback(async (agentName: string) => {
    setLoadingSessions(true);
    try {
      const cached = sessionsCacheRef.current[agentName];
      if (cached && cached.length > 0) {
        setSessions(cached);
        setLoadingSessions(false);
        return;
      }

      const list = await getSessions(agentName);
      setSessions(list);
      sessionsCacheRef.current[agentName] = list;
    } catch (err) {
      console.error('加载会话列表失败', err);
    } finally {
      setLoadingSessions(false);
    }
  }, []);

  /* ---------- 加载会话历史 ---------- */
  const loadSessionMessages = useCallback(
    async (sessionId: string, agentName: string) => {
      setLoadingHistory(true);
      try {
        const history = await getHistory(sessionId, agentName);
        const mapped = history.map((item) => ({
          role: (item.role === 'assistant' ? 'assistant' : 'user') as 'user' | 'assistant',
          content: item.content,
        }));
        setMessages(mapped);
      } catch (err) {
        console.error('加载会话历史失败', err);
      } finally {
        setLoadingHistory(false);
      }
    },
    [],
  );

  /* ---------- 新建会话 ---------- */
  const handleNewSession = useCallback(async () => {
    if (!agent) return;
    try {
      const newSession = await createSession(agent.name);
      if (newSession) {
        // 保存当前会话消息到缓存
        if (session?.id) {
          messagesCacheRef.current[session.id] = messages;
        }
        setSession(newSession);
        setMessages([]);
        setSessions((prev) => {
          const updated = [newSession, ...prev];
          if (agent) sessionsCacheRef.current[agent.name] = updated;
          return updated;
        });
        setStreamingText('');
        setInputValue('');
        messagesCacheRef.current[newSession.id] = [];
      }
    } catch (err) {
      antdMsg.error('创建会话失败');
    }
  }, [agent, session?.id, messages]);

  /* ---------- 切换会话（含本地缓存） ---------- */
  const handleSelectSession = useCallback(
    async (target: ISession) => {
      if (target.id === session?.id) return;

      // 保存当前会话消息到本地缓存
      if (session?.id) {
        messagesCacheRef.current[session.id] = messages;
      }

      // 切换会话
      setSession(target);
      setStreamingText('');

      // 优先使用本地缓存
      const cached = messagesCacheRef.current[target.id];
      if (cached && cached.length > 0) {
        setMessages(cached);
        return;
      }

      // 缓存未命中，从后端加载
      await loadSessionMessages(target.id, agent?.name || '');
    },
    [session?.id, agent?.name, messages, loadSessionMessages],
  );

  /* ---------- 删除会话 ---------- */
  const handleDeleteSession = useCallback(
    async (targetId: string, e: React.MouseEvent) => {
      e.stopPropagation();
      if (!agent) return;
      try {
        const ok = await deleteSession(targetId, agent.name);
        if (ok) {
          setSessions((prev) => {
            const updated = prev.filter((item) => item.id !== targetId);
            if (agent) sessionsCacheRef.current[agent.name] = updated;
            return updated;
          });
          delete messagesCacheRef.current[targetId];
          if (session?.id === targetId) {
            setSession(null);
            setMessages([]);
          }
        }
      } catch (err) {
        antdMsg.error('删除会话失败');
      }
    },
    [agent, session?.id],
  );

  /* ---------- 发送消息 ---------- */
  const handleSend = useCallback(async () => {
    const text = inputValue.trim();
    if (!text || !agent) return;

    setInputValue('');

    const userMsg: IMessage = { role: 'user', content: text, timestamp: Date.now() };
    setMessages((prev) => {
      const updated = [...prev, userMsg];
      // 实时缓存到 messagesCacheRef
      if (session?.id) {
        messagesCacheRef.current[session.id] = updated;
      }
      return updated;
    });

    /* 确保会话存在 */
    let currentSession = session;
    if (!currentSession) {
      try {
        const newSession = await createSession(agent.name);
        if (newSession) {
          currentSession = newSession;
          setSession(newSession);
          setSessions((prev) => [newSession, ...prev]);
          messagesCacheRef.current[newSession.id] = [userMsg];
          sessionCreatedRef.current = true;
        }
      } catch (err) {
        antdMsg.error('创建会话失败');
        return;
      }
    }

    /* 更新会话标题（取首条用户消息前 20 字，默认标题为「新对话」时自动更新） */
    const shouldUpdateTitle = currentSession && currentSession.title === '新对话';
    if (shouldUpdateTitle) {
      const title = text.length > 20 ? text.slice(0, 20) + '...' : text;
      setSessions((prev) => {
        const updated = prev.map((s) => (s.id === currentSession!.id ? { ...s, title } : s));
        if (agent) sessionsCacheRef.current[agent.name] = updated;
        return updated;
      });
      // 同步到后端的 Redis 缓存（确保切换回来不会丢标题）
      updateSessionTitle(currentSession!.id, agent.name, title).catch(() => {});
      // 同步到缓存中的 session 对象（用于后续 isNew 判断）
      if (currentSession) {
        currentSession = { ...currentSession, title };
        setSession(currentSession);
      }
    }

    /* 开始流式接收 */
    setStreaming(true);
    setStreamingText('');

    abortRef.current = chatSSE(
      agent.name,
      {
        session_id: currentSession?.id || 'default',
        search_text: text,
        stream: true,
      },
      (chunk) => {
        setStreamingText((prev) => prev + chunk);
      },
      (fullText) => {
        setStreaming(false);
        setStreamingText('');
        const sid = currentSession?.id || 'default';
        if (fullText) {
          const assistantMsg: IMessage = {
            role: 'assistant',
            content: fullText,
            timestamp: Date.now(),
          };
          setMessages((prev) => {
            const updated = [...prev, assistantMsg];
            if (currentSession?.id) {
              messagesCacheRef.current[currentSession.id] = updated;
            }
            return updated;
          });
        }
      },
      (err) => {
        setStreaming(false);
        antdMsg.error('对话出错: ' + err.message);
      },
    );
  }, [inputValue, agent, session]);

  /* ---------- 键盘事件 ---------- */
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend],
  );

  /* ---------- 清空当前会话 ---------- */
  const handleClear = useCallback(() => {
    setMessages([]);
    setStreamingText('');
    if (session?.id) {
      messagesCacheRef.current[session.id] = [];
    }
  }, [session?.id]);

  /* ---------- 停止生成 ---------- */
  const handleStop = useCallback(() => {
    abortRef.current?.abort();
    setStreaming(false);
    if (streamingText) {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant' as const, content: streamingText, timestamp: Date.now() },
      ]);
      setStreamingText('');
    }
  }, [streamingText]);

  /* ================================================================
     Render
     ================================================================ */

  /* 欢迎界面 */
  if (!agent) {
    return (
      <div className="chat-area-empty">
        <div className="chat-area-empty-content">
          <RobotOutlined style={{ fontSize: 64, color: '#d9d9d9' }} />
          <h3>选择一个AI助手开始对话</h3>
          <Text type="secondary">
            从左侧边栏选择机器人或AI助手，开启 AI 对话之旅
          </Text>
        </div>
      </div>
    );
  }

  return (
    <div className="chat-area">
      {/* ======== 顶部栏 ======== */}
      <div className="chat-area-header">
        <div className="chat-area-header-left">
          {agent.icon ? (
            <Avatar
              src={`data:image/svg+xml;utf8,${encodeURIComponent(agent.icon)}`}
              size={36}
            />
          ) : (
            <Avatar
              icon={<RobotOutlined />}
              style={{ backgroundColor: '#1677ff' }}
              size={36}
            />
          )}
          <span className="chat-area-header-name">
            {agent.label || agent.name}
          </span>
        </div>
        <div className="chat-area-header-right">
          <Button
            type="text"
            size="small"
            icon={<PlusOutlined />}
            onClick={handleNewSession}
          >
            新建会话
          </Button>
          <Button
            type="text"
            size="small"
            icon={<ClearOutlined />}
            onClick={handleClear}
          >
            清空
          </Button>
          {onOpenSettings && (
            <Button
              type="text"
              size="small"
              icon={<SettingOutlined />}
              onClick={onOpenSettings}
            >
              设置
            </Button>
          )}
        </div>
      </div>

      {/* ======== 会话横向滚动列表 ======== */}
      {sessions.length > 0 && (
        <div className="chat-session-bar">
          <HistoryOutlined className="chat-session-bar-icon" />
          <div className="chat-session-list">
            {sessions.map((item) => (
              <div
                key={item.id}
                className={
                  'chat-session-chip' +
                  (session?.id === item.id ? ' active' : '')
                }
                onClick={() => handleSelectSession(item)}
              >
                <span className="chat-session-chip-title">
                  {item.title || '新会话'}
                </span>
                <Popconfirm
                  title="确定删除该会话？"
                  onConfirm={(e) =>
                    handleDeleteSession(item.id, e as unknown as React.MouseEvent)
                  }
                  okText="删除"
                  cancelText="取消"
                  placement="bottom"
                >
                  <DeleteOutlined
                    className="chat-session-chip-delete"
                    onClick={(e) => e.stopPropagation()}
                  />
                </Popconfirm>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ======== 消息列表 ======== */}
      <div className="chat-area-messages">
        {loadingHistory ? (
          <div className="chat-area-loading">
            <Spin size="small" /> 加载历史消息...
          </div>
        ) : messages.length === 0 && !streamingText ? (
          <div className="chat-area-welcome">
            <div className="chat-area-welcome-hello">
              {stripHtml(agent.hello) || '你好！有什么可以帮你的？'}
            </div>
            {agent.tips && agent.tips.length > 0 && (
              <div className="chat-area-welcome-tips">
                <Text type="secondary">试试这些：</Text>
                <div className="chat-area-welcome-tip-list">
                  {agent.tips.map((tip, i) => (
                    <div
                      key={i}
                      className="chat-area-welcome-tip-item"
                      onClick={() => setInputValue(tip)}
                    >
                      {stripHtml(tip)}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        ) : null}

        {messages.map((msg, i) => (
          <div
            key={i}
            className={
              'chat-message ' +
              (msg.role === 'user'
                ? 'chat-message-user'
                : 'chat-message-assistant')
            }
          >
            <div className="chat-message-avatar">
              {msg.role === 'user' ? (
                <Avatar
                  icon={<UserOutlined />}
                  style={{ backgroundColor: '#1677ff' }}
                  size={32}
                />
              ) : agent.icon ? (
                <Avatar
                  src={`data:image/svg+xml;utf8,${encodeURIComponent(agent.icon)}`}
                  size={32}
                />
              ) : (
                <Avatar
                  icon={<RobotOutlined />}
                  style={{ backgroundColor: '#52c41a' }}
                  size={32}
                />
              )}
            </div>
            <div className="chat-message-content">
              <div
                className="chat-message-text"
                dangerouslySetInnerHTML={{
                  __html: simpleMarkdown(msg.content),
                }}
              />
            </div>
          </div>
        ))}

        {/* 流式生成中的临时消息 */}
        {streamingText && (
          <div className="chat-message chat-message-assistant">
            <div className="chat-message-avatar">
              {agent.icon ? (
                <Avatar
                  src={`data:image/svg+xml;utf8,${encodeURIComponent(agent.icon)}`}
                  size={32}
                />
              ) : (
                <Avatar
                  icon={<RobotOutlined />}
                  style={{ backgroundColor: '#52c41a' }}
                  size={32}
                />
              )}
            </div>
            <div className="chat-message-content">
              <div
                className="chat-message-text"
                dangerouslySetInnerHTML={{
                  __html: simpleMarkdown(streamingText),
                }}
              />
              <span className="chat-message-cursor">|</span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* ======== 输入区域 ======== */}
      <div className="chat-area-input">
        <TextArea
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={
            streaming
              ? 'AI 正在生成中...'
              : '输入消息，Enter 发送，Shift+Enter 换行'
          }
          autoSize={{ minRows: 2, maxRows: 5 }}
          disabled={streaming}
        />
        <div className="chat-area-input-actions">
          {streaming ? (
            <Button
              type="primary"
              danger
              icon={<LoadingOutlined />}
              onClick={handleStop}
            >
              停止生成
            </Button>
          ) : (
            <Button
              type="primary"
              icon={<SendOutlined />}
              onClick={handleSend}
              disabled={!inputValue.trim()}
              style={{ minWidth: 80, fontWeight: 600, fontSize: 14 }}
            >
              发送
            </Button>
          )}
        </div>
      </div>
    </div>
  );
};

export default ChatArea;
