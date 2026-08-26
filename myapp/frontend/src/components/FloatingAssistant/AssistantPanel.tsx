import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Button, Input, Popconfirm, Tooltip, message as antdMessage } from 'antd';
import { CloseOutlined, CopyOutlined, DeleteOutlined, PlusOutlined, ReloadOutlined, SendOutlined, UnorderedListOutlined, UserOutlined } from '@ant-design/icons';
import { CopyToClipboard } from 'react-copy-to-clipboard';
import DOMPurify from 'dompurify';
import { marked } from 'marked';
import { AssistantMessage, AssistantSession, INPUT_MAX_CHARS } from './useAssistant';
import tubiaoImg from '../../images/tubiao.png';
import './style.css';

interface IProps {
  messages: AssistantMessage[];
  sessions: AssistantSession[];
  currentSessionId: string;
  loading: boolean;
  initialized: boolean;
  onSend: (text: string) => void;
  onStop: () => void;
  onClose: () => void;
  onClear: () => void | Promise<void>;
  onRegenerate: () => void;
  onSwitchSession: (sid: string) => void | Promise<void>;
  onNewSession: () => void | Promise<void>;
}

const QUICK_QUESTIONS = [
  { label: '怎么微调模型？', message: '怎么微调模型？' },
  { label: '数据集格式？', message: '数据集格式是什么？dataset_info.json 怎么写？' },
  { label: '创建流水线', message: '怎么创建流水线？任务怎么串联？' },
  { label: '镜像规范', message: '镜像构建规范是什么？Tag 怎么命名？' },
];

const renderMarkdown = (content: string): string => {
  try {
    const html = marked.parse(content, { async: false }) as string;
    // 防 XSS：DOMPurify 默认会移除 <script>、onerror=、javascript: 等危险标签/属性
    // 加 ADD_ATTR 让 a 标签的 target=_blank 能用（marked 输出 a 标签时设的）
    return DOMPurify.sanitize(html, {
      ADD_ATTR: ['target'],
    });
  } catch {
    return DOMPurify.sanitize(content);
  }
};

const AssistantPanel: React.FC<IProps> = ({
  messages,
  sessions,
  currentSessionId,
  loading,
  initialized,
  onSend,
  onStop,
  onClose,
  onClear,
  onRegenerate,
  onSwitchSession,
  onNewSession,
}) => {
  const [inputValue, setInputValue] = useState('');
  const [showHistory, setShowHistory] = useState(false);
  const listRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (listRef.current) {
      listRef.current.scrollTop = listRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSend = (text?: string) => {
    const content = (text ?? inputValue).trim();
    if (!content || loading) return;
    if (content.length > INPUT_MAX_CHARS) {
      antdMessage.warning(`单条消息不能超过 ${INPUT_MAX_CHARS} 字`);
      return;
    }
    onSend(content);
    setInputValue('');
  };

  const handleCopy = () => {
    antdMessage.success('已复制');
  };

  const renderedMessages = useMemo(
    () =>
      messages.map((msg) => ({
        ...msg,
        html: msg.role === 'assistant' ? renderMarkdown(msg.content) : '',
      })),
    [messages]
  );

  // Find the last assistant message index — mark it as "thinking" during loading
  const lastAssistantIdx = useMemo(() => {
    for (let i = renderedMessages.length - 1; i >= 0; i--) {
      if (renderedMessages[i].role === 'assistant') return i;
    }
    return -1;
  }, [renderedMessages]);

  // 最后一条 AI 回复的索引（用于显示"重新生成"按钮，loading 时不显示）
  const lastReplyIdx = useMemo(() => {
    if (loading) return -1;
    for (let i = renderedMessages.length - 1; i >= 0; i--) {
      if (renderedMessages[i].role === 'assistant' && renderedMessages[i].content) return i;
    }
    return -1;
  }, [renderedMessages, loading]);

  return (
    <div className="fa-panel">
      {/* 历史会话侧边栏（覆盖在面板上） */}
      {showHistory && (
        <div className="fa-history-overlay">
          <div className="fa-history-header">
            <span>历史对话</span>
            <CloseOutlined className="fa-history-close" onClick={() => setShowHistory(false)} />
          </div>
          <div
            className="fa-history-new"
            onClick={async () => {
              await onNewSession();
              setShowHistory(false);
            }}
          >
            <PlusOutlined />
            <span>新建对话</span>
          </div>
          <div className="fa-history-list">
            {sessions.length === 0 ? (
              <div className="fa-history-empty">暂无历史对话</div>
            ) : (
              sessions.map((s) => (
                <div
                  key={s.session_id}
                  className={`fa-history-item${s.session_id === currentSessionId ? ' active' : ''}`}
                  onClick={async () => {
                    await onSwitchSession(s.session_id);
                    setShowHistory(false);
                  }}
                >
                  <div className="fa-history-item-title">{s.title || '(无标题)'}</div>
                  <div className="fa-history-item-time">{s.changed_on || s.created_on}</div>
                </div>
              ))
            )}
          </div>
        </div>
      )}

      {/* Header */}
      <div className="fa-panel-header">
        <div className="fa-panel-title">
          <img src={tubiaoImg} alt="AI" className="fa-title-icon" />
          平台助手
        </div>
        <div className="fa-panel-header-actions">
          <Tooltip title="历史对话">
            <UnorderedListOutlined
              className="fa-header-btn"
              onClick={() => setShowHistory((v) => !v)}
            />
          </Tooltip>
          <Popconfirm
            title="确认清空当前对话？"
            okText="清空"
            cancelText="取消"
            onConfirm={async () => {
              await onClear();
              antdMessage.success('已清空对话');
            }}
          >
            <Tooltip title="清空对话">
              <DeleteOutlined className="fa-header-btn" />
            </Tooltip>
          </Popconfirm>
          <Tooltip title="收起">
            <CloseOutlined className="fa-panel-close" onClick={onClose} />
          </Tooltip>
        </div>
      </div>

      {/* Messages */}
      <div className="fa-panel-messages" ref={listRef}>
        {!initialized ? (
          <div className="fa-loading-state">正在加载历史对话…</div>
        ) : (
          renderedMessages.map((msg, idx) => (
            <div key={msg.id} className={`fa-msg fa-msg-${msg.role}`}>
              <div className="fa-msg-avatar">
                {msg.role === 'assistant' ? (
                  <img src={tubiaoImg} alt="AI" className="fa-msg-avatar-img" />
                ) : (
                  <UserOutlined />
                )}
              </div>
              <div className="fa-msg-content">
                {msg.role === 'assistant' ? (
                  <div
                    className={`fa-msg-bubble fa-msg-bubble-md${
                      loading && idx === lastAssistantIdx ? ' fa-thinking' : ''
                    }`}
                    dangerouslySetInnerHTML={{
                      __html: msg.html || '<span class="fa-typing"><i></i><i></i><i></i></span>',
                    }}
                  />
                ) : (
                  <div className="fa-msg-bubble">{msg.content}</div>
                )}
                {msg.role === 'assistant' && msg.content && (
                  <div className="fa-msg-actions">
                    <CopyToClipboard text={msg.content} onCopy={handleCopy}>
                      <Tooltip title="复制">
                        <CopyOutlined className="fa-action-btn" />
                      </Tooltip>
                    </CopyToClipboard>
                    {idx === lastReplyIdx && (
                      <Tooltip title="重新生成">
                        <ReloadOutlined className="fa-action-btn" onClick={onRegenerate} />
                      </Tooltip>
                    )}
                  </div>
                )}
              </div>
            </div>
          ))
        )}
      </div>

      {/* Quick actions — underlined chips */}
      <div className="fa-panel-quick">
        {QUICK_QUESTIONS.map((q) => (
          <span key={q.label} className="fa-quick-btn" onClick={() => handleSend(q.message)}>
            {q.label}
          </span>
        ))}
      </div>

      {/* Input */}
      <div className="fa-panel-input">
        <Input.TextArea
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onPressEnter={(e) => {
            if (!e.shiftKey) {
              e.preventDefault();
              handleSend();
            }
          }}
          placeholder="输入问题，Enter 发送，Shift+Enter 换行"
          rows={1}
          disabled={loading || !initialized}
          maxLength={INPUT_MAX_CHARS}
          showCount
        />
        {loading ? (
          <Button className="fa-send-btn" onClick={onStop}>
            停止
          </Button>
        ) : (
          <Button
            type="primary"
            className="fa-send-btn"
            icon={<SendOutlined />}
            onClick={() => handleSend()}
            disabled={!inputValue.trim() || !initialized}
          >
            发送
          </Button>
        )}
      </div>
    </div>
  );
};

export default AssistantPanel;
