/**
 * Chat Page — AI助手对话主页
 * =============================
 * 布局：左侧边栏（280px） + 右侧聊天区域（flex 填充）
 *
 * 交互流程：
 *   1. 左侧边栏选择AI助手 → 右侧显示欢迎语 / 聊天界面
 *   2. 点击"设置"按钮 → 右侧滑出凭证配置面板
 *   3. 输入消息 → 发送到后端 → SSE 流式接收回答
 */

import React, { useState, useCallback, useMemo } from 'react';

import Sidebar from './components/Sidebar';
import ChatArea from './components/ChatArea';
import ConfigPanel from './components/ConfigPanel';
import type { IAgentItem, AgentCategory } from './types';
import './Chat.less';

const LS_KEY_AGENT = 'chat_v2_selected_agent';

/** 从 localStorage 读取上次选中的AI助手 */
function readPersistedAgent(): { name: string; category: string } | null {
  try {
    const raw = localStorage.getItem(LS_KEY_AGENT);
    if (raw) return JSON.parse(raw);
  } catch {}
  return null;
}

/** 持久化当前选中的AI助手 */
function persistAgent(agent: IAgentItem | null) {
  if (agent) {
    localStorage.setItem(
      LS_KEY_AGENT,
      JSON.stringify({ name: agent.name, category: agent.agentCategory }),
    );
  } else {
    localStorage.removeItem(LS_KEY_AGENT);
  }
}

const ChatPage: React.FC = () => {
  const persisted = useMemo(() => readPersistedAgent(), []);
  const [selectedAgent, setSelectedAgent] = useState<IAgentItem | null>(null);
  const [configVisible, setConfigVisible] = useState(false);

  /** 选择AI助手 */
  const handleSelectAgent = useCallback((agent: IAgentItem) => {
    setSelectedAgent(agent);
    persistAgent(agent);
    setConfigVisible(false);
  }, []);

  return (
    <div className="chat-page">
      {/* 左侧边栏 */}
      <div className="chat-sidebar-wrapper">
        <Sidebar
          selectedAgent={selectedAgent}
          onSelectAgent={handleSelectAgent}
          initialTab={persisted?.category as AgentCategory | undefined}
          initialAgentName={persisted?.name}
        />
      </div>

      {/* 右侧区域 */}
      <div className="chat-main">
        {/* 配置按钮（选中AI助手后显示） */}
        {/* 聊天区域 */}
        <ChatArea
          agent={selectedAgent}
          onOpenSettings={
            selectedAgent && selectedAgent.agentCategory !== 'knowledge_base'
              ? () => setConfigVisible(true)
              : undefined
          }
        />

        {/* 凭证配置面板 */}
        <ConfigPanel
          agent={selectedAgent}
          visible={configVisible}
          onClose={() => setConfigVisible(false)}
          onSaved={() => {
            // 配置更新后刷新（可选）
          }}
        />
      </div>
    </div>
  );
};

export default ChatPage;
