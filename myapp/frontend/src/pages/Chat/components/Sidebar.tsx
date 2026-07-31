/**
 * Sidebar — 左侧边栏
 * ====================
 * 两个 Tab：AI助手 | 机器人（点击进入对话）
 * 底部固定：知识库（外链跳转）
 */

import React, { useEffect, useState } from 'react';
import { Tabs, List, Avatar, Badge, Input, Spin, Empty } from 'antd';
import {
  RobotOutlined,
  BookOutlined,
  BugOutlined,
  LinkOutlined,
  SearchOutlined,
} from '@ant-design/icons';
import { getAgents } from '../api';
import type { IAgentItem, AgentCategory } from '../types';

/** 去除 HTML 标签，保留纯文本 */
function stripHtml(html: string): string {
  if (!html) return '';
  const tmp = document.createElement('div');
  tmp.innerHTML = html;
  return tmp.textContent || tmp.innerText || '';
}

const { TabPane } = Tabs;

const LS_KEY_TAB = 'chat_v2_active_tab';

interface SidebarProps {
  selectedAgent: IAgentItem | null;
  onSelectAgent: (agent: IAgentItem) => void;
  /* 刷新恢复：初始 Tab 和需要自动选中的AI助手名 */
  initialTab?: AgentCategory;
  initialAgentName?: string;
}

/** Tab 配置 — 定义三个分类的图标、标签、分类值 */
const TAB_CONFIG: { key: AgentCategory; label: string; icon: React.ReactNode }[] = [
  { key: 'agent', label: 'AI助手', icon: <BugOutlined /> },
  { key: 'robot', label: '机器人', icon: <RobotOutlined /> },
];

const Sidebar: React.FC<SidebarProps> = ({
  selectedAgent,
  onSelectAgent,
  initialTab,
  initialAgentName,
}) => {
  const [activeTab, setActiveTab] = useState<AgentCategory>(
    initialTab || 'agent',
  );
  const [restored, setRestored] = useState(false);
  const [agents, setAgents] = useState<IAgentItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [searchText, setSearchText] = useState('');
  const [kbAgents, setKbAgents] = useState<IAgentItem[]>([]);

  // 挂载时加载知识库列表（仅一次）
  useEffect(() => {
    getAgents('knowledge_base')
      .then(setKbAgents)
      .catch(() => {});
  }, []);

  // 切换 Tab 时重新加载AI助手列表
  useEffect(() => {
    loadAgents(activeTab);
  }, [activeTab]);

  const loadAgents = async (category: AgentCategory) => {
    setLoading(true);
    try {
      const list = await getAgents(category);
      setAgents(list);
      // 刷新恢复：自动选中上次的AI助手
      if (!restored && initialAgentName) {
        const target = list.find((a) => a.name === initialAgentName);
        if (target) {
          setRestored(true);
          onSelectAgent(target);
        }
      }
    } catch (err) {
      console.error('加载AI助手列表失败', err);
    } finally {
      setLoading(false);
    }
  };

  // 搜索过滤
  const filteredAgents = agents.filter((a) => {
    if (!searchText) return true;
    const kw = searchText.toLowerCase();
    return (
      a.name.toLowerCase().includes(kw) ||
      a.label.toLowerCase().includes(kw)
    );
  });

  /** 渲染每个AI助手的 SVG 图标 */
  const renderIcon = (iconSvg: string) => {
    if (!iconSvg) {
      return (
        <Avatar
          icon={<RobotOutlined />}
          style={{ backgroundColor: '#1677ff' }}
          size={36}
        />
      );
    }
    return (
      <Avatar
        src={`data:image/svg+xml;utf8,${encodeURIComponent(iconSvg)}`}
        size={36}
      />
    );
  };

  /** 处理AI助手点击 */
  const handleClick = (agent: IAgentItem) => {
    onSelectAgent(agent);
  };

  return (
    <div className="chat-sidebar">
      <Tabs
        activeKey={activeTab}
        onChange={(key) => {
          setActiveTab(key as AgentCategory);
          setSearchText('');
          localStorage.setItem(LS_KEY_TAB, key);
        }}
        size="small"
        centered
      >
        {TAB_CONFIG.map((tab) => (
          <TabPane
            tab={
              <span>
                {tab.icon}
                <span style={{ marginLeft: 4 }}>{tab.label}</span>
              </span>
            }
            key={tab.key}
          />
        ))}
      </Tabs>

      <div className="chat-sidebar-search">
        <Input
          prefix={<SearchOutlined />}
          placeholder="搜索AI助手..."
          value={searchText}
          onChange={(e) => setSearchText(e.target.value)}
          allowClear
          size="small"
        />
      </div>

      <div className="chat-sidebar-list">
        <Spin spinning={loading}>
          {filteredAgents.length === 0 && !loading ? (
            <Empty
              image={Empty.PRESENTED_IMAGE_SIMPLE}
              description="暂无可用AI助手"
              style={{ marginTop: 40 }}
            />
          ) : (
            <List
              dataSource={filteredAgents}
              renderItem={(agent) => {
                const isActive = selectedAgent?.name === agent.name;

                return (
                  <div
                    key={agent.name}
                    className={`chat-sidebar-item ${isActive ? 'active' : ''}`}
                    onClick={() => handleClick(agent)}
                  >
                    <div className="chat-sidebar-item-icon">
                      {renderIcon(agent.icon)}
                    </div>
                    <div className="chat-sidebar-item-content">
                      <div className="chat-sidebar-item-label">
                        {agent.label || agent.name}
                      </div>
                      {agent.hello && (
                        <div className="chat-sidebar-item-desc">
                          {stripHtml(agent.hello)}
                        </div>
                      )}
                    </div>
                    {isActive && (
                      <Badge status="processing" style={{ marginRight: 8 }} />
                    )}
                  </div>
                );
              }}
            />
          )}
        </Spin>
      </div>

      {/* ======== 知识库（外链跳转） ======== */}
      {kbAgents.length > 0 && (
        <div className="chat-sidebar-kb">
          <div className="chat-sidebar-kb-header">
            <BookOutlined style={{ marginRight: 6 }} />
            知识库
          </div>
          <div className="chat-sidebar-kb-list">
            {kbAgents.map((kb) => (
              <a
                key={kb.name}
                className="chat-sidebar-kb-item"
                href={kb.externalUrl || '#'}
                target={kb.openInNewTab !== false ? '_blank' : '_self'}
                rel="noreferrer"
                title={stripHtml(kb.hello) || kb.label || kb.name}
              >
                <div className="chat-sidebar-kb-item-icon">
                  {kb.icon ? (
                    <Avatar
                      src={`data:image/svg+xml;utf8,${encodeURIComponent(kb.icon)}`}
                      size={28}
                    />
                  ) : (
                    <img
                      src="/frontend/logo-wiki.png"
                      alt="知识库"
                      style={{ width: 28, height: 28, borderRadius: 4, objectFit: 'contain' }}
                    />
                  )}
                </div>
                <div className="chat-sidebar-kb-item-content">
                  <span className="chat-sidebar-kb-item-label">
                    {kb.label || kb.name}
                  </span>
                  <LinkOutlined className="chat-sidebar-kb-item-link" />
                </div>
              </a>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default Sidebar;
