/**
 * Chat 模块类型定义
 * ====================
 * 定义智能体、会话、消息、配置字段等核心类型
 */

/** 智能体分类枚举 */
export type AgentCategory = 'robot' | 'knowledge_base' | 'agent';

/** 智能体列表项（不含敏感凭证） */
export interface IAgentItem {
  name: string;
  label: string;
  icon: string;
  agentCategory: AgentCategory;
  chatType: string;
  hello: string;
  tips: string[];
  owner: string;
  /** 知识库外链地址 */
  externalUrl?: string;
  openInNewTab?: boolean;
}

/** 配置字段描述 — 前端根据此动态渲染表单 */
export interface IConfigField {
  key: string;
  label: string;
  type: 'text' | 'password' | 'select' | 'textarea' | 'json';
  required?: boolean;
  placeholder?: string;
  options?: { label: string; value: string }[];
}

/** 智能体详情（含凭证和配置字段） */
export interface IAgentDetail extends IAgentItem {
  serviceType: string;
  serviceConfig: Record<string, any>;
  credentials: Record<string, any>;
  knowledge: Record<string, any>;
  sessionNum: number;
  prompt: string;
  configFields: IConfigField[];
  expand: Record<string, any>;
}

/** 会话 */
export interface ISession {
  id: string;
  agentName: string;
  title: string;
  lastMessage?: string;
  createdAt: number;
  messageCount?: number;
}

/** 聊天消息 */
export interface IMessage {
  role: 'user' | 'assistant';
  content: string;
  timestamp?: number;
}

/** 发送聊天请求 */
export interface IChatRequest {
  session_id: string;
  search_text: string;
  stream?: boolean;
}

/** SSE 流式响应的单个 chunk */
export interface IChatChunk {
  status: number;
  result: Array<{ text: string }>;
  finish: boolean;
  message: string;
}

/** 凭证更新请求 */
export interface ICredentialsUpdate {
  credentials?: Record<string, any>;
  configFields?: IConfigField[];
  prompt?: string;
  hello?: string;
  tips?: string[];
  session_num?: number;
}
