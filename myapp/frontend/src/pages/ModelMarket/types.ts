// 模型市场 - 类型定义

export interface IInputSchema {
  name: string;
  type: 'file' | 'text' | 'number' | 'select' | 'image' | 'audio';
  label: string;
  accept?: string;
  required: boolean;
  options?: string[];
  default?: any;
  placeholder?: string;
}

export interface IOutputSchema {
  type: 'classification' | 'text' | 'image' | 'audio' | 'json' | 'chat';
  fields?: string[];
  description?: string;
}

export interface IModelMarketItem {
  id: number;
  name: string;
  display_name: string;
  description: string;
  category: 'vision' | 'audio' | 'nlp' | 'multimodal' | 'llm';
  task_type: string;
  framework: string;
  tags: string[];
  cover: string;
  support_experience: boolean;
  support_develop: boolean;
  support_finetune: boolean;
  support_deploy: boolean;
  python_version?: string;
  cuda_version?: string;
  image?: string;
  model_path?: string;
  notebook_image?: string;
  finetune_image?: string;
  inference_image?: string;
  volume_mount?: string;
  input_schema: IInputSchema[];
  output_schema: IOutputSchema;
  hot?: number;
  version?: string;
}

export interface IModelMarketListParams {
  category?: string;
  keyword?: string;
  task_type?: string;
  framework?: string;
  support_experience?: boolean;
  support_develop?: boolean;
  support_finetune?: boolean;
  support_deploy?: boolean;
  page?: number;
  page_size?: number;
}

export interface IModelMarketListResponse {
  data: IModelMarketItem[];
  total: number;
  page: number;
  page_size: number;
}

export interface IModelCategory {
  key: string;
  label: string;
  count: number;
}

export interface IInferRequest {
  inputs: Record<string, any>;
  files?: File[];
}

export interface IInferResponse {
  success: boolean;
  result: any;
  task_id?: string;
  message?: string;
  source?: string;
}

export interface INotebookConfig {
  name?: string;
  image: string;
  python_version: string;
  cuda_version: string;
  cpu: number;
  memory: number;
  gpu: number;
  work_dir: string;
  mount_demo: boolean;
}

export interface IDeployConfig {
  name: string;
  image: string;
  cpu: number;
  memory: number;
  gpu: number;
  replicas: number;
  env_vars: Record<string, string>;
  inference_entry: string;
  expose_type: 'internal' | 'gateway';
  enable_experience: boolean;
  model_version?: string;
  model_path?: string;
  command?: string;
  working_dir?: string;
}

export interface IFinetuneConfig {
  dataset_id: number;
  epochs: number;
  batch_size: number;
  learning_rate: number;
  image: string;
  cpu: number;
  gpu: number;
  memory: number;
  output_model_name: string;
  auto_register: boolean;
}

export type TaskStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';

export interface ITaskInfo {
  id: string;
  type: 'notebook' | 'finetune' | 'deploy' | 'infer';
  status: TaskStatus;
  progress: number;
  message: string;
  result?: any;
  created_at: string;
  updated_at: string;
  logs?: string[];
}

// ===== Backend response types for one-click actions =====

export interface INotebookResponse {
  // Top-level response fields (axios res.data = backend JSON body)
  code?: number;
  data?: INotebookResponseData;
  message?: string;
  // Legacy flat fields (used when component unwraps data)
  success?: boolean;
  existed?: boolean;
  action_id?: number;
  target_type?: string;
  target_id?: number;
  target_name?: string;
  url?: string;
  notebook_id?: number;
  notebook_name?: string;
  status?: string;
  redirect_url?: string;
  jupyter_url?: string | null;
  ready?: boolean;
}

export interface INotebookResponseData {
  action_id?: number;
  target_type?: string;
  target_id?: number;
  target_name?: string;
  url?: string;
  jupyter_url?: string;
  ready?: boolean;
}

export interface IPipelineResponse {
  // Backend flat response (after unwrapping data)
  success?: boolean;
  existed?: boolean;
  action_id?: number;
  target_type?: string;
  target_id?: number;
  target_name?: string;
  url?: string;
  pipeline_id?: number;
  pipeline_name?: string;
  run_id?: string;
  run_status?: string;
  redirect_url?: string;
  message?: string;
  // Wrapper fields
  code?: number;
  data?: any;
}

export interface IDeployResponse {
  success?: boolean;
  existed?: boolean;
  action_id?: number;
  target_type?: string;
  target_id?: number;
  target_name?: string;
  url?: string;
  service_id?: number;
  service_name?: string;
  market_service_id?: number;
  status?: string;
  api_url?: string;
  redirect_url?: string;
  message?: string;
  model_version?: string;
  model_path?: string;
  code?: number;
  data?: any;
}
