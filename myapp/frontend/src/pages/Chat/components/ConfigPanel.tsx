/**
 * ConfigPanel — 凭证配置抽屉
 * =============================
 * 右侧滑出，根据后端返回的 configFields 动态渲染表单。
 * - 文本、密码、下拉选择、文本域、JSON 编辑器
 * - 机器人/AI助手各自的凭证结构不同
 */

import React, { useEffect, useState } from 'react';
import {
  Drawer,
  Form,
  Input,
  Select,
  Button,
  message,
  Space,
  Divider,
  Typography,
  Spin,
} from 'antd';
import { SettingOutlined, SaveOutlined } from '@ant-design/icons';
import { getAgentDetail, updateAgentConfig } from '../api';
import type { IAgentItem, IAgentDetail, IConfigField } from '../types';
import NotificationSettings from '../../NotificationSettings';

const { Text } = Typography;
const { TextArea } = Input;

interface ConfigPanelProps {
  agent: IAgentItem | null;
  visible: boolean;
  onClose: () => void;
  onSaved: () => void;
}

/** 根据 configField.type 渲染对应的表单控件 */
function renderField(field: IConfigField) {
  const commonProps = {
    placeholder: field.placeholder || `请输入${field.label}`,
  };

  switch (field.type) {
    case 'password':
      return <Input.Password {...commonProps} />;
    case 'textarea':
      return <TextArea rows={3} {...commonProps} />;
    case 'select':
      return (
        <Select {...commonProps} allowClear>
          {(field.options || []).map((opt) => (
            <Select.Option key={opt.value} value={opt.value}>
              {opt.label}
            </Select.Option>
          ))}
        </Select>
      );
    case 'json':
      return <TextArea rows={5} {...commonProps} placeholder='{"key":"value"}' />;
    default:
      return <Input {...commonProps} />;
  }
}

const ConfigPanel: React.FC<ConfigPanelProps> = ({
  agent,
  visible,
  onClose,
  onSaved,
}) => {
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [configFields, setConfigFields] = useState<IConfigField[]>([]);

  // 打开时加载AI助手详情
  useEffect(() => {
    if (visible && agent && agent.name !== 'dingtalk') {
      loadDetail(agent.name);
    }
  }, [visible, agent?.name]);

  const loadDetail = async (name: string) => {
    setLoading(true);
    try {
      const detail = await getAgentDetail(name);
      if (detail) {
        setConfigFields(detail.configFields || []);

        // 将 credentials 中的值回填表单
        const creds = detail.credentials || {};
        const formValues: Record<string, any> = {};

        (detail.configFields || []).forEach((field) => {
          if (creds[field.key] !== undefined) {
            formValues[field.key] =
              typeof creds[field.key] === 'object'
                ? JSON.stringify(creds[field.key], null, 2)
                : creds[field.key];
          }
        });

        form.setFieldsValue(formValues);
      }
    } catch (err) {
      message.error('加载配置失败');
    } finally {
      setLoading(false);
    }
  };

  /** 保存凭证配置 */
  const handleSave = async () => {
    if (!agent) return;

    try {
      const values = await form.validateFields();
      setSaving(true);

      // 构建 credentials 对象
      const credentials: Record<string, any> = {};
      configFields.forEach((field) => {
        let val = values[field.key];
        if (val === undefined || val === '') return;

        // JSON 类型解析
        if (field.type === 'json' && typeof val === 'string') {
          try {
            val = JSON.parse(val);
          } catch {
            message.error(`${field.label} 不是有效的 JSON`);
            throw new Error('Invalid JSON');
          }
        }
        credentials[field.key] = val;
      });

      const success = await updateAgentConfig(agent.name, { credentials });
      if (success) {
        message.success('配置已保存');
        onSaved();
        onClose();
      } else {
        message.error('保存失败');
      }
    } catch (err: any) {
      if (err?.message !== 'Invalid JSON') {
        message.error('保存失败');
      }
    } finally {
      setSaving(false);
    }
  };

  if (agent?.name === 'dingtalk') {
    return (
      <Drawer
        title={
          <Space>
            <SettingOutlined />
            <span>{agent.label || agent.name}</span>
          </Space>
        }
        placement="right"
        width={760}
        open={visible}
        onClose={onClose}
      >
        <NotificationSettings />
      </Drawer>
    );
  }

  return (
    <Drawer
      title={
        <Space>
          <SettingOutlined />
          <span>凭证配置 — {agent?.label || agent?.name}</span>
        </Space>
      }
      placement="right"
      width={420}
      open={visible}
      onClose={onClose}
      extra={
        <Button
          type="primary"
          icon={<SaveOutlined />}
          onClick={handleSave}
          loading={saving}
        >
          保存
        </Button>
      }
    >
      <Spin spinning={loading}>
        {configFields.length === 0 && !loading ? (
          <div style={{ textAlign: 'center', padding: 40 }}>
            <Text type="secondary">该AI助手无需额外配置</Text>
          </div>
        ) : (
          <Form
            form={form}
            layout="vertical"
            autoComplete="off"
          >
            {configFields.map((field) => (
              <Form.Item
                key={field.key}
                name={field.key}
                label={field.label}
                rules={
                  field.required
                    ? [{ required: true, message: `请输入${field.label}` }]
                    : []
                }
              >
                {renderField(field)}
              </Form.Item>
            ))}
          </Form>
        )}
      </Spin>
    </Drawer>
  );
};

export default ConfigPanel;
