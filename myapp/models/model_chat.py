import json
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
from flask_appbuilder import Model
from sqlalchemy import Text
from sqlalchemy import Column, Integer, String, ForeignKey ,Date,DateTime
from myapp import app
from sqlalchemy import Column, Integer, String

from flask import Markup
from myapp.models.base import MyappModelBase
metadata = Model.metadata
conf = app.config


class Chat(Model,MyappModelBase):
    __tablename__ = 'chat'

    id = Column(Integer, primary_key=True,comment='id主键')

    name = Column(String(200), nullable=True, default='',unique=True,comment='英文名')
    icon = Column(Text, nullable=True, default='',comment='图标svg内容')
    label = Column(String(200), nullable=True, default='',comment='中文名')
    doc = Column(String(200), nullable=True, default='',comment='文档链接')
    session_num = Column(String(200), nullable=True, default='0',comment='会话保持的个数，默认为1，只记录当前的会话')
    chat_type = Column(String(200), nullable=True, default='text',comment='聊天会话的界面类型')
    hello = Column(String(200), nullable=True, default='',comment='欢迎语')
    tips = Column(String(4000), nullable=True, default='',comment='提示词数组')
    knowledge = Column(Text, nullable=True, default='',comment='加载问题前面的先验知识')
    prompt = Column(Text, nullable=True, default='{{prompt}}',comment='提示词模板')
    service_type = Column(String(200), nullable=True, default='',comment='推理服务的类型')
    service_config = Column(Text, nullable=True, default='{}',comment='推理服务的配置，url header json等')
    owner = Column(String(2000), nullable=True, default='*',comment='可访问用户，* 表示所有用户')
    expand = Column(Text, nullable=True, default='{}',comment='扩展参数')

    # ========== 🆕 v3.0 新增字段 ==========
    # agent_category: 智能体分类，决定左侧 Tab 分组和交互方式
    #   'robot'          — 机器人：LLM 对话，配置简单（API Key + Endpoint）
    #   'knowledge_base' — 知识库：纯外链跳转，不参与对话
    #   'agent'          — 智能体：LLM 对话 + 工具调用，配置复杂（AppID + Secret + Token）
    agent_category = Column(
        String(50),
        nullable=True,
        default='robot',
        comment='智能体分类: robot(机器人) / knowledge_base(知识库外链) / agent(智能体)'
    )

    # credentials: 独立凭证配置，与 service_config 分离
    #   robot 示例:  {"authType":"api_key","apiKey":"sk-xxx","apiEndpoint":"http://...","description":"..."}
    #   agent 示例:  {"authType":"app_secret","appId":"app_xxx","secretKey":"sec_xxx","token":"jwt_xxx",...}
    #   knowledge_base: {}（不使用，外链凭证存在 expand 中）
    credentials = Column(
        Text,
        nullable=True,
        default='{}',
        comment='独立凭证配置，按 agent_category 不同结构不同'
    )

    def clone(self):
        return Chat(
            name=self.name,
            icon=self.icon,
            label=self.label+__("(副本)"),
            doc=self.doc,
            session_num=self.session_num,
            chat_type=self.chat_type,
            hello=self.hello,
            tips=self.tips,
            knowledge=self.knowledge,
            prompt=self.prompt,
            service_type=self.service_type,
            service_config=self.service_config,
            owner=self.owner,
            expand=self.expand,
            agent_category=self.agent_category,   # 🆕 复制时保留分类
            credentials=self.credentials,          # 🆕 复制时保留凭证
        )


class ChatLog(Model,MyappModelBase):
    __tablename__ = 'chat_log'
    id = Column(Integer, primary_key=True,comment='id主键')
    username = Column(String(400), nullable=False, default='',comment='用户名')
    chat_id = Column(Integer,comment='场景id')
    query = Column(String(5000), nullable=False, default='',comment='问题')
    answer = Column(String(5000), nullable=False, default='',comment='回答')
    manual_feedback = Column(String(400), nullable=False, default='',comment='反馈')
    answer_status = Column(String(400), nullable=False, default='',comment='回答状态')
    answer_cost = Column(String(400), nullable=False, default='',comment='话费时长')
    err_msg = Column(Text,comment='报错消息')
    created_on = Column(DateTime, nullable=True, default='',comment='创建时间')
    changed_on = Column(DateTime, nullable=True, default='',comment='修改时间')
