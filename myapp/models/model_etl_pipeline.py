from flask_appbuilder import Model
from sqlalchemy.orm import relationship
from sqlalchemy import Text
import datetime
import json
import urllib.parse

from myapp.models.helpers import AuditMixinNullable
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
from myapp import app
from myapp.models.helpers import ImportMixin
from myapp.models.model_team import Project
from sqlalchemy import Column, Integer, String, ForeignKey
from flask import Markup
from myapp.models.base import MyappModelBase
metadata = Model.metadata
conf = app.config



class ETL_Pipeline(Model,ImportMixin,AuditMixinNullable,MyappModelBase):
    __tablename__ = 'etl_pipeline'
    id = Column(Integer, primary_key=True,comment='id主键')
    name = Column(String(100),nullable=False,unique=True,comment='英文名')
    describe = Column(String(200),nullable=False,comment='描述')
    project_id = Column(Integer, ForeignKey('project.id'),nullable=False,comment='项目组id')  # 定义外键
    project = relationship(
        "Project", foreign_keys=[project_id], lazy='selectin'
    )
    workflow = Column(String(200),nullable=True,comment='调度引擎')   #
    dag_json = Column(Text(65536),nullable=True,default='{}',comment='pipeline的上下游关系')  #
    config = Column(Text(65536),default='{}',comment='pipeline的全局配置')   #
    expand = Column(Text(65536),default='[]',comment='扩展参数')

    # export_children = "etl_task"

    def __repr__(self):
        return self.name

    @property
    def etl_pipeline_url(self):
        pipeline_url="/etl_pipeline_modelview/api/web/" +str(self.id)
        return Markup(f'<a target=_blank href="{pipeline_url}">{self.describe}</a>')


    def clone(self):
        return ETL_Pipeline(
            name=self.name.replace('_', '-'),
            describe=self.describe,
            project_id=self.project_id,
            dag_json=self.dag_json,
            config=self.config,
            expand=self.expand,
            workflow=self.workflow
        )


class ETL_Task(Model,ImportMixin,AuditMixinNullable,MyappModelBase):
    __tablename__ = 'etl_task'
    id = Column(Integer, primary_key=True,comment='id主键')
    name = Column(String(100),nullable=False,unique=True,comment='英文名')
    describe = Column(String(200),nullable=False,comment='描述')
    # etl_pipeline_id = Column(Integer, ForeignKey('etl_pipeline.id'),nullable=False)  # 定义外键
    # etl_pipeline = relationship(
    #     "ETL_Pipeline", foreign_keys=[etl_pipeline_id]
    # )
    etl_pipeline_id = Column(Integer,comment='任务流id')
    template = Column(String(100),nullable=False,comment='任务模板')
    task_args = Column(Text(65536),default='{}',comment='任务参数')
    etl_task_id = Column(String(100),nullable=False,comment='远程调度系统任务id')
    expand = Column(Text(65536),default='[]',comment='扩展参数')


    # export_parent = "etl_pipeline"


    def __repr__(self):
        return self.name


class ETL_Task_Instance(Model, ImportMixin, AuditMixinNullable, MyappModelBase):
    __tablename__ = 'etl_task_instance'
    id = Column(Integer, primary_key=True, comment='id主键')
    run_id = Column(String(100), nullable=False, comment='运行id')
    etl_pipeline_id = Column(Integer, ForeignKey('etl_pipeline.id'), nullable=False, comment='任务流id')
    etl_pipeline = relationship("ETL_Pipeline", foreign_keys=[etl_pipeline_id], lazy='selectin')
    etl_task_id = Column(Integer, ForeignKey('etl_task.id'), nullable=True, comment='任务id')
    etl_task = relationship("ETL_Task", foreign_keys=[etl_task_id], lazy='selectin')
    workflow = Column(String(100), nullable=False, comment='调度引擎')
    status = Column(String(50), nullable=False, default='Submitted', comment='状态')
    scheduler_url = Column(String(500), nullable=True, comment='调度实例地址')
    log_url = Column(String(500), nullable=True, comment='日志地址')
    expand = Column(Text(65536), default='{}', comment='扩展参数')

    def __repr__(self):
        return self.run_id

    @property
    def pipeline_url(self):
        if self.etl_pipeline:
            return Markup(
                f'<a target=_blank href="/etl_pipeline_modelview/api/web/{self.etl_pipeline.id}">{self.etl_pipeline.describe}</a>'
            )
        return Markup('-')

    @property
    def task_url(self):
        if self.etl_task:
            filter_value = urllib.parse.quote(
                json.dumps([{"key": "id", "value": self.etl_task.id}], ensure_ascii=False)
            )
            url = conf.get('MODEL_URLS', {}).get('etl_task', '') + '?filter=' + filter_value
            return Markup(f'<a target=_blank href="{url}">{self.etl_task.describe or self.etl_task.name}</a>')
        return Markup('-')

    @property
    def elapsed_time(self):
        if not self.created_on:
            return 'unknown'
        finish_time = self.changed_on or datetime.datetime.now()
        elapsed = (finish_time - self.created_on).total_seconds() / 60 / 60
        return str(round(elapsed, 2)) + "h"

    @property
    def trace(self):
        if self.scheduler_url:
            return Markup(f'<a target=_blank href="{self.scheduler_url}">{__("跟踪")}</a>')
        return Markup('-')

    @property
    def log(self):
        if self.log_url:
            return Markup(f'<a target=_blank href="{self.log_url}">{__("日志")}</a>')
        return Markup('-')



