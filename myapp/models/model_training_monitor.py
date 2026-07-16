"""
训练监控表 —— 存储 SwanLab 等监控平台的实验记录映射
一个运行实例（run_id）可能包含多个训练节点，每个节点一条记录
"""
from flask_appbuilder import Model
from sqlalchemy import Column, Integer, String, Text, DateTime
from myapp.models.helpers import AuditMixinNullable
from myapp.models.base import MyappModelBase
from myapp import db


class TrainingMonitor(Model, AuditMixinNullable, MyappModelBase):
    """训练任务监控记录，关联 run_id ↔ monitor_url"""

    __tablename__ = 'mlops_training_monitor'

    id = Column(Integer, primary_key=True, comment='自增主键')

    # ---- 运行实例信息 ----
    pipeline_id = Column(String(64), nullable=True, default='', comment='任务流 ID')
    pipeline_name = Column(String(256), nullable=True, default='', comment='任务流名称')

    run_id = Column(String(128), nullable=False, default='', comment='运行实例 run-id（取自 Workflow.labels["run-id"]）')
    workflow_name = Column(String(256), nullable=True, default='', comment='Argo Workflow 名称')

    # ---- 节点信息 ----
    task_id = Column(String(64), nullable=True, default='', comment='Task ID（backend Task 表主键）')
    task_name = Column(String(256), nullable=True, default='', comment='Task 名称')
    node_name = Column(String(256), nullable=False, default='', comment='DAG 节点名称（job_template name / task label）')
    pod_name = Column(String(256), nullable=True, default='', comment='K8s Pod 名称')
    job_template_name = Column(String(256), nullable=True, default='', comment='算子模板名称')

    # ---- 监控信息 ----
    monitor_type = Column(String(32), nullable=False, default='swanlab', comment='监控类型：swanlab / mlflow / wandb 等')
    monitor_url = Column(Text, nullable=True, comment='监控平台实验详情页地址')
    monitor_status = Column(String(32), nullable=False, default='INIT',
                            comment='监控状态：INIT / RUNNING / SUCCEEDED / FAILED')
    experiment_run_id = Column(String(256), nullable=True, default='',
                              comment='SwanLab run 目录名，例如 run-20260713_175627-4wm7y5d2')

    # ---- 元信息 ----
    creator = Column(String(100), nullable=True, default='', comment='创建者用户名')

    # label_columns 用于前端展示中文列名
    label_columns = {
        **MyappModelBase.label_columns,
        "pipeline_id": "任务流 ID",
        "pipeline_name": "任务流名称",
        "run_id": "运行实例 ID",
        "workflow_name": "Workflow 名称",
        "task_id": "Task ID",
        "task_name": "Task 名称",
        "node_name": "节点名称",
        "pod_name": "Pod 名称",
        "job_template_name": "算子模板",
        "monitor_type": "监控类型",
        "monitor_url": "监控地址",
        "monitor_status": "监控状态",
        "experiment_run_id": "SwanLab Run 目录名",
        "creator": "创建者",
    }

    def __repr__(self):
        return f"<TrainingMonitor(id={self.id}, run_id={self.run_id}, node={self.node_name}, type={self.monitor_type})>"
