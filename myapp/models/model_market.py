import json
from datetime import datetime

from flask_appbuilder import Model
from myapp import db


class ModelMarketModel(Model):
    __tablename__ = "model_market_model"

    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)

    name = db.Column(db.String(128), nullable=False, unique=True)
    display_name = db.Column(db.String(255), nullable=False)
    category = db.Column(db.String(64))
    task_type = db.Column(db.String(64))
    framework = db.Column(db.String(64))
    description = db.Column(db.Text)
    cover_url = db.Column(db.String(1024))
    tags = db.Column(db.String(1024))

    support_experience = db.Column(db.Boolean, default=True)
    support_develop = db.Column(db.Boolean, default=True)
    support_finetune = db.Column(db.Boolean, default=True)
    support_deploy = db.Column(db.Boolean, default=True)

    model_path = db.Column(db.String(1024))
    default_version = db.Column(db.String(128), default="v1")

    python_version = db.Column(db.String(32), default="3.10")
    cuda_version = db.Column(db.String(32), default="11.8")

    notebook_image = db.Column(db.String(1024))
    finetune_image = db.Column(db.String(1024))
    inference_image = db.Column(db.String(1024))

    default_cpu = db.Column(db.String(32), default="2")
    default_memory = db.Column(db.String(32), default="4G")
    default_gpu = db.Column(db.String(32), default="0")
    default_ports = db.Column(db.String(64), default="8000")

    volume_mount = db.Column(db.Text)
    command = db.Column(db.Text)
    env_json = db.Column(db.Text)

    notebook_template = db.Column(db.Text)
    finetune_template = db.Column(db.Text)
    deploy_template = db.Column(db.Text)

    demo_input_type = db.Column(db.String(64))
    demo_output_type = db.Column(db.String(64))
    demo_dataset_url = db.Column(db.String(1024))
    api_schema_json = db.Column(db.Text)

    status = db.Column(db.String(32), default="online")
    created_by = db.Column(db.BigInteger)

    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)

    def env(self):
        return json.loads(self.env_json or "{}")

    def api_schema(self):
        return json.loads(self.api_schema_json or "{}")

    def to_dict(self):
        # Determine best image to show as primary "image" field
        primary_image = self.inference_image or self.finetune_image or self.notebook_image or ""

        return {
            "id": self.id,
            "name": self.name,
            "display_name": self.display_name,
            "category": self.category,
            "task_type": self.task_type,
            "framework": self.framework,
            "description": self.description,
            "cover_url": self.cover_url,
            "tags": self.tags.split(",") if self.tags else [],

            "support_experience": bool(self.support_experience),
            "support_develop": bool(self.support_develop),
            "support_finetune": bool(self.support_finetune),
            "support_deploy": bool(self.support_deploy),

            "model_path": self.model_path,
            "default_version": self.default_version,
            "python_version": self.python_version or "3.10",
            "cuda_version": self.cuda_version or "11.8",
            "image": primary_image,
            "notebook_image": self.notebook_image,
            "finetune_image": self.finetune_image,
            "inference_image": self.inference_image,
            "default_cpu": self.default_cpu,
            "default_memory": self.default_memory,
            "default_gpu": self.default_gpu,
            "default_ports": self.default_ports,
            "volume_mount": self.volume_mount,
            "env_json": self.env_json,

            "demo_input_type": self.demo_input_type,
            "demo_output_type": self.demo_output_type,
            "api_schema": self.api_schema(),
            "status": self.status,
        }


class ModelMarketAction(db.Model):
    __tablename__ = "model_market_action"

    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)

    model_id = db.Column(db.BigInteger, nullable=False)
    action_type = db.Column(db.String(32), nullable=False)
    status = db.Column(db.String(32), default="created")

    target_type = db.Column(db.String(64))
    target_id = db.Column(db.BigInteger)
    target_name = db.Column(db.String(255))
    target_url = db.Column(db.String(1024))

    project_id = db.Column(db.BigInteger)
    namespace = db.Column(db.String(255))

    request_json = db.Column(db.Text)
    response_json = db.Column(db.Text)
    error_msg = db.Column(db.Text)

    created_by = db.Column(db.BigInteger)
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)


class ModelMarketService(db.Model):
    __tablename__ = "model_market_service"

    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)

    model_id = db.Column(db.BigInteger, nullable=False)
    action_id = db.Column(db.BigInteger)

    service_id = db.Column(db.BigInteger, nullable=False, unique=True)
    service_name = db.Column(db.String(255))
    service_status = db.Column(db.String(64), default="created")

    project_id = db.Column(db.BigInteger)
    namespace = db.Column(db.String(255))

    endpoint = db.Column(db.String(1024))
    pc_demo_url = db.Column(db.String(1024))
    mobile_demo_url = db.Column(db.String(1024))

    api_schema_json = db.Column(db.Text)
    extra_json = db.Column(db.Text)

    created_by = db.Column(db.BigInteger)
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)
