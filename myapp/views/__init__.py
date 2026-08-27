from . import base
from . import home
from . import route
from . import view_k8s
from . import view_team
from . import view_metadata
from . import view_metadata_metric
from . import view_dimension
from . import view_dataset
from . import view_storage
from . import view_images
from . import view_etl_pipeline
from . import view_notebook
from . import view_docker

from . import view_job_template
from . import view_task
from . import view_pipeline
from . import view_runhistory
from . import view_workflow
from . import view_nni
from . import view_train_model
from . import view_serving
from . import view_inferenceserving
# from . import view_service_pipeline
from . import view_log
from . import view_user_role

from . import view_sqllab
from . import view_aihub
from . import view_total_resource
from . import view_node
from . import view_chat
from . import view_bill
from . import view_llm_gateway
from . import view_model_market
from . import view_training_monitor
from . import view_notification
from . import view_inference_monitor
from . import view_runtime
from . import view_assistant

# ========== 🆕 Chat API v2.0 Blueprint ==========
from . import view_chat_v2
from myapp import appbuilder

appbuilder.get_app.register_blueprint(view_chat_v2.chat_api_bp)
appbuilder.get_app.register_blueprint(view_notification.notification_api_bp)
appbuilder.get_app.register_blueprint(view_assistant.assistant_bp)
