from myapp import app


def get_pipeline_priority_config(priority: str) -> dict:
    conf = app.config.get("PIPELINE_PRIORITY_CONFIG", {}) or {}
    default = app.config.get("PIPELINE_PRIORITY_DEFAULT", "high")
    key = (priority or default).lower()
    return conf.get(key, conf.get(default, {}))
