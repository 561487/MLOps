


# 迁移 scratch 验证用临时配置（MySQL 本地 scratch 库）：
# 复用真实配置全部键，仅把数据库指向本地 mysql 容器的 kubeflow_scratch 库。
# 注意：该文件位于 bind mount 的 myapp 目录（容器 /home/myapp/myapp/scratch_config.py），
# 仅用于本地迁移验证，勿在正式环境启用。
SQLALCHEMY_DATABASE_URI = 'mysql+pymysql://root:admin@mysql:3306/kubeflow_scratch?charset=utf8mb4'
SQLALCHEMY_TRACK_MODIFICATIONS = False
