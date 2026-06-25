#!/bin/bash
# 自动获取 MinIO K8s ClusterIP 并更新 docker-compose.yml 中的 extra_hosts 配置
# 用法: bash update-minio-host.sh [kubeconfig-path]
#   kubeconfig-path: 可选，默认为 ./kubeconfig/dev-kubeconfig

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
KUBECONFIG="${1:-$SCRIPT_DIR/kubeconfig/dev-kubeconfig}"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.yml"

echo "=== 获取 MinIO K8s ClusterIP ==="

# 获取 MinIO Service 的 ClusterIP
MINIO_IP=$(kubectl --kubeconfig="$KUBECONFIG" get svc minio -n kubeflow \
  -o jsonpath='{.spec.clusterIP}' 2>/dev/null)

if [ -z "$MINIO_IP" ]; then
  echo "ERROR: 无法获取 minio.kubeflow Service 的 ClusterIP"
  echo "请检查: kubectl --kubeconfig=$KUBECONFIG get svc -n kubeflow minio"
  exit 1
fi

echo "MinIO ClusterIP: $MINIO_IP"

# 更新 docker-compose.yml 中的 extra_hosts
if grep -q "minio.kubeflow:" "$COMPOSE_FILE"; then
  # 替换已有的 IP
  sed -i "s/minio.kubeflow:[0-9.]\+/minio.kubeflow:$MINIO_IP/g" "$COMPOSE_FILE"
  echo "已更新 docker-compose.yml 中的 minio.kubeflow → $MINIO_IP"
else
  echo "WARNING: docker-compose.yml 中未找到 minio.kubeflow 的 extra_hosts 配置"
  echo "请手动确保 myapp 服务包含: extra_hosts: [\"minio.kubeflow:$MINIO_IP\"]"
fi

echo ""
echo "=== 重启 myapp 容器使配置生效 ==="
cd "$SCRIPT_DIR"
docker compose up -d myapp

echo ""
echo "=== 验证连通性 ==="
sleep 2
docker exec docker-myapp-1 getent hosts minio.kubeflow
docker exec docker-myapp-1 curl -s -o /dev/null -w "HTTP状态码: %{http_code}\n" --connect-timeout 3 http://minio.kubeflow:9000
echo ""
echo "✅ MinIO 网络连通性修复完成"
