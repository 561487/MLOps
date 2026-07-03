#!/bin/bash
#
# 构建 ops 平台前后端镜像并推送到 Harbor
#
# 用法:
#   sh deploy_ops.sh                              # 默认：编译前端 + 构建推送前后端镜像
#   sh deploy_ops.sh -t 2026.07.01                # 指定版本号
#   sh deploy_ops.sh --frontend-assets            # 仅编译前端静态资源
#   sh deploy_ops.sh --backend-image              # 仅构建推送后端镜像
#   sh deploy_ops.sh --frontend-image             # 仅构建推送前端镜像
#   sh deploy_ops.sh --update-kustomize           # 仅更新 kustomization.yml
#   sh deploy_ops.sh --frontend-assets --backend-image --update-kustomize
#
# 配置: 复制 deploy_ops.env.example 为 deploy_ops.env 并填写 Harbor 凭证
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
KUSTOMIZATION_FILE="${PROJECT_ROOT}/install/kubernetes/cube/overlays/kustomization.yml"

if [[ -f "${SCRIPT_DIR}/deploy_ops.env" ]]; then
  # shellcheck source=/dev/null
  source "${SCRIPT_DIR}/deploy_ops.env"
fi

HARBOR_REGISTRY="${HARBOR_REGISTRY:-10.121.177.20:8082}"
HARBOR_PROJECT="${HARBOR_PROJECT:-mlops}"
HARBOR_USER="${HARBOR_USER:-admin}"
HARBOR_PASSWORD="${HARBOR_PASSWORD:-}"

IMAGE_TAG="${IMAGE_TAG:-$(date +%Y.%m.%d)}"
PUSH_LATEST=false

RUN_FRONTEND_ASSETS=false
RUN_BACKEND_IMAGE=false
RUN_FRONTEND_IMAGE=false
RUN_UPDATE_KUSTOMIZE=false
EXPLICIT_STEPS=false

usage() {
  cat <<'EOF'
构建 ops 平台前后端镜像并推送到 Harbor

用法:
  sh deploy_ops.sh [步骤选项...] [通用选项...]

步骤选项（可组合，按顺序执行）:
  --frontend-assets      编译前端静态资源（npm/yarn build）
  --backend-image        构建并推送后端镜像
  --frontend-image       构建并推送前端镜像
  --update-kustomize     更新 install/kubernetes/cube/overlays/kustomization.yml

未指定任何步骤选项时，默认执行:
  --frontend-assets --backend-image --frontend-image

通用选项:
  -t, --tag TAG          镜像版本号（默认: 当天日期 YYYY.MM.DD）
  --push-latest          推送镜像时额外打上 latest 标签
  -h, --help             显示帮助

示例:
  sh deploy_ops.sh --frontend-assets
  sh deploy_ops.sh --backend-image --frontend-image -t 2026.07.01
  sh deploy_ops.sh --update-kustomize -t 2026.07.01
  sh deploy_ops.sh --frontend-assets --backend-image --frontend-image --update-kustomize

配置: 复制 deploy_ops.env.example 为 deploy_ops.env 并填写 Harbor 凭证
EOF
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -t|--tag)
        IMAGE_TAG="$2"
        shift 2
        ;;
      --frontend-assets)
        RUN_FRONTEND_ASSETS=true
        EXPLICIT_STEPS=true
        shift
        ;;
      --backend-image)
        RUN_BACKEND_IMAGE=true
        EXPLICIT_STEPS=true
        shift
        ;;
      --frontend-image)
        RUN_FRONTEND_IMAGE=true
        EXPLICIT_STEPS=true
        shift
        ;;
      --update-kustomize)
        RUN_UPDATE_KUSTOMIZE=true
        EXPLICIT_STEPS=true
        shift
        ;;
      --push-latest)
        PUSH_LATEST=true
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        echo "未知参数: $1" >&2
        usage
        exit 1
        ;;
    esac
  done

  if [[ "${EXPLICIT_STEPS}" != "true" ]]; then
    RUN_FRONTEND_ASSETS=true
    RUN_BACKEND_IMAGE=true
    RUN_FRONTEND_IMAGE=true
  fi
}

BACKEND_IMAGE="${HARBOR_REGISTRY}/${HARBOR_PROJECT}/kubeflow-dashboard"
FRONTEND_IMAGE="${HARBOR_REGISTRY}/${HARBOR_PROJECT}/kubeflow-dashboard-frontend"

log() {
  echo "[$(date '+%H:%M:%S')] $*"
}

harbor_login() {
  if [[ -z "${HARBOR_PASSWORD}" ]]; then
    echo "错误: 未设置 HARBOR_PASSWORD，请在 deploy_ops.env 中配置或导出环境变量" >&2
    echo "  cp ${SCRIPT_DIR}/deploy_ops.env.example ${SCRIPT_DIR}/deploy_ops.env" >&2
    exit 1
  fi
  log "登录 Harbor: ${HARBOR_REGISTRY}"
  echo "${HARBOR_PASSWORD}" | docker login -u "${HARBOR_USER}" --password-stdin "${HARBOR_REGISTRY}"
}

build_frontend_assets() {
  log "编译前端静态资源..."
  cd "${PROJECT_ROOT}/myapp/frontend"
  npm install
  npm run build

  cd "${PROJECT_ROOT}/myapp/vision"
  npm install
  npm run build

  cd "${PROJECT_ROOT}/myapp/visionPlus"
  yarn
  npm run build

  cd "${PROJECT_ROOT}"
  log "前端静态资源编译完成"
}

build_and_push_backend() {
  log "构建后端镜像: ${BACKEND_IMAGE}:${IMAGE_TAG}"
  docker build --network=host \
    -t "${BACKEND_IMAGE}:${IMAGE_TAG}" \
    -f "${SCRIPT_DIR}/Dockerfile" \
    "${PROJECT_ROOT}"

  log "推送后端镜像..."
  docker push "${BACKEND_IMAGE}:${IMAGE_TAG}"

  if [[ "${PUSH_LATEST}" == "true" ]]; then
    docker tag "${BACKEND_IMAGE}:${IMAGE_TAG}" "${BACKEND_IMAGE}:latest"
    docker push "${BACKEND_IMAGE}:latest"
  fi
}

build_and_push_frontend() {
  log "构建前端镜像: ${FRONTEND_IMAGE}:${IMAGE_TAG}"
  docker build --network=host \
    -t "${FRONTEND_IMAGE}:${IMAGE_TAG}" \
    -f "${SCRIPT_DIR}/dockerFrontend/Dockerfile" \
    "${PROJECT_ROOT}"

  log "推送前端镜像..."
  docker push "${FRONTEND_IMAGE}:${IMAGE_TAG}"

  if [[ "${PUSH_LATEST}" == "true" ]]; then
    docker tag "${FRONTEND_IMAGE}:${IMAGE_TAG}" "${FRONTEND_IMAGE}:latest"
    docker push "${FRONTEND_IMAGE}:latest"
  fi
}

update_kustomization() {
  if [[ ! -f "${KUSTOMIZATION_FILE}" ]]; then
    echo "警告: 未找到 ${KUSTOMIZATION_FILE}，跳过 kustomize 更新" >&2
    return
  fi
  log "更新 kustomization.yml 镜像地址与标签..."
  # 先匹配较长的 frontend 镜像名，避免误替换
  sed -i \
    -e "s|newName: .*kubeflow-dashboard-frontend|newName: ${FRONTEND_IMAGE}|" \
    -e "s|newName: .*kubeflow-dashboard|newName: ${BACKEND_IMAGE}|" \
    -e "s/newTag: .*/newTag: ${IMAGE_TAG}/g" \
    "${KUSTOMIZATION_FILE}"
  log "已更新: ${KUSTOMIZATION_FILE}"
}

print_plan() {
  local steps=()
  [[ "${RUN_FRONTEND_ASSETS}" == "true" ]] && steps+=("frontend-assets")
  [[ "${RUN_BACKEND_IMAGE}" == "true" ]] && steps+=("backend-image")
  [[ "${RUN_FRONTEND_IMAGE}" == "true" ]] && steps+=("frontend-image")
  [[ "${RUN_UPDATE_KUSTOMIZE}" == "true" ]] && steps+=("update-kustomize")
  log "执行步骤: ${steps[*]}"
}

print_summary() {
  echo ""
  log "========== 完成 =========="
  [[ "${RUN_FRONTEND_ASSETS}" == "true" ]] && echo "  [√] 前端静态资源编译"
  [[ "${RUN_BACKEND_IMAGE}" == "true" ]] && echo "  [√] 后端镜像: ${BACKEND_IMAGE}:${IMAGE_TAG}"
  [[ "${RUN_FRONTEND_IMAGE}" == "true" ]] && echo "  [√] 前端镜像: ${FRONTEND_IMAGE}:${IMAGE_TAG}"
  [[ "${RUN_UPDATE_KUSTOMIZE}" == "true" ]] && echo "  [√] 已更新 kustomization.yml (tag: ${IMAGE_TAG})"

  if [[ "${RUN_BACKEND_IMAGE}" == "true" || "${RUN_FRONTEND_IMAGE}" == "true" || "${RUN_UPDATE_KUSTOMIZE}" == "true" ]]; then
    echo ""
    echo "下一步部署到 K8s:"
    echo "  cd install/kubernetes"
    if [[ "${RUN_UPDATE_KUSTOMIZE}" != "true" ]]; then
      echo "  # 或执行: sh install/docker/deploy_ops.sh --update-kustomize -t ${IMAGE_TAG}"
    fi
    echo "  kubectl apply -k cube/overlays"
    echo "  kubectl rollout status deployment/kubeflow-dashboard -n infra"
    echo "  kubectl rollout status deployment/kubeflow-dashboard-frontend -n infra"
  fi
}

main() {
  parse_args "$@"

  log "项目根目录: ${PROJECT_ROOT}"
  log "镜像版本: ${IMAGE_TAG}"
  log "Harbor: ${HARBOR_REGISTRY}/${HARBOR_PROJECT}"
  print_plan

  if [[ "${RUN_BACKEND_IMAGE}" == "true" || "${RUN_FRONTEND_IMAGE}" == "true" ]]; then
    harbor_login
  fi

  if [[ "${RUN_FRONTEND_ASSETS}" == "true" ]]; then
    build_frontend_assets
  fi

  if [[ "${RUN_BACKEND_IMAGE}" == "true" ]]; then
    build_and_push_backend
  fi

  if [[ "${RUN_FRONTEND_IMAGE}" == "true" ]]; then
    build_and_push_frontend
  fi

  if [[ "${RUN_UPDATE_KUSTOMIZE}" == "true" ]]; then
    update_kustomization
  fi

  print_summary
}

main "$@"
