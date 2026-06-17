#!/usr/bin/env python3
"""
上传 job template 脚本到 PVC 共享存储
用法: python scripts/upload_to_pvc.py

自动寻找 kubeconfig，适配不同环境。
"""
import os, sys, time, base64, glob

# 自动寻找 kubeconfig 文件
KUBECONFIG_PATHS = [
    '/home/myapp/kubeconfig',
    os.path.join(os.path.dirname(__file__), '..', 'install', 'docker', 'kubeconfig'),
    '/root/kubeconfig',
    os.environ.get('KUBECONFIG', ''),
]

KC = None
for p in KUBECONFIG_PATHS:
    if os.path.isfile(p):
        KC = p
        break
    elif os.path.isdir(p):
        files = [f for f in os.listdir(p) if 'kubeconfig' in f or 'config' in f]
        if files:
            KC = os.path.join(p, files[0])
            break

if not KC or not os.path.isfile(KC):
    print('Error: cannot find kubeconfig')
    print('Searched:', KUBECONFIG_PATHS)
    sys.exit(1)

print(f'Using kubeconfig: {KC}')

# 脚本映射：目标名 → 源文件路径（相对项目根目录）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = {
    'llm_quantize.py': os.path.join(PROJECT_ROOT, 'job-template/job/llm_quantize/launcher.py'),
    'model_prune.py': os.path.join(PROJECT_ROOT, 'job-template/job/model_prune/launcher.py'),
}

sys.path.insert(0, os.path.join(PROJECT_ROOT, 'myapp'))
os.environ.setdefault('STAGE', 'DEV')
os.environ.setdefault('ENVIRONMENT', 'DEV')

from myapp import app
with app.app_context():
    from myapp.utils.py.py_k8s import K8s
    k8s = K8s(KC)
    api = k8s.CoreV1Api

    NS = 'pipeline'
    PVC = 'kubeflow-user-workspace'

    # 读取并编码脚本文件
    contents = {}
    for dest, src in SCRIPTS.items():
        if os.path.isfile(src):
            with open(src, 'rb') as f:
                contents[dest] = base64.b64encode(f.read()).decode()
            print(f'  Loaded {dest} ({len(contents[dest])} bytes base64)')
        else:
            print(f'  SKIP: {src} not found')

    if not contents:
        print('No files to upload')
        sys.exit(1)

    # 创建 Pod 并写入文件
    pod_name = 'script-uploader-' + str(int(time.time()))
    cmds = 'set -e\nmkdir -p /mnt/scripts/\n'
    for dest in contents:
        n = 'F_' + dest.upper().replace('.', '_')
        cmds += f'echo ${n} | base64 -d > /mnt/scripts/{dest}\n'
    cmds += 'echo "--- Files on PVC ---"\nls -la /mnt/scripts/\n'

    envs = [{'name': 'F_' + d.upper().replace('.', '_'), 'value': v} for d, v in contents.items()]

    body = {
        'apiVersion': 'v1', 'kind': 'Pod',
        'metadata': {'name': pod_name, 'namespace': NS},
        'spec': {
            'containers': [{
                'name': 'copier', 'image': 'busybox',
                'command': ['sh', '-c', cmds],
                'env': envs,
                'volumeMounts': [{'name': 'workspace', 'mountPath': '/mnt'}]
            }],
            'volumes': [{'name': 'workspace', 'persistentVolumeClaim': {'claimName': PVC}}],
            'restartPolicy': 'Never'
        }
    }

    print(f'Creating pod {pod_name}...')
    api.create_namespaced_pod(NS, body)

    for i in range(30):
        time.sleep(2)
        p = api.read_namespaced_pod(pod_name, NS)
        if p.status.phase in ('Succeeded', 'Failed'):
            break
        print(f'  Waiting... ({p.status.phase})')

    print('Pod logs:')
    print(api.read_namespaced_pod_log(pod_name, NS))

    api.delete_namespaced_pod(pod_name, NS)
    print('Done')
