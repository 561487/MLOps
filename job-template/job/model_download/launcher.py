
import shutil, os
shutil.rmtree('/tmp/ms_cache', ignore_errors=True)
import sys
import argparse
import datetime
import json
import time
import uuid
import pysnooper
import re
import requests
import copy
import os
KFJ_CREATOR = os.getenv('KFJ_CREATOR', 'admin')
SENTINEL_FILE = '.model_downloaded'
host = os.getenv('HOST',os.getenv('KFJ_MODEL_REPO_API_URL','http://kubeflow-dashboard.infra')).strip('/')

def check_model_exists(save_path, model_name, model_version='', source_from=''):
    """
    检查 save_path 下是否已有同一模型的下载记录，返回 True 表示可以跳过下载。
    哨兵文件记录 model/version/from 信息，三者全部匹配才跳过。
    """
    sentinel_path = os.path.join(save_path, SENTINEL_FILE)
    if not os.path.isfile(sentinel_path):
        return False

    try:
        with open(sentinel_path, 'r') as f:
            content = f.read()
            stored_model = re.search(r'model=(\S*)', content)
            stored_version = re.search(r'version=(\S*)', content)
            stored_from = re.search(r'from=(\S*)', content)

            model_match = stored_model and stored_model.group(1) == str(model_name)
            version_match = stored_version and stored_version.group(1) == str(model_version)
            from_match = stored_from and stored_from.group(1) == str(source_from)

            if model_match and version_match and from_match:
                print(f'模型已存在，跳过下载: {save_path}')
                return True
            else:
                print(f'哨兵文件信息不匹配，重新下载')
                os.remove(sentinel_path)
                return False
    except Exception as e:
        print(f'读取哨兵文件失败: {e}，重新下载')
        if os.path.exists(sentinel_path):
            os.remove(sentinel_path)
        return False


def write_sentinel(save_path, model_name, model_version='', source_from=''):
    """写入哨兵文件，记录模型下载信息"""
    sentinel_path = os.path.join(save_path, SENTINEL_FILE)
    try:
        with open(sentinel_path, 'w') as f:
            f.write(f'downloaded at {datetime.datetime.now().isoformat()} model={model_name} version={model_version} from={source_from}\n')
        print(f'写入标记文件: {sentinel_path}')
    except Exception as e:
        print(f'写入标记文件失败: {e}')


# @pysnooper.snoop()
def download(**kwargs):
    # print(kwargs)
    headers = {
        'Content-Type': 'application/json',
        'Authorization': KFJ_CREATOR
    }
    model_path=""
    exist_model = {}
    # 从注册的模型中下载模型
    if kwargs['from']=='模型管理':
        url = host + "/training_model_modelview/api/?form_data=" + json.dumps({
            "filters": [
                {
                    "col": "name",
                    "opr": "eq",
                    "value": kwargs['model_name']
                },
                {
                    "col": "version",
                    "opr": "eq",
                    "value": kwargs['model_version']
                }
            ],
            "columns":['id','project','name','version','describe','path','framework','run_id','run_time','metrics','md5','api_type','pipeline_id']
        })

        # print(url)
        res = requests.get(url, headers=headers, allow_redirects=False)
        # print(res.content)
        if res.status_code == 200:
            exist_model = res.json().get('result', {}).get('data', [])
            if exist_model:
                exist_model = exist_model[0]
                print(exist_model)
                if exist_model['path']:
                    model_path = exist_model['path']
        else:
            print('访问平台获取模型失败')
            print(res.content)
            exit(1)

    elif kwargs['from']=='推理服务' or 'inference' in kwargs['from']:
        filters = [
            {
                "col": "model_name",
                "opr": "eq",
                "value": kwargs['model_name']
            },
            {
                "col": "model_version",
                "opr": "eq",
                "value": kwargs['model_version']
            }
        ]
        if kwargs['model_status']:
            filters.append({
                "col": "model_status",
                "opr": "eq",
                "value": kwargs['model_status']
            })


        url = host+"/inferenceservice_modelview/api/?form_data="+json.dumps({
            "filters":filters,
            "columns":  ['service_type','project', 'name', 'label','model_name', 'model_version', 'images', 'model_path', 'images', 'volume_mount','sidecar','working_dir', 'command', 'env', 'resource_memory',
                    'resource_cpu', 'resource_gpu', 'min_replicas', 'max_replicas', 'ports', 'inference_host_url','hpa','priority', 'canary', 'shadow', 'health','model_status','expand','metrics','deploy_history','host','inference_config']
        })

        # print(url)
        res = requests.get(url,headers=headers, allow_redirects=False)
        # print(res.content)
        if res.status_code==200:
            exist_service = res.json().get('result', {}).get('data', [])
            if exist_service:
                exist_service = exist_service[0]
                print(exist_service)
                if exist_service['model_path']:
                    model_path = exist_service['model_path']
        else:
            print('访问平台获取模型失败')
            print(res.content)
            exit(1)

    try:
        model_path=json.loads(model_path)
        model_path = model_path[kwargs['sub_model_name']]
    except Exception as e:
        pass
    if model_path:
        save_path = kwargs['save_path']
        os.makedirs(save_path, exist_ok=True)

        # 检查是否已下载过同一模型，是则跳过
        if check_model_exists(save_path, kwargs.get('model_name', ''),
                              kwargs.get('model_version', ''), kwargs['from']):
            exit(0)

        # 如果是在线地址，这下载
        if 'https://' in model_path or 'http://' in model_path:
            file_name = model_path.split("/")[-1]

            # 下载文件并保存到本地目录
            response = requests.get(model_path)
            with open(os.path.join(save_path, file_name), "wb") as file:
                file.write(response.content)
                file.close()
                print('模型保存至',save_path)

        elif not os.path.exists(model_path):
            print(f'{model_path}下不存在模型')
            exit(1)

        elif os.path.isdir(model_path):
            g = os.walk(model_path)
            for path, dir_list, file_list in g:
                for file_name in file_list:
                    one_file_path = os.path.join(path, file_name)
                    try:
                        des_path = os.path.join(save_path,file_name)
                        if os.path.exists(des_path):
                            os.remove(des_path)
                        shutil.copy2(one_file_path,des_path)
                        print('模型保存至',des_path)
                    except Exception as e:
                        print(e)
        else:
            shutil.copy2(model_path,save_path)
            print('模型保存至',save_path)

        # 同时将模型信息写入到存储中,比如计算指标
        if kwargs['from']=='模型管理':
            if exist_model:
                json.dump(exist_model,open(os.path.join(save_path,f'{exist_model["name"]}.{exist_model["version"]}.json'),mode='w'))

        # 写入哨兵文件，下次可跳过
        write_sentinel(save_path, kwargs.get('model_name', ''),
                       kwargs.get('model_version', ''), kwargs['from'])
    else:
        print('未发现模型')
        exit(1)

from subprocess import Popen, PIPE, STDOUT

def exe_command(command):
    """
    执行 shell 命令并实时打印输出
    :param command: shell 命令
    :return: process, exitcode
    """
    print(command)
    process = Popen(command, stdout=PIPE, stderr=STDOUT, shell=True)
    with process.stdout:
        for line in iter(process.stdout.readline, b''):
            print(line.decode().strip(),flush=True)
    exitcode = process.wait()
    return exitcode


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser("download model launcher")
    arg_parser.add_argument('--from', type=str, help="模型来源地", default='train_model')
    arg_parser.add_argument('--model_name', type=str, help="模型名", default='demo')
    arg_parser.add_argument('--sub_model_name', type=str, help="子模型名", default='')
    arg_parser.add_argument('--model_version', type=str, help="模型版本号",default='')
    arg_parser.add_argument('--model_status', type=str, help="模型状态", default='')
    arg_parser.add_argument('--save_path', type=str, help="下载目录", default='')

    args = arg_parser.parse_args()
    kwargs = args.__dict__
    # print("{} args: {}".format(__file__, args))
    if kwargs['from'] == 'modelscope' or kwargs['from'] == '魔塔':
        # 检查是否已下载过同一模型，是则跳过
        if check_model_exists(kwargs['save_path'], kwargs['model_name'],
                              kwargs.get('model_version', ''), kwargs['from']):
            exit(0)
        command = f'modelscope download --model {kwargs["model_name"]} --repo-type model --local_dir {kwargs["save_path"]} --cache_dir /tmp/ms_cache'
        exitcode = exe_command(command)
        if exitcode == 0:
            write_sentinel(kwargs['save_path'], kwargs['model_name'],
                           kwargs.get('model_version', ''), kwargs['from'])
        exit(exitcode)
    elif kwargs['from']=='huggingface':
        command = f'huggingface-cli download --repo-type model --resume-download {kwargs["model_name"]} --revision {kwargs["model_version"]} --local-dir {kwargs["save_path"]} --local-dir-use-symlinks False'
        exitcode = exe_command(command)
        exit(exitcode)
    elif kwargs['from'] == '模型管理' or kwargs['from']=='推理服务' or 'inference' in kwargs['from']:
        download(**kwargs)


