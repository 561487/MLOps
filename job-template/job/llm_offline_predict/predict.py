# ============================================================
# LLaMA 离线推理
# 支持两种推理后端：
#   1. --backend transformers  （默认，适合小模型）
#   2. --backend vllm          （适合大模型，省显存，加速）
#
# 参数（通过 command 传入）：
#   --model_path      模型路径
#   --input_file      输入文件，每行一条待推理文本
#   --output_file     输出文件（JSONL 格式）
#   --max_new_tokens  最大生成 token 数（默认 512）
#   --temperature     采样温度，控制随机性（默认 0.3）
#   --top_k           Top-K 采样，只保留概率最高的 K 个 token（默认 10）
#   --top_p           Top-P (nucleus) 采样，累积概率阈值（默认 0.7）
#   --backend         推理后端：transformers / vllm（默认 transformers）
# ============================================================

import os, sys, json, argparse, datetime, time, traceback, subprocess

# ── 关键修复：Python 3.9 + vLLM 0.9.x V1 引擎 multiprocessing spawn 不兼容 ──
# V1 引擎需要 spawn 子进程做 engine core，Python 3.9 的 spawn + CUDA 已初始化
# 会导致 "Engine core initialization failed. Failed core proc(s): {}"
# 强制使用 V0 引擎（稳定，无 multiprocessing 问题）
os.environ.setdefault('VLLM_USE_V1', '0')

def log(msg):
    """带时间戳的日志"""
    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{ts}] {msg}', flush=True)


def dump_diagnostics():
    """启动诊断：输出环境信息，方便排查调度/GPU/内存问题"""
    log('========== STARTUP DIAGNOSTICS ==========')
    log(f'[DIAG] Hostname: {os.uname().nodename}')
    log(f'[DIAG] Python: {sys.version}')
    log(f'[DIAG] PID: {os.getpid()}')
    log(f'[DIAG] CWD: {os.getcwd()}')

    # GPU 信息
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.total,memory.free,memory.used,utilization.gpu,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=15)
        if result.returncode == 0 and result.stdout.strip():
            for line in result.stdout.strip().split('\n'):
                log(f'[DIAG] GPU: {line}')
        else:
            log('[DIAG] GPU: nvidia-smi returned no data')
    except FileNotFoundError:
        log('[DIAG] GPU: nvidia-smi not found (no GPU driver?)')
    except Exception as e:
        log(f'[DIAG] GPU: nvidia-smi error: {e}')

    # CUDA 可见设备
    log(f'[DIAG] CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES", "(not set)")}')
    log(f'[DIAG] NVIDIA_VISIBLE_DEVICES={os.environ.get("NVIDIA_VISIBLE_DEVICES", "(not set)")}')

    # 内存
    try:
        result = subprocess.run(['free', '-h'], capture_output=True, text=True, timeout=5)
        for line in result.stdout.strip().split('\n'):
            log(f'[DIAG] MEM: {line}')
    except Exception as e:
        log(f'[DIAG] MEM: free error: {e}')

    # 磁盘（模型和数据路径）
    for p in ['/mnt/storage/models-share-volume', '/mnt/admin']:
        try:
            if os.path.exists(p):
                stat = os.statvfs(p)
                free_gb = (stat.f_frsize * stat.f_bavail) / (1024 ** 3)
                total_gb = (stat.f_frsize * stat.f_blocks) / (1024 ** 3)
                log(f'[DIAG] DISK {p}: free={free_gb:.1f}G / total={total_gb:.1f}G')
            else:
                log(f'[DIAG] DISK {p}: not mounted / not exist')
        except Exception as e:
            log(f'[DIAG] DISK {p}: error: {e}')

    # 关键环境变量
    for v in ['MODEL_PATH', 'INPUT_FILE', 'OUTPUT_FILE', 'MAX_NEW_TOKENS', 'BACKEND',
              'TEMPERATURE', 'TOP_K', 'TOP_P',
              'RABBIT_HOST', 'VC_TASK_INDEX', 'VC_TASK_NUM', 'VLLM_USE_V1',
              'KFJ_TASK_RESOURCE_GPU', 'KFJ_TASK_RESOURCE_CPU', 'KFJ_TASK_RESOURCE_MEMORY']:
        log(f'[DIAG] ENV {v}={os.environ.get(v, "(not set)")}')

    # 关键包版本
    for pkg in ['torch', 'vllm', 'transformers', 'pika', 'numpy']:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, '__version__', 'unknown')
            log(f'[DIAG] PKG {pkg}=={ver}')
        except Exception:
            log(f'[DIAG] PKG {pkg}: NOT INSTALLED')

    # PyTorch CUDA 检测
    try:
        import torch
        log(f'[DIAG] torch.cuda.is_available()={torch.cuda.is_available()}')
        if torch.cuda.is_available():
            log(f'[DIAG] torch.cuda.device_count()={torch.cuda.device_count()}')
            for i in range(torch.cuda.device_count()):
                log(f'[DIAG] torch.cuda device[{i}]: {torch.cuda.get_device_name(i)}')
                try:
                    prop = torch.cuda.get_device_properties(i)
                    # PyTorch 2.x: total_memory (bytes)
                    mem_total = getattr(prop, 'total_memory', None)
                    if mem_total is None:
                        mem_total = getattr(prop, 'total_mem', 0)  # fallback
                    log(f'[DIAG] torch.cuda device[{i}] total_mem={mem_total/(1024**3):.1f}G')
                except Exception as e2:
                    log(f'[DIAG] torch.cuda device[{i}] mem query error: {e2}')
    except Exception as e:
        log(f'[DIAG] torch.cuda check error: {e}')

    log('========== END DIAGNOSTICS ==========')


from predict_model import Offline_Predict


class LLaMA_Offline_Predict(Offline_Predict):

    def __init__(self):
        init_start = time.time()
        try:
            # 判断角色：生产者不需要加载模型（只用 ROLE 判断，不用 VC_TASK_INDEX）
            role = os.environ.get('ROLE', '')
            self._is_producer = (role == 'master')

            # 启动诊断
            dump_diagnostics()

            # 优先从环境变量读取（worker Pod 由 launcher-rabbitmq.py 注入）
            # CLI 参数可以覆盖环境变量
            parser = argparse.ArgumentParser(description='LLaMA 离线推理')
            parser.add_argument('--model_path',
                                default=os.environ.get('MODEL_PATH', '/mnt/storage/models-share-volume/output/lora/qwen3-0.6b-lora'))
            parser.add_argument('--input_file',
                                default=os.environ.get('INPUT_FILE', '/mnt/storage/models-share-volume/data/input.txt'))
            parser.add_argument('--output_file',
                                default=os.environ.get('OUTPUT_FILE', '/mnt/storage/models-share-volume/output/inference/result.jsonl'))
            parser.add_argument('--max_new_tokens', type=int,
                                default=int(os.environ.get('MAX_NEW_TOKENS', '512')))
            parser.add_argument('--temperature', type=float,
                                default=float(os.environ.get('TEMPERATURE', '0.3')),
                                help='采样温度，控制随机性，越低越确定性（默认 0.3）')
            parser.add_argument('--top_k', type=int,
                                default=int(os.environ.get('TOP_K', '10')),
                                help='Top-K 采样，只保留概率最高的 K 个 token（默认 10）')
            parser.add_argument('--top_p', type=float,
                                default=float(os.environ.get('TOP_P', '0.7')),
                                help='Top-P (nucleus) 采样，累积概率阈值（默认 0.7）')
            parser.add_argument('--backend', default=os.environ.get('BACKEND', 'transformers'),
                                choices=['transformers', 'vllm'],
                                help='推理后端：transformers（默认）或 vllm')
            args = parser.parse_args()
            self.args = args

            log(f'[LLaMA_Offline_Predict] backend={args.backend}')
            log(f'[LLaMA_Offline_Predict] model_path={args.model_path}')
            log(f'[LLaMA_Offline_Predict] input_file={args.input_file}')
            log(f'[LLaMA_Offline_Predict] output_file={args.output_file}')
            log(f'[LLaMA_Offline_Predict] max_new_tokens={args.max_new_tokens}')
            log(f'[LLaMA_Offline_Predict] temperature={args.temperature}, top_k={args.top_k}, top_p={args.top_p}')
            vc_task_index = os.environ.get('VC_TASK_INDEX', 'N/A')
            local_rank = os.environ.get('LOCAL_RANK', 'N/A')
            log(f'[LLaMA_Offline_Predict] is_producer={self._is_producer} (VC_TASK_INDEX={vc_task_index}, LOCAL_RANK={local_rank})')

            os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

            if self._is_producer:
                # 生产者只需要读文件和发队列，不需要模型/GPU
                log('[LLaMA_Offline_Predict] Role: PRODUCER - skipping model loading (no GPU needed)')
                # 清空输出文件，避免上次运行残留数据混入本次结果
                with open(args.output_file, 'w', encoding='utf-8') as f:
                    f.write('')
                log(f'[LLaMA_Offline_Predict] Output file cleared: {args.output_file}')
                self.model = None
                self.tokenizer = None
                self.vllm_llm = None
                self.vllm_sampling = None
            elif args.backend == 'vllm':
                self._init_vllm()
            else:
                self._init_transformers()

            elapsed = time.time() - init_start
            log(f'[LLaMA_Offline_Predict] init OK, elapsed={elapsed:.1f}s')
        except Exception as e:
            log(f'[LLaMA_Offline_Predict] ERROR in __init__: {e}')
            log(f'[LLaMA_Offline_Predict] Full traceback:\n{traceback.format_exc()}')
            sys.exit(1)

    # ── transformers 后端 ──────────────────────────────────
    def _init_transformers(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        log(f'[LLaMA] Loading model (transformers) from {self.args.model_path}')
        log(f'[LLaMA] model_path exists: {os.path.exists(self.args.model_path)}')
        if not os.path.exists(self.args.model_path):
            log(f'[LLaMA] ERROR: model path not found: {self.args.model_path}')
            sys.exit(1)
        log(f'[LLaMA] model dir contents: {os.listdir(self.args.model_path)}')

        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path)
        log(f'[LLaMA] Tokenizer loaded in {time.time()-t0:.1f}s')

        t0 = time.time()
        self.model = AutoModelForCausalLM.from_pretrained(
            self.args.model_path,
            torch_dtype='auto',
            device_map='auto'
        )
        log(f'[LLaMA] Model loaded in {time.time()-t0:.1f}s')
        self.model.eval()
        log('[LLaMA] Model loaded (transformers)')

    def _predict_transformers(self, text):
        import torch
        t0 = time.time()
        inputs = self.tokenizer(text, return_tensors='pt').to(self.model.device)
        log(f'[LLaMA] Tokenized in {time.time()-t0:.1f}s, input_len={inputs["input_ids"].shape[1]}')
        t0 = time.time()
        # 构建 generate 参数（debug 信息）
        gen_kwargs = {
            'max_new_tokens': self.args.max_new_tokens,
            'do_sample': True,
            'temperature': self.args.temperature,
            'top_k': self.args.top_k,
            'top_p': self.args.top_p,
            'repetition_penalty': 1.15,
            'no_repeat_ngram_size': 4,
        }
        log(f'[LLaMA] generate kwargs: max_new_tokens={gen_kwargs["max_new_tokens"]}, '
            f'temperature={gen_kwargs["temperature"]}, top_k={gen_kwargs["top_k"]}, '
            f'top_p={gen_kwargs["top_p"]}, do_sample={gen_kwargs["do_sample"]}')
        outputs = self.model.generate(**inputs, **gen_kwargs)
        gen_time = time.time() - t0
        out_len = outputs.shape[1] - inputs['input_ids'].shape[1]
        log(f'[LLaMA] Generated in {gen_time:.1f}s, output_len={out_len}, speed={out_len/gen_time:.1f} tok/s')
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    # ── vLLM 后端 ─────────────────────────────────────────
    def _init_vllm(self):
        log('[VLLM] ====== 使用 vLLM 推理后端 (V0 引擎) ======')
        log(f'[VLLM] VLLM_USE_V1={os.environ.get("VLLM_USE_V1", "not set")}')
        from vllm import LLM, SamplingParams
        log(f'[VLLM] 加载模型路径: {self.args.model_path}')
        log(f'[VLLM] model_path exists: {os.path.exists(self.args.model_path)}')
        if not os.path.exists(self.args.model_path):
            log(f'[VLLM] 错误: 模型路径不存在: {self.args.model_path}')
            sys.exit(1)
        log(f'[VLLM] 模型目录内容: {os.listdir(self.args.model_path)}')
        log(f'[VLLM] 正在初始化 vLLM 引擎 (max_new_tokens={self.args.max_new_tokens}, '
            f'temperature={self.args.temperature}, top_k={self.args.top_k}, top_p={self.args.top_p})...')

        t0 = time.time()
        try:
            self.vllm_llm = LLM(model=self.args.model_path)
            log(f'[VLLM] vLLM engine initialized in {time.time()-t0:.1f}s')
        except Exception as e:
            log(f'[VLLM] ERROR initializing vLLM engine: {e}')
            log(f'[VLLM] Full traceback:\n{traceback.format_exc()}')
            sys.exit(1)

        self.vllm_sampling = SamplingParams(
            max_tokens=self.args.max_new_tokens,
            temperature=self.args.temperature,
            top_k=self.args.top_k,
            top_p=self.args.top_p,
        )
        log(f'[VLLM] 采样参数: max_tokens={self.args.max_new_tokens}, '
            f'temperature={self.args.temperature}, top_k={self.args.top_k}, top_p={self.args.top_p}')
        log('[VLLM] ====== vLLM 模型加载完成 ======')

    def _predict_vllm(self, text):
        log(f'[VLLM] 使用 vLLM 推理: "{text[:50]}..."')
        t0 = time.time()
        outputs = self.vllm_llm.generate([text], self.vllm_sampling)
        gen_time = time.time() - t0
        result = outputs[0].outputs[0].text
        log(f'[VLLM] vLLM 推理完成, output_len={len(result)}, time={gen_time:.1f}s, speed={len(result)/gen_time:.1f} tok/s')
        return result

    # ── 统一接口 ──────────────────────────────────────────
    def datasource(self):
        """生产者：读取输入文件"""
        log(f'[LLaMA] Reading input file: {self.args.input_file}')
        if not os.path.exists(self.args.input_file):
            log(f'[LLaMA] ERROR: input file not found: {self.args.input_file}')
            return []
        with open(self.args.input_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        log(f'[LLaMA] Total input lines: {len(lines)}')
        return lines

    def _write_result(self, record):
        """线程安全写入结果文件（fcntl 文件锁，防止多 worker 并发写错乱）"""
        import fcntl
        with open(self.args.output_file, 'a', encoding='utf-8') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                f.flush()
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def predict(self, text):
        """消费者：推理单条文本"""
        item_start = time.time()
        try:
            log(f'[LLaMA] Predicting line: {text[:80]}...')
            if self.args.backend == 'vllm':
                result = self._predict_vllm(text)
            else:
                result = self._predict_transformers(text)

            self._write_result({'input': text, 'output': result})
            elapsed = time.time() - item_start
            log(f'[LLaMA] Predict OK, elapsed={elapsed:.1f}s, output -> {self.args.output_file}')
            return result
        except Exception as e:
            elapsed = time.time() - item_start
            log(f'[LLaMA] ERROR in predict (elapsed={elapsed:.1f}s): {e}')
            log(f'[LLaMA] Full traceback:\n{traceback.format_exc()}')
            try:
                self._write_result({'input': text, 'error': str(e)})
            except Exception as we:
                log(f'[LLaMA] ERROR writing error result: {we}')
            return str(e)


if __name__ == '__main__':
    try:
        log('[LLaMA] ====== Starting offline predict ======')
        predictor = LLaMA_Offline_Predict()
        t0 = time.time()
        predictor.run()
        log(f'[LLaMA] ====== Run completed, total time={time.time()-t0:.1f}s ======')
    except Exception as e:
        log(f'[LLaMA] Fatal error: {e}')
        log(f'[LLaMA] Full traceback:\n{traceback.format_exc()}')
        sys.exit(1)
