import copy


def build_master_worker_replica_specs(master_spec, num_workers):
    """组装 Master/Worker replicaSpecs。

    num_workers 为总节点数（含 1 个 Master）。
    单机（num_workers <= 1）时不写入 Worker key：
    - 避免 train-operator 将未指定 replicas 默认成 1
    - 避免写 replicas:0 时先建后删 Worker、短暂多申请一份资源
    """
    replica_specs = {"Master": master_spec}
    worker_replicas = int(num_workers) - 1
    if worker_replicas > 0:
        worker_spec = copy.deepcopy(master_spec)
        worker_spec["replicas"] = worker_replicas
        replica_specs["Worker"] = worker_spec
    return replica_specs
