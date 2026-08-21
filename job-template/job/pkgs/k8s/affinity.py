HOST_TOPOLOGY_KEY = "kubernetes.io/hostname"


def build_pod_anti_affinity(match_labels, replicas):
    """同一组分布式 Pod 跨 Node 打散。

    match_labels 必须与这些 Pod 自身 labels 一致。
    replicas > 1 时硬反亲和，禁止落到同一 Node；
    replicas <= 1 时软反亲和。
    """
    if not match_labels:
        raise ValueError("match_labels is required")
    term = {
        "topologyKey": HOST_TOPOLOGY_KEY,
        "labelSelector": {
            "matchLabels": dict(match_labels),
        },
    }
    if int(replicas) > 1:
        return {
            "requiredDuringSchedulingIgnoredDuringExecution": [term],
        }
    return {
        "preferredDuringSchedulingIgnoredDuringExecution": [
            {
                "weight": 100,
                "podAffinityTerm": term,
            }
        ],
    }
