docker login 10.121.177.20:8082
docker pull 10.121.177.20:8082/mlops/shell:v0.3.1 && docker tag 10.121.177.20:8082/mlops/shell:v0.3.1 rancher/shell:v0.3.1 &
docker pull 10.121.177.20:8082/mlops/hyperkube:v1.28.15-rancher1 && docker tag 10.121.177.20:8082/mlops/hyperkube:v1.28.15-rancher1 rancher/hyperkube:v1.28.15-rancher1 &
docker pull 10.121.177.20:8082/mlops/calico-cni:v3.27.4-rancher1 && docker tag 10.121.177.20:8082/mlops/calico-cni:v3.27.4-rancher1 rancher/calico-cni:v3.27.4-rancher1 &
docker pull 10.121.177.20:8082/mlops/mirrored-calico-node:v3.27.4 && docker tag 10.121.177.20:8082/mlops/mirrored-calico-node:v3.27.4 rancher/mirrored-calico-node:v3.27.4 &
docker pull 10.121.177.20:8082/mlops/mirrored-pause:3.7 && docker tag 10.121.177.20:8082/mlops/mirrored-pause:3.7 rancher/mirrored-pause:3.7 &
docker pull 10.121.177.20:8082/mlops/kube-api-auth:v0.2.4 && docker tag 10.121.177.20:8082/mlops/kube-api-auth:v0.2.4 rancher/kube-api-auth:v0.2.4 &
docker pull 10.121.177.20:8082/mlops/mirrored-coreos-etcd:v3.5.10 && docker tag 10.121.177.20:8082/mlops/mirrored-coreos-etcd:v3.5.10 rancher/mirrored-coreos-etcd:v3.5.10 &
docker pull 10.121.177.20:8082/mlops/rancher:v2.10.3 && docker tag 10.121.177.20:8082/mlops/rancher:v2.10.3 rancher/rancher:v2.10.3 &
docker pull 10.121.177.20:8082/mlops/rancher-webhook:v0.6.4 && docker tag 10.121.177.20:8082/mlops/rancher-webhook:v0.6.4 rancher/rancher-webhook:v0.6.4 &
docker pull 10.121.177.20:8082/mlops/rke-tools:v0.1.105 && docker tag 10.121.177.20:8082/mlops/rke-tools:v0.1.105 rancher/rke-tools:v0.1.105 &
docker pull 10.121.177.20:8082/mlops/mirrored-coredns-coredns:1.10.1 && docker tag 10.121.177.20:8082/mlops/mirrored-coredns-coredns:1.10.1 rancher/mirrored-coredns-coredns:1.10.1 &
docker pull 10.121.177.20:8082/mlops/rancher-agent:v2.10.3 && docker tag 10.121.177.20:8082/mlops/rancher-agent:v2.10.3 rancher/rancher-agent:v2.10.3 &
docker pull 10.121.177.20:8082/mlops/mirrored-flannel-flannel:v0.24.2 && docker tag 10.121.177.20:8082/mlops/mirrored-flannel-flannel:v0.24.2 rancher/mirrored-flannel-flannel:v0.24.2 &
docker pull 10.121.177.20:8082/mlops/mirrored-cluster-proportional-autoscaler:v1.8.9 && docker tag 10.121.177.20:8082/mlops/mirrored-cluster-proportional-autoscaler:v1.8.9 rancher/mirrored-cluster-proportional-autoscaler:v1.8.9 &
docker pull 10.121.177.20:8082/mlops/mirrored-calico-kube-controllers:v3.27.4 && docker tag 10.121.177.20:8082/mlops/mirrored-calico-kube-controllers:v3.27.4 rancher/mirrored-calico-kube-controllers:v3.27.4 &
docker pull 10.121.177.20:8082/mlops/mirrored-metrics-server:v0.7.0 && docker tag 10.121.177.20:8082/mlops/mirrored-metrics-server:v0.7.0 rancher/mirrored-metrics-server:v0.7.0 &
docker pull 10.121.177.20:8082/mlops/rke-tools:v0.1.109 && docker tag 10.121.177.20:8082/mlops/rke-tools:v0.1.109 rancher/rke-tools:v0.1.109 &

wait
