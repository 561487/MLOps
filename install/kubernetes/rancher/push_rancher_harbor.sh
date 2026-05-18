docker login 10.121.177.20:8082
docker pull rancher/shell:v0.3.1 && docker tag rancher/shell:v0.3.1 10.121.177.20:8082/rancher/shell:v0.3.1 && docker push 10.121.177.20:8082/rancher/shell:v0.3.1 &
docker pull rancher/hyperkube:v1.28.15-rancher1 && docker tag rancher/hyperkube:v1.28.15-rancher1 10.121.177.20:8082/rancher/hyperkube:v1.28.15-rancher1 && docker push 10.121.177.20:8082/rancher/hyperkube:v1.28.15-rancher1 &
docker pull rancher/calico-cni:v3.27.4-rancher1 && docker tag rancher/calico-cni:v3.27.4-rancher1 10.121.177.20:8082/rancher/calico-cni:v3.27.4-rancher1 && docker push 10.121.177.20:8082/rancher/calico-cni:v3.27.4-rancher1 &
docker pull rancher/mirrored-calico-node:v3.27.4 && docker tag rancher/mirrored-calico-node:v3.27.4 10.121.177.20:8082/rancher/mirrored-calico-node:v3.27.4 && docker push 10.121.177.20:8082/rancher/mirrored-calico-node:v3.27.4 &
docker pull rancher/mirrored-pause:3.7 && docker tag rancher/mirrored-pause:3.7 10.121.177.20:8082/rancher/mirrored-pause:3.7 && docker push 10.121.177.20:8082/rancher/mirrored-pause:3.7 &
docker pull rancher/kube-api-auth:v0.2.4 && docker tag rancher/kube-api-auth:v0.2.4 10.121.177.20:8082/rancher/kube-api-auth:v0.2.4 && docker push 10.121.177.20:8082/rancher/kube-api-auth:v0.2.4 &
docker pull rancher/mirrored-coreos-etcd:v3.5.10 && docker tag rancher/mirrored-coreos-etcd:v3.5.10 10.121.177.20:8082/rancher/mirrored-coreos-etcd:v3.5.10 && docker push 10.121.177.20:8082/rancher/mirrored-coreos-etcd:v3.5.10 &
docker pull rancher/rancher:v2.10.3 && docker tag rancher/rancher:v2.10.3 10.121.177.20:8082/rancher/rancher:v2.10.3 && docker push 10.121.177.20:8082/rancher/rancher:v2.10.3 &
docker pull rancher/rancher-webhook:v0.6.4 && docker tag rancher/rancher-webhook:v0.6.4 10.121.177.20:8082/rancher/rancher-webhook:v0.6.4 && docker push 10.121.177.20:8082/rancher/rancher-webhook:v0.6.4 &
docker pull rancher/rke-tools:v0.1.105 && docker tag rancher/rke-tools:v0.1.105 10.121.177.20:8082/rancher/rke-tools:v0.1.105 && docker push 10.121.177.20:8082/rancher/rke-tools:v0.1.105 &
docker pull rancher/mirrored-coredns-coredns:1.10.1 && docker tag rancher/mirrored-coredns-coredns:1.10.1 10.121.177.20:8082/rancher/mirrored-coredns-coredns:1.10.1 && docker push 10.121.177.20:8082/rancher/mirrored-coredns-coredns:1.10.1 &
docker pull rancher/rancher-agent:v2.10.3 && docker tag rancher/rancher-agent:v2.10.3 10.121.177.20:8082/rancher/rancher-agent:v2.10.3 && docker push 10.121.177.20:8082/rancher/rancher-agent:v2.10.3 &
docker pull rancher/mirrored-flannel-flannel:v0.24.2 && docker tag rancher/mirrored-flannel-flannel:v0.24.2 10.121.177.20:8082/rancher/mirrored-flannel-flannel:v0.24.2 && docker push 10.121.177.20:8082/rancher/mirrored-flannel-flannel:v0.24.2 &
docker pull rancher/mirrored-cluster-proportional-autoscaler:v1.8.9 && docker tag rancher/mirrored-cluster-proportional-autoscaler:v1.8.9 10.121.177.20:8082/rancher/mirrored-cluster-proportional-autoscaler:v1.8.9 && docker push 10.121.177.20:8082/rancher/mirrored-cluster-proportional-autoscaler:v1.8.9 &
docker pull rancher/mirrored-calico-kube-controllers:v3.27.4 && docker tag rancher/mirrored-calico-kube-controllers:v3.27.4 10.121.177.20:8082/rancher/mirrored-calico-kube-controllers:v3.27.4 && docker push 10.121.177.20:8082/rancher/mirrored-calico-kube-controllers:v3.27.4 &
docker pull rancher/mirrored-metrics-server:v0.7.0 && docker tag rancher/mirrored-metrics-server:v0.7.0 10.121.177.20:8082/rancher/mirrored-metrics-server:v0.7.0 && docker push 10.121.177.20:8082/rancher/mirrored-metrics-server:v0.7.0 &
docker pull rancher/rke-tools:v0.1.109 && docker tag rancher/rke-tools:v0.1.109 10.121.177.20:8082/rancher/rke-tools:v0.1.109 && docker push 10.121.177.20:8082/rancher/rke-tools:v0.1.109 &

wait
