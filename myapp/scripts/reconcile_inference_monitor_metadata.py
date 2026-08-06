"""
推理监控 metadata 审计/补注册工具。

扫描数据库中所有受监控推理服务（vllm/vllm-distributed/sglang），
对照 K8s 内部 Service 和 Prometheus Target，输出完整的监控接入状态。

默认 dry-run。只有显式传入 --apply 才允许 patch K8s Service。

用法：
    python -m myapp.scripts.reconcile_inference_monitor_metadata --dry-run
    python -m myapp.scripts.reconcile_inference_monitor_metadata --apply --service-id 66
"""
import sys
import os
import argparse
import json

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..'))


def reconcile(dry_run=True, target_service_id=None):
    """扫描并审计推理服务的监控 metadata。"""
    from myapp import app, db
    with app.app_context():
        from myapp.models.model_serving import InferenceService
        from myapp.services.inference_monitor_service import (
            resolve_engine_type,
            resolve_metrics_endpoint,
            build_inference_monitor_metadata,
            MONITORED_SERVICE_TYPES,
        )

        services = db.session.query(InferenceService).filter(
            InferenceService.service_type.in_(list(MONITORED_SERVICE_TYPES)),
            InferenceService.model_status != 'offline',
        )
        if target_service_id:
            services = services.filter(InferenceService.id == int(target_service_id))
        services = services.order_by(InferenceService.id.asc()).all()

        if not services:
            print("没有需要审计的受监控推理服务。")
            return 0

        needs_fix = 0
        results = []

        for svc in services:
            engine = resolve_engine_type(svc.service_type)
            ep = resolve_metrics_endpoint(svc)

            expected_a, expected_l = build_inference_monitor_metadata(
                svc.id, svc.name, svc.service_type,
                metrics_port=ep.get("port"), metrics_path=ep.get("path"),
            )

            result = {
                "db_id": svc.id,
                "service_name": svc.name,
                "model_name": svc.model_name,
                "engine": engine,
                "service_type": svc.service_type,
                "status": svc.model_status,
                "namespace": svc.namespace or "",
                "expected_labels": expected_l,
                "expected_annotations": expected_a,
                "current_labels": {},
                "current_annotations": {},
                "missing_labels": {},
                "missing_annotations": {},
                "needs_fix": False,
                "k8s_error": None,
            }

            # 读取 K8s 内部 Service
            try:
                from myapp.utils.py.py_k8s import K8s
                k8s_client = K8s(svc.project.cluster.get("KUBECONFIG", ""))
                k8s_svc = k8s_client.v1.read_namespaced_service(
                    name=svc.name, namespace=svc.namespace or svc.project.service_namespace,
                    _request_timeout=5,
                )
                result["current_labels"] = dict(k8s_svc.metadata.labels or {})
                result["current_annotations"] = dict(k8s_svc.metadata.annotations or {})

                # 计算缺失项
                result["missing_labels"] = {
                    k: v for k, v in expected_l.items()
                    if result["current_labels"].get(k) != v
                }
                result["missing_annotations"] = {
                    k: v for k, v in expected_a.items()
                    if result["current_annotations"].get(k) != v
                }
                result["needs_fix"] = bool(result["missing_labels"] or result["missing_annotations"])

            except Exception as e:
                result["k8s_error"] = str(e)
                result["needs_fix"] = True  # can't verify

            results.append(result)
            if result["needs_fix"]:
                needs_fix += 1

        # 输出报告
        print(f"\n{'='*70}")
        print(f"推理监控 Metadata 审计报告")
        print(f"模式: {'DRY-RUN' if dry_run else 'APPLY'}")
        print(f"扫描服务数: {len(results)}")
        print(f"需要修复: {needs_fix}")
        print(f"{'='*70}\n")

        for r in results:
            status = "需修复" if r["needs_fix"] else "正常"
            print(f"[{status}] DB ID={r['db_id']} name={r['service_name']} engine={r['engine']}")
            print(f"  model={r['model_name']} type={r['service_type']} status={r['status']}")

            if r["k8s_error"]:
                print(f"  K8s ERROR: {r['k8s_error']}")
                continue

            if r["missing_labels"]:
                print(f"  Missing labels: {json.dumps(r['missing_labels'], indent=4)}")
            if r["missing_annotations"]:
                print(f"  Missing annotations: {json.dumps(r['missing_annotations'], indent=4)}")

            if r["needs_fix"] and not dry_run:
                # Apply patch
                patch_body = {"metadata": {}}
                if r["missing_labels"]:
                    patch_body["metadata"]["labels"] = {
                        **r["current_labels"], **r["missing_labels"]
                    }
                if r["missing_annotations"]:
                    patch_body["metadata"]["annotations"] = {
                        **(r["current_annotations"] or {}), **r["missing_annotations"]
                    }
                try:
                    from myapp.utils.py.py_k8s import K8s
                    k8s_client = K8s(svc.project.cluster.get("KUBECONFIG", ""))
                    k8s_client.v1.patch_namespaced_service(
                        name=r["service_name"],
                        namespace=r["namespace"],
                        body=patch_body,
                    )
                    print(f"  ✅ PATCHED: {list(r['missing_labels'].keys())}")
                except Exception as e:
                    print(f"  ❌ PATCH FAILED: {e}")
            print()

        if needs_fix and dry_run:
            print("提示: 添加 --apply 参数以实际执行 K8s Service patch。")

        return needs_fix


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="推理监控 metadata 审计/补注册工具")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="仅审计，不执行 K8s patch（默认）")
    parser.add_argument("--apply", dest="dry_run", action="store_false",
                        help="执行 K8s Service patch")
    parser.add_argument("--service-id", type=int, default=None,
                        help="仅审计指定 service_id")
    args = parser.parse_args()

    needs_fix = reconcile(dry_run=args.dry_run, target_service_id=args.service_id)
    sys.exit(min(needs_fix, 1))
