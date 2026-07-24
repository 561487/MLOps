import argparse
import json
import tempfile
import unittest
from pathlib import Path

import launcher


class CurationTest(unittest.TestCase):
    def test_extract_and_classify(self):
        record = {
            "instruction": "Kubernetes Pod出现CrashLoopBackOff如何排查？",
            "input": "",
            "output": "先执行 kubectl describe pod 检查Events，再查看容器日志。",
            "id": "one",
        }
        instruction, response, source_id = launcher.extract_pair(record)
        category, score, _ = launcher.classify_domain(instruction, response)
        self.assertEqual(source_id, "one")
        self.assertEqual(category, "kubernetes")
        self.assertGreaterEqual(score, 3)

    def test_end_to_end_jsonl(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "raw"
            output = root / "output"
            work = root / "work"
            raw.mkdir()
            records = [
                {
                    "id": "k8s",
                    "instruction": "Kubernetes Pod CrashLoopBackOff如何排查？",
                    "output": (
                        "故障判断：容器反复退出。排查步骤：先运行"
                        " `kubectl describe pod` 查看Events，再使用"
                        " `kubectl logs --previous` 查看上次日志。"
                        "根据退出码修复配置，变更前备份并保留回滚版本。"
                    ),
                },
                {
                    "id": "nccl",
                    "instruction": "两台机器NCCL connection timeout且GPU利用率为0。",
                    "output": (
                        "故障判断：通信初始化失败。关键证据是GPU尚未进入计算。"
                        "排查步骤：检查MASTER_ADDR、端口、防火墙和NCCL_SOCKET_IFNAME；"
                        "使用 `nc -vz master 29500` 验证连通性。"
                        "修改网络配置前确认影响并准备回滚。"
                    ),
                },
                {
                    "id": "irrelevant",
                    "instruction": "请写一首春天的诗。",
                    "output": "春风吹过花园，花朵慢慢开放。这是一首普通的诗歌。",
                },
            ]
            with (raw / "data.jsonl").open("w", encoding="utf-8") as handle:
                for item in records:
                    handle.write(json.dumps(item, ensure_ascii=False) + "\n")

            args = argparse.Namespace(
                dataset_id="test/dataset",
                revision="main",
                raw_dir=str(raw),
                output_dir=str(output),
                work_dir=str(work),
                download_if_missing=False,
                overwrite=False,
                target_size=10,
                min_domain_score=3,
                min_quality_score=3,
                min_instruction_chars=10,
                max_instruction_chars=4000,
                min_response_chars=40,
                max_response_chars=24000,
                simhash_distance=3,
                batch_size=128,
                train_ratio=0.8,
                validation_ratio=0.1,
                seed=42,
                max_source_records=0,
                rejected_examples_per_reason=10,
                add_empty_think=True,
                keep_work_db=False,
                system_prompt=launcher.DEFAULT_SYSTEM_PROMPT,
            )
            self.assertEqual(launcher.curate(args), 0)
            self.assertTrue((output / "_SUCCESS").exists())
            report = json.loads(
                (output / "reports" / "statistics.json").read_text("utf-8")
            )
            self.assertEqual(report["total_selected"], 2)
            written = "".join(
                path.read_text("utf-8")
                for path in (output / "sft").glob("*.jsonl")
            )
            self.assertIn("<think>\\n\\n</think>", written)
            self.assertNotIn("春天的诗", written)


if __name__ == "__main__":
    unittest.main()
