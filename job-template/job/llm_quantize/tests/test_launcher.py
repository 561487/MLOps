import argparse
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "launcher.py"
SPEC = importlib.util.spec_from_file_location("quant_launcher", MODULE_PATH)
quant = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(quant)


class FakeDataset:
    def __init__(self, rows):
        self.rows = rows
        self.column_names = list(rows[0]) if rows else []

    def __len__(self):
        return len(self.rows)

    def select(self, indexes):
        return FakeDataset([self.rows[index] for index in indexes])

    def __getitem__(self, column):
        return [row[column] for row in self.rows]


def make_args(**overrides):
    values = {
        "method": "gptq",
        "bits": 4,
        "group_size": 128,
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "output": "/tmp/quantized",
        "dataset": "wikitext2",
        "nsamples": 128,
        "force": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class ValidationTests(unittest.TestCase):
    def test_valid_method_combinations(self):
        for method, bits in (("gptq", 3), ("gptq", 8), ("awq", 4), ("bnb", 4), ("bnb", 8)):
            quant.validate_args(make_args(method=method, bits=bits))

    def test_awq_rejects_non_four_bit(self):
        with self.assertRaisesRegex(ValueError, "AutoAWQ"):
            quant.validate_args(make_args(method="awq", bits=8))

    def test_bnb_rejects_unsupported_bit_width(self):
        with self.assertRaisesRegex(ValueError, "bitsandbytes"):
            quant.validate_args(make_args(method="bnb", bits=3))

    def test_gptq_rejects_invalid_group_size(self):
        with self.assertRaisesRegex(ValueError, "group_size"):
            quant.validate_args(make_args(group_size=96))

    def test_sample_count_is_bounded(self):
        with self.assertRaisesRegex(ValueError, "nsamples"):
            quant.validate_args(make_args(nsamples=0))

    def test_boolean_parser(self):
        self.assertTrue(quant._parse_bool("true"))
        self.assertFalse(quant._parse_bool("0"))
        with self.assertRaises(argparse.ArgumentTypeError):
            quant._parse_bool("maybe")


class DatasetTests(unittest.TestCase):
    def test_extracts_supported_text_column_and_drops_blanks(self):
        dataset = FakeDataset([{"prompt": " first "}, {"prompt": ""}, {"prompt": "second"}])
        self.assertEqual(quant._extract_texts(dataset, 3), ["first", "second"])

    def test_uses_train_split(self):
        dataset = {"validation": FakeDataset([{"text": "validation"}]),
                   "train": FakeDataset([{"text": "train"}])}
        self.assertEqual(quant._extract_texts(dataset, 1), ["train"])

    def test_rejects_unknown_columns(self):
        with self.assertRaisesRegex(ValueError, "缺少文本列"):
            quant._extract_texts(FakeDataset([{"label": 1}]), 1)


class ManifestTests(unittest.TestCase):
    def test_manifest_is_written_and_success_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = {"status": "success", "method": "gptq"}
            path = quant._write_manifest(directory, manifest)
            self.assertEqual(json.loads(Path(path).read_text(encoding="utf-8")), manifest)
            self.assertEqual(quant._successful_manifest(directory), manifest)

    def test_nonempty_output_requires_force(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "partial.bin").write_bytes(b"partial")
            with self.assertRaises(FileExistsError):
                quant._ensure_output_ready(directory, False)
            self.assertIsNone(quant._ensure_output_ready(directory, True))

    def test_successful_output_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = {"status": "success", "method": "gptq"}
            quant._write_manifest(directory, manifest)
            self.assertEqual(quant._ensure_output_ready(directory, False), manifest)

class RunTests(unittest.TestCase):
    def test_success_then_idempotent_skip(self):
        with tempfile.TemporaryDirectory() as directory:
            args = make_args(method="bnb", bits=4, output=directory)
            original = quant.quantize_bnb

            def fake_quantize(model, output, bits):
                Path(output, "model.bin").write_bytes(b"model")
                return {"engine": "fake-bnb", "bits": bits, "output": output}

            try:
                quant.quantize_bnb = fake_quantize
                result = quant.run(args)
                self.assertEqual(result["status"], "success")
                self.assertEqual(result["artifacts"]["file_count"], 1)
                skipped = quant.run(args)
                self.assertEqual(skipped["status"], "skipped")
            finally:
                quant.quantize_bnb = original

if __name__ == "__main__":
    unittest.main()
