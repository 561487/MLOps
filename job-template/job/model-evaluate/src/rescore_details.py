"""Offline scoring of saved model outputs. Never loads a model or overwrites reports."""
import argparse
import json
from pathlib import Path
from custom_evaluator import score_prediction, _aggregate, SCORING_VERSION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='Existing custom_details.jsonl')
    parser.add_argument('--output_dir', required=True, help='New, nonexistent report directory')
    args = parser.parse_args()
    source = Path(args.input).resolve()
    rows = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines() if line.strip()]
    revised = []
    for row in rows:
        new = dict(row)
        new['previous_scoring'] = {k: row.get(k) for k in
                                   ('score', 'passed', 'parsed_answer', 'error_type', 'scoring_version')}
        if not row.get('inference_error'):
            new.pop('error_type', None)
            new.update(score_prediction(row, row['model_output']))
        revised.append(new)
    summary = _aggregate(revised, [])
    summary.update(source_details=str(source), scoring_version=SCORING_VERSION,
                   invalid_count=None, invalid_count_note='Original rejected records are not present in details',
                   changed_pass_count=sum(bool(a.get('passed')) != bool(b.get('passed'))
                                          for a, b in zip(rows, revised)))
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'rescored_details.jsonl').open('x', encoding='utf-8') as stream:
        for row in revised:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    with (output / 'rescored_summary.json').open('x', encoding='utf-8') as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
