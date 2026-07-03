# audio-process

This directory implements three audio processing functions with one codebase and one Docker image:

- `quality_assessment`: audio quality assessment
- `clean`: audio cleaning
- `augment`: audio augmentation

Three platform nodes can share this image and set different `--process_type` values.

## Directory

```text
job-template/job/audio-process/
  launcher.py
  requirements.txt
  Dockerfile
  build.sh
  audio_process/
    common/
    processors/
```

## 1. Quality assessment

```bash
python3 launcher.py \
  --process_type quality_assessment \
  --input_manifest /mnt/storage/models-storage/datasets/audio/manifest.csv \
  --audio_path_col audio_path \
  --target_sample_rate 16000 \
  --min_duration 1.0 \
  --max_duration 30.0 \
  --output_report_path /mnt/storage/models-storage/output/audio-quality/report.json \
  --output_summary_csv /mnt/storage/models-storage/output/audio-quality/summary.csv \
  --valid_manifest_path /mnt/storage/models-storage/output/audio-quality/valid_manifest.csv \
  --invalid_manifest_path /mnt/storage/models-storage/output/audio-quality/invalid_manifest.csv
```

## 2. Clean

```bash
python3 launcher.py \
  --process_type clean \
  --input_manifest /mnt/storage/models-storage/output/audio-quality/valid_manifest.csv \
  --audio_path_col audio_path \
  --target_sample_rate 16000 \
  --target_channels 1 \
  --output_format wav \
  --normalize_volume true \
  --target_dbfs -20 \
  --trim_silence true \
  --output_audio_dir /mnt/storage/models-storage/output/audio-clean/audio \
  --output_manifest_path /mnt/storage/models-storage/output/audio-clean/cleaned_manifest.csv \
  --output_report_path /mnt/storage/models-storage/output/audio-clean/report.json
```

## 3. Quality assessment after clean

```bash
python3 launcher.py \
  --process_type quality_assessment \
  --input_manifest /mnt/storage/models-storage/output/audio-clean/cleaned_manifest.csv \
  --output_report_path /mnt/storage/models-storage/output/audio-quality-after-clean/report.json \
  --output_summary_csv /mnt/storage/models-storage/output/audio-quality-after-clean/summary.csv \
  --valid_manifest_path /mnt/storage/models-storage/output/audio-quality-after-clean/valid_manifest.csv \
  --invalid_manifest_path /mnt/storage/models-storage/output/audio-quality-after-clean/invalid_manifest.csv
```

## 4. Augment

```bash
python3 launcher.py \
  --process_type augment \
  --input_manifest /mnt/storage/models-storage/output/audio-quality-after-clean/valid_manifest.csv \
  --audio_path_col audio_path \
  --split_col split \
  --augment_split train \
  --augment_times 2 \
  --output_audio_dir /mnt/storage/models-storage/output/audio-augment/audio \
  --output_manifest_path /mnt/storage/models-storage/output/audio-augment/augmented_manifest.csv \
  --output_report_path /mnt/storage/models-storage/output/audio-augment/report.json
```

## Platform mapping

Use one image:

```text
10.121.177.20:8082/mlops/audio-process:20260630
```

Map three nodes by parameter:

```text
audio-quality-assessment -> --process_type quality_assessment
audio-clean              -> --process_type clean
audio-augment            -> --process_type augment
```
