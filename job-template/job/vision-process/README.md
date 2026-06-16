# 视觉处理模板

镜像：`ccr.ccs.tencentyun.com/cube-studio/vision-process:20260610`

统一入口：

```bash
python launcher.py --deal_type resize --input_path /mnt/admin/dataset/images --output_path /mnt/admin/pipeline/vision-output
```
```bash
python launcher.py \
  --deal_type augmentation \
  --input_path picture.jpg \
  --output_path ./test-output \
  --config '{"num_outputs": 3, "transforms": [{"type": "flip", "direction": "horizontal", "p": 0.5}, {"type": "rotate", "angle": 15, "p": 0.5}, {"type": "brightness_contrast", "brightness": 0.2, "contrast": 0.2, "p": 0.5}]}'
```
公共参数：

- `--deal_type`：处理类型，支持 `quality_assessment`、`blur`、`resize`、`normalize`、`cropping`、`equalize`、`cvtcolor`、`augmentation`。
- `--input_path`：输入图片目录或单张图片路径。
- `--output_path`：输出目录，处理后的图片写入 `output_path/images`。
- `--config`：JSON 字符串形式的处理参数。
- `--recursive`：是否递归处理子目录，默认 `true`。
- `--fail_fast`：遇到坏图是否立即失败，默认 `false`。

所有组件都会生成 `output_path/report.json`、`output_path/report.csv` 和 `/metric.json`。

## 质量评估

```bash
python launcher.py \
  --deal_type quality_assessment \
  --input_path /mnt/admin/dataset/images \
  --output_path /mnt/admin/pipeline/quality-report \
  --config '{"min_width": 32, "min_height": 32, "min_file_size": 1024, "blur_threshold": 100, "brightness_min": 10, "brightness_max": 245, "copy_valid": false}'
```

`copy_valid=true` 时，合格图片会复制到 `output_path/images` 供下游组件继续处理。

## 图片去噪

```bash
python launcher.py \
  --deal_type blur \
  --input_path /mnt/admin/dataset/images \
  --output_path /mnt/admin/pipeline/blur \
  --config '{"method": "gaussian", "kernel_size": 3}'
```

`method` 支持 `gaussian`、`median`、`bilateral`。

## 图片缩放

```bash
python launcher.py \
  --deal_type resize \
  --input_path /mnt/admin/dataset/images \
  --output_path /mnt/admin/pipeline/resize \
  --config '{"width": 640, "height": 640, "keep_ratio": true, "pad_color": "114,114,114"}'
```

## 图片标准化

```bash
python launcher.py \
  --deal_type normalize \
  --input_path /mnt/admin/dataset/images \
  --output_path /mnt/admin/pipeline/normalize \
  --config '{"mode": "minmax", "save_format": "image"}'
```

如需输出训练脚本直接读取的数组，可设置 `save_format=npy`。

## 图片裁剪

```bash
python launcher.py \
  --deal_type cropping \
  --input_path /mnt/admin/dataset/images \
  --output_path /mnt/admin/pipeline/crop \
  --config '{"mode": "center", "width": 512, "height": 512}'
```

`mode` 支持 `center`、`box`、`ratio`。`box` 使用 `x1,y1,x2,y2`。

## 图片均衡化

```bash
python launcher.py \
  --deal_type equalize \
  --input_path /mnt/admin/dataset/images \
  --output_path /mnt/admin/pipeline/equalize \
  --config '{"method": "clahe", "clip_limit": 2.0, "tile_grid_size": "8,8"}'
```

## 颜色空间转换

```bash
python launcher.py \
  --deal_type cvtcolor \
  --input_path /mnt/admin/dataset/images \
  --output_path /mnt/admin/pipeline/cvtcolor \
  --config '{"from_color": "BGR", "to_color": "GRAY"}'
```

支持 `RGB`、`BGR`、`GRAY`、`HSV`、`LAB` 间的常用转换。

## 图像增强

```bash
python launcher.py \
  --deal_type augmentation \
  --input_path /mnt/admin/dataset/images \
  --output_path /mnt/admin/pipeline/augmentation \
  --config '{"num_outputs": 3, "transforms": [{"type": "flip", "direction": "horizontal", "p": 0.5}, {"type": "rotate", "angle": 15, "p": 0.5}, {"type": "brightness_contrast", "brightness": 0.2, "contrast": 0.2, "p": 0.5}]}'
```

当前增强只处理图片本身，不同步检测框、分割 mask 等标注文件。

