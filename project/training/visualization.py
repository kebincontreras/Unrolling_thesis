from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch


def to_uint8_image(tensor: torch.Tensor) -> Image.Image:
    array = tensor.detach().cpu().numpy()
    if array.ndim == 4:
        array = array[0]
    array = np.clip(array, 0.0, 1.0)
    if array.ndim == 3 and array.shape[0] == 3:
        array = np.transpose(array, (1, 2, 0))
        return Image.fromarray((255.0 * array).astype(np.uint8), mode="RGB")
    array = np.squeeze(array)
    return Image.fromarray((255.0 * array).astype(np.uint8), mode="L")


def save_training_panel(I: torch.Tensor, I_new: torch.Tensor, I_blur: torch.Tensor, output_path: Path) -> None:
    panels = [
        ("I (objetivo)", to_uint8_image(I)),
        ("I_new = f_theta(I)", to_uint8_image(I_new)),
        ("h * I_new", to_uint8_image(I_blur)),
    ]
    panel_size = 220
    header_h = 40
    canvas = Image.new("L", (panel_size * len(panels), panel_size + header_h), color=255)
    draw = ImageDraw.Draw(canvas)
    for idx, (title, image) in enumerate(panels):
        tile = image.convert("RGB").resize((panel_size, panel_size), Image.Resampling.BICUBIC)
        x0 = idx * panel_size
        if canvas.mode != "RGB":
            canvas = canvas.convert("RGB")
            draw = ImageDraw.Draw(canvas)
        canvas.paste(tile, (x0, header_h))
        draw.text((x0 + 8, 12), title, fill=0)
    canvas.save(output_path)


def save_metrics_table_image(
    summary_rows: list[tuple[str, str, str, str]],
    output_path: Path,
    severity_label: str,
) -> None:
    row_h = 42
    width = 980
    height = 120 + row_h * (len(summary_rows) + 1)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    draw.text((24, 20), f"Myopia Severity = {severity_label}", fill="black")
    draw.line([(24, 52), (width - 24, 52)], fill="black", width=2)

    headers = ["Metodo", "SSIM", "PSNR", "LPIPS"]
    col_x = [24, 420, 610, 790]
    for x, header in zip(col_x, headers):
        draw.text((x, 66), header, fill="black")
    draw.line([(24, 96), (width - 24, 96)], fill="black", width=2)

    for idx, row in enumerate(summary_rows):
        y = 108 + idx * row_h
        for x, value in zip(col_x, row):
            draw.text((x, y), value, fill="black")
        draw.line([(24, y + 30), (width - 24, y + 30)], fill=(220, 220, 220), width=1)

    canvas.save(output_path)


def add_zoom_strip(base: Image.Image, zoom_box: tuple[int, int, int, int], target_width: int, target_height: int) -> Image.Image:
    crop = base.crop(zoom_box).resize((target_width, target_height), Image.Resampling.BICUBIC)
    outlined = crop.convert("RGB")
    draw = ImageDraw.Draw(outlined)
    draw.rectangle([0, 0, target_width - 1, target_height - 1], outline=(50, 255, 90), width=3)
    return outlined


def save_qualitative_panel(
    I: torch.Tensor,
    I_corrected: torch.Tensor,
    I_wiener: torch.Tensor,
    I_new: torch.Tensor,
    I_baseline: torch.Tensor,
    metrics_labels: dict[str, str],
    output_path: Path,
) -> None:
    top_images = [
        ("Ground Truth", to_uint8_image(I)),
        ("Corrected", to_uint8_image(I_corrected)),
        ("Myopia", to_uint8_image(I_baseline)),
    ]
    panel_w = 220
    panel_h = 220
    zoom_h = 90
    header_h = 54
    footer_h = 110
    canvas_w = panel_w * 3
    canvas_h = header_h + panel_h + footer_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)
    zoom_box = (40, 40, 88, 88)

    def paste_group(images: list[tuple[str, Image.Image]], y_offset: int, x_positions: list[int]) -> None:
        for (title, image), x0 in zip(images, x_positions):
            resized = image.resize((panel_w, panel_h), Image.Resampling.BICUBIC)
            canvas.paste(resized, (x0, y_offset + header_h))
            draw.text((x0 + 12, y_offset + 14), title, fill="black")
            metric_text = metrics_labels.get(title, "")
            if metric_text:
                draw.text((x0 + 12, y_offset + 32), metric_text, fill="black")

            zx0 = int(zoom_box[0] * panel_w / image.width)
            zy0 = int(zoom_box[1] * panel_h / image.height)
            zx1 = int(zoom_box[2] * panel_w / image.width)
            zy1 = int(zoom_box[3] * panel_h / image.height)
            draw.rectangle(
                [x0 + zx0, y_offset + header_h + zy0, x0 + zx1, y_offset + header_h + zy1],
                outline=(50, 255, 90),
                width=3,
            )

            zoom = add_zoom_strip(resized, (zx0, zy0, zx1, zy1), panel_w, zoom_h)
            canvas.paste(zoom, (x0, y_offset + header_h + panel_h + 10))

    paste_group(top_images, 0, [0, panel_w, 2 * panel_w])

    canvas.save(output_path)
