#!/usr/bin/env python3
import argparse
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Crop a 16x10 digit tile grid into individual tile PNGs."
    )
    parser.add_argument(
        "image_path",
        nargs="?",
        default="a.png",
        help="Path to input image, default: a.png",
    )
    return parser.parse_args()


def ensure_clean_folder(folder_path: Path):
    if folder_path.exists():
        if folder_path.is_dir():
            shutil.rmtree(folder_path)
        else:
            folder_path.unlink()
    folder_path.mkdir(parents=True, exist_ok=True)


def safe_round(value: float) -> int:
    return int(round(value))


def build_tile_mask(cell_bgr: np.ndarray) -> np.ndarray:
    """Return an alpha mask covering the white tile and digit inside a cell."""
    hsv = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2HSV)

    # Detect green background.
    green_mask = cv2.inRange(hsv, (35, 40, 40), (95, 255, 255))

    # Detect white tile pixels.
    white_mask = cv2.inRange(hsv, (0, 0, 120), (179, 70, 255))

    # Detect dark digit pixels.
    _, _, v = cv2.split(hsv)
    black_mask = cv2.inRange(v, 0, 80)

    # Keep white tile and black digit pixels, but exclude green background.
    opaque_mask = cv2.bitwise_or(white_mask, black_mask)
    opaque_mask = cv2.bitwise_and(opaque_mask, cv2.bitwise_not(green_mask))

    if opaque_mask.sum() == 0:
        return opaque_mask

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opaque_mask = cv2.morphologyEx(opaque_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    opaque_mask = cv2.morphologyEx(opaque_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opaque_mask, connectivity=8)
    if num_labels <= 1:
        return opaque_mask

    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    tile_mask = np.uint8(labels == largest_label) * 255

    tile_mask = cv2.morphologyEx(tile_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    tile_mask = cv2.dilate(tile_mask, kernel, iterations=1)

    return tile_mask


def crop_tile_from_cell(cell_bgr: np.ndarray, index: int, output_folder: Path) -> None:
    mask = build_tile_mask(cell_bgr)
    if mask.sum() == 0:
        raise ValueError("Unable to identify a white tile area inside cell index {}".format(index))

    coords = cv2.findNonZero(mask)
    x, y, w, h = cv2.boundingRect(coords)

    pad = 2
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(cell_bgr.shape[1], x + w + pad)
    y1 = min(cell_bgr.shape[0], y + h + pad)

    cropped = cell_bgr[y0:y1, x0:x1]
    cropped_mask = mask[y0:y1, x0:x1]

    rgba = cv2.cvtColor(cropped, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = cropped_mask
    transparent_pixels = cropped_mask == 0
    rgba[transparent_pixels] = [255, 255, 255, 0]

    output_path = output_folder / f"{index:03d}.png"
    Image.fromarray(cv2.cvtColor(rgba, cv2.COLOR_BGRA2RGBA)).save(output_path)


def main():
    args = parse_args()
    image_path = Path(args.image_path)

    if not image_path.exists():
        print(f"Error: input file does not exist: {image_path}")
        sys.exit(1)

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        print(f"Error: failed to load image: {image_path}")
        sys.exit(1)

    rows = 16
    cols = 10
    height, width = image.shape[:2]
    cell_w = width / cols
    cell_h = height / rows

    output_folder = Path("temp")
    ensure_clean_folder(output_folder)

    tile_index = 1
    for row in range(rows):
        for col in range(cols):
            x0 = safe_round(col * cell_w)
            x1 = safe_round((col + 1) * cell_w)
            y0 = safe_round(row * cell_h)
            y1 = safe_round((row + 1) * cell_h)

            # Expand the estimated cell region by a few pixels to safely capture the tile.
            pad_x = max(8, int(cell_w * 0.08))
            pad_y = max(8, int(cell_h * 0.08))
            x0a = max(0, x0 - pad_x)
            x1a = min(width, x1 + pad_x)
            y0a = max(0, y0 - pad_y)
            y1a = min(height, y1 + pad_y)

            cell = image[y0a:y1a, x0a:x1a]
            try:
                crop_tile_from_cell(cell, tile_index, output_folder)
            except Exception as exc:
                print(f"Warning: cell {tile_index:03d} fallback to fixed cell crop due to: {exc}")
                fallback = image[y0:y1, x0:x1]
                hsv_fallback = cv2.cvtColor(fallback, cv2.COLOR_BGR2HSV)
                green_mask_fallback = cv2.inRange(hsv_fallback, (35, 40, 40), (95, 255, 255))
                alpha_fallback = cv2.bitwise_not(green_mask_fallback)
                alpha_fallback = cv2.morphologyEx(alpha_fallback, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=2)
                fallback_rgba = cv2.cvtColor(fallback, cv2.COLOR_BGR2BGRA)
                fallback_rgba[:, :, 3] = alpha_fallback
                transparent_pixels = alpha_fallback == 0
                fallback_rgba[transparent_pixels] = [255, 255, 255, 0]
                Image.fromarray(cv2.cvtColor(fallback_rgba, cv2.COLOR_BGRA2RGBA)).save(
                    output_folder / f"{tile_index:03d}.png"
                )

            tile_index += 1

    if tile_index - 1 != rows * cols:
        print("Warning: expected 160 tiles but created {}".format(tile_index - 1))

    print(f"Wrote {rows * cols} tile images to '{output_folder}'")


if __name__ == "__main__":
    main()
