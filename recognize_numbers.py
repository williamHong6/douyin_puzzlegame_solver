#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


ROWS = 16
COLS = 10
TOTAL_TILES = ROWS * COLS
CANVAS_SIZE = 64


def parse_args():
    parser = argparse.ArgumentParser(
        description="Recognize digits from cropped tile images and reconstruct a 16x10 board."
    )
    parser.add_argument("tiles_folder", help="Folder with cropped tiles, e.g. temp")
    parser.add_argument("templates_folder", help="Folder with templates, e.g. numberpic")
    return parser.parse_args()


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if image is None:
        raise ValueError(f"Cannot load image: {path}")

    # If transparent PNG, put it on a white background.
    if image.ndim == 3 and image.shape[2] == 4:
        bgr = image[:, :, :3]
        alpha = image[:, :, 3].astype(np.float32) / 255.0

        white = np.ones_like(bgr, dtype=np.uint8) * 255
        result = np.zeros_like(bgr, dtype=np.uint8)

        for c in range(3):
            result[:, :, c] = (
                bgr[:, :, c].astype(np.float32) * alpha
                + white[:, :, c].astype(np.float32) * (1.0 - alpha)
            ).astype(np.uint8)

        return result

    return image


def extract_digit_mask(image: np.ndarray) -> np.ndarray:
    """
    Extract only the black digit.

    Output:
    - digit = white pixels 255
    - background = black pixels 0
    """

    if image.ndim == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Slight blur helps remove small edge noise.
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Important:
    # Use a stricter threshold.
    # Black digit is very dark.
    # Tile edge / shadow should not be included.
    mask = np.where(gray < 100, 255, 0).astype(np.uint8)

    # Remove tiny noise.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("No digit pixels found")

    x1 = xs.min()
    x2 = xs.max()
    y1 = ys.min()
    y2 = ys.max()

    digit = mask[y1:y2 + 1, x1:x2 + 1]

    return digit


def normalize_digit_keep_ratio(digit_mask: np.ndarray, canvas_size: int = CANVAS_SIZE) -> np.ndarray:
    """
    Resize digit while keeping aspect ratio.
    Put it centered on a 64x64 black canvas.

    This is very important.
    Do NOT stretch digit into a square.
    """

    h, w = digit_mask.shape[:2]

    if h == 0 or w == 0:
        raise ValueError("Empty digit mask")

    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

    # Leave margin around the digit.
    max_digit_size = int(canvas_size * 0.75)

    scale = max_digit_size / max(h, w)

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(
        digit_mask,
        (new_w, new_h),
        interpolation=cv2.INTER_NEAREST
    )

    # Center it.
    x_offset = (canvas_size - new_w) // 2
    y_offset = (canvas_size - new_h) // 2

    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return canvas


def normalize_image_to_mask(image: np.ndarray) -> np.ndarray:
    digit = extract_digit_mask(image)
    normalized = normalize_digit_keep_ratio(digit)
    return normalized


def compare_masks(a: np.ndarray, b: np.ndarray) -> float:
    """
    Smaller score = better match.

    This combines:
    1. MSE
    2. IoU-style penalty
    """

    a_bin = (a > 0).astype(np.uint8)
    b_bin = (b > 0).astype(np.uint8)

    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)

    intersection = np.logical_and(a_bin, b_bin).sum()
    union = np.logical_or(a_bin, b_bin).sum()

    if union == 0:
        iou_penalty = 1.0
    else:
        iou = intersection / union
        iou_penalty = 1.0 - iou

    # Combine both.
    score = mse + iou_penalty * 20000

    return float(score)


def load_templates(templates_folder: Path) -> dict[int, list[np.ndarray]]:
    """
    Supports both:
    numberpic/1.png
    numberpic/2.png

    And also:
    numberpic/1/a.png
    numberpic/1/b.png

    Multiple templates per digit are better.
    """

    templates = {}

    for digit in range(1, 10):
        templates[digit] = []

        single_file = templates_folder / f"{digit}.png"
        digit_folder = templates_folder / str(digit)

        paths = []

        if single_file.exists():
            paths.append(single_file)

        if digit_folder.exists() and digit_folder.is_dir():
            paths.extend(sorted(digit_folder.glob("*.png")))

        if not paths:
            raise FileNotFoundError(
                f"Missing template for digit {digit}. Need {single_file} or folder {digit_folder}/"
            )

        for path in paths:
            image = load_image(path)
            mask = normalize_image_to_mask(image)
            templates[digit].append(mask)

    return templates


def recognize_one_tile(tile_path: Path, templates: dict[int, list[np.ndarray]]) -> tuple[int, float, float]:
    image = load_image(tile_path)
    tile_mask = normalize_image_to_mask(image)

    best_digit = None
    best_score = float("inf")

    second_best_score = float("inf")

    for digit, masks in templates.items():
        for template_mask in masks:
            score = compare_masks(tile_mask, template_mask)

            if score < best_score:
                second_best_score = best_score
                best_score = score
                best_digit = digit
            elif score < second_best_score:
                second_best_score = score

    if best_digit is None:
        raise ValueError(f"Could not recognize {tile_path.name}")

    # Confidence based on distance between best and second best.
    # Bigger gap = more confident.
    if second_best_score == float("inf"):
        confidence = 1.0
    else:
        gap = second_best_score - best_score
        confidence = max(0.0, min(1.0, gap / max(second_best_score, 1.0)))

    return best_digit, confidence, best_score


def get_tile_paths(tiles_folder: Path) -> list[Path]:
    paths = sorted(
        [p for p in tiles_folder.iterdir() if p.suffix.lower() == ".png"],
        key=lambda p: p.name
    )

    if len(paths) != TOTAL_TILES:
        raise ValueError(f"Expected {TOTAL_TILES} png files, found {len(paths)}")

    return paths


def save_results(
    flat_numbers: list[int],
    board: list[list[int]],
    details: list[dict],
    output_folder: Path
):
    data = {
        "rows": ROWS,
        "cols": COLS,
        "flat_numbers": flat_numbers,
        "board": board,
        "details": details
    }

    with open(output_folder / "board.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    with open(output_folder / "board.txt", "w", encoding="utf-8") as f:
        for row in board:
            f.write(" ".join(str(x) for x in row) + "\n")


def main():
    args = parse_args()

    tiles_folder = Path(args.tiles_folder)
    templates_folder = Path(args.templates_folder)

    if not tiles_folder.exists():
        print(f"Error: tiles folder not found: {tiles_folder}")
        sys.exit(1)

    if not templates_folder.exists():
        print(f"Error: templates folder not found: {templates_folder}")
        sys.exit(1)

    try:
        templates = load_templates(templates_folder)
    except Exception as e:
        print(f"Error loading templates: {e}")
        sys.exit(1)

    try:
        tile_paths = get_tile_paths(tiles_folder)
    except Exception as e:
        print(f"Error reading tile folder: {e}")
        sys.exit(1)

    flat_numbers = []
    details = []

    for index, tile_path in enumerate(tile_paths, start=1):
        try:
            digit, confidence, score = recognize_one_tile(tile_path, templates)
        except Exception as e:
            print(f"Error recognizing {tile_path.name}: {e}")
            sys.exit(1)

        flat_numbers.append(digit)

        details.append({
            "index": index,
            "file": tile_path.name,
            "digit": digit,
            "confidence": confidence,
            "score": score
        })

    board = [
        flat_numbers[i:i + COLS]
        for i in range(0, TOTAL_TILES, COLS)
    ]

    print("\nRecognized board:")
    print("-----------------")
    for row in board:
        print(" ".join(str(x) for x in row))

    low = [d for d in details if d["confidence"] < 0.15]

    if low:
        print("\nWarning: low confidence tiles:")
        for d in low:
            print(
                f'{d["file"]}: digit={d["digit"]}, '
                f'confidence={d["confidence"]:.3f}, score={d["score"]:.1f}'
            )

    save_results(flat_numbers, board, details, tiles_folder)

    print("\nSaved:")
    print(f"  {tiles_folder / 'board.json'}")
    print(f"  {tiles_folder / 'board.txt'}")


if __name__ == "__main__":
    main()