#!/usr/bin/env python3
import argparse
import json
import shutil
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
import numpy as np


CANVAS_SIZE = 64
TARGET_SUM = 10
LOW_CONFIDENCE_THRESHOLD = 0.15
BATCH_COLORS = [
    "#ffcc66",
    "#99ccff",
    "#ff9999",
    "#99dd99",
    "#cc99ff",
    "#ffb366",
    "#66cccc",
    "#f2a6d8",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Local tkinter batch helper for a number elimination game."
    )
    parser.add_argument(
        "image_path",
        nargs="?",
        default="a.png",
        help="Board screenshot path, default: a.png",
    )
    parser.add_argument("--rows", type=int, default=16, help="Board row count")
    parser.add_argument("--cols", type=int, default=10, help="Board column count")
    parser.add_argument("--templates", default="numberpic", help="Template folder")
    parser.add_argument("--output", default="output", help="Output folder")
    parser.add_argument(
        "--tile-trim",
        type=int,
        default=3,
        help="Inner trim in pixels applied to each cropped tile, default: 3",
    )
    return parser.parse_args()


def load_image(path: Path, flags=cv2.IMREAD_UNCHANGED) -> np.ndarray:
    image = cv2.imread(str(path), flags)
    if image is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return image


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def ensure_clean_dir(path: Path):
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    path.mkdir(parents=True, exist_ok=True)


def clone_board(board: list[list[int]]) -> list[list[int]]:
    return [row[:] for row in board]


def rgba_on_white(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 3:
        return image

    bgr = image[:, :, :3].astype(np.float32)
    alpha = image[:, :, 3:4].astype(np.float32) / 255.0
    white = np.full_like(bgr, 255.0)
    blended = bgr * alpha + white * (1.0 - alpha)
    return blended.astype(np.uint8)


def build_white_mask(image_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv, (0, 0, 130), (179, 80, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return white_mask


def get_candidate_boxes(image_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    mask = build_white_mask(image_bgr)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = image_bgr.shape[:2]
    image_area = height * width
    boxes = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        rect_area = w * h
        if area < image_area * 0.0008:
            continue
        if w < width * 0.025 or h < height * 0.02:
            continue
        if rect_area <= 0:
            continue
        fill_ratio = area / rect_area
        aspect = w / max(h, 1)
        if not (0.5 <= aspect <= 2.2):
            continue
        if fill_ratio < 0.45:
            continue
        boxes.append((x, y, w, h))

    return boxes


def cluster_sorted_values(values: list[float], target_groups: int) -> list[list[float]]:
    if not values:
        return []

    groups = [[values[0]]]
    if len(values) == 1:
        return groups

    diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    positive_diffs = [diff for diff in diffs if diff > 0]
    base_gap = np.median(positive_diffs) if positive_diffs else 0.0
    threshold = max(8.0, base_gap * 0.6)

    for value in values[1:]:
        if abs(value - np.mean(groups[-1])) <= threshold:
            groups[-1].append(value)
        else:
            groups.append([value])

    while len(groups) > target_groups:
        merge_index = min(
            range(len(groups) - 1),
            key=lambda idx: abs(np.mean(groups[idx + 1]) - np.mean(groups[idx])),
        )
        groups[merge_index].extend(groups.pop(merge_index + 1))

    while len(groups) < target_groups:
        split_index = max(range(len(groups)), key=lambda idx: len(groups[idx]))
        group = groups[split_index]
        if len(group) <= 1:
            break
        midpoint = len(group) // 2
        groups[split_index:split_index + 1] = [group[:midpoint], group[midpoint:]]

    return groups


def select_evenly_spaced_boxes(
    boxes: list[tuple[int, int, int, int]], target_count: int
) -> list[tuple[int, int, int, int]]:
    if len(boxes) <= target_count:
        return boxes

    centers = np.array([box[0] + box[2] / 2.0 for box in boxes], dtype=np.float32)
    expected = np.linspace(centers.min(), centers.max(), target_count)
    selected = []
    used = set()

    for point in expected:
        remaining = [
            (abs(centers[idx] - point), idx)
            for idx in range(len(boxes))
            if idx not in used
        ]
        _, chosen = min(remaining, key=lambda item: item[0])
        used.add(chosen)
        selected.append(boxes[chosen])

    return sorted(selected, key=lambda box: box[0] + box[2] / 2.0)


def group_boxes_into_rows(
    boxes: list[tuple[int, int, int, int]], rows: int, cols: int
) -> list[tuple[int, int, int, int]] | None:
    if len(boxes) < rows * cols:
        return None

    boxes_sorted = sorted(boxes, key=lambda box: box[1] + box[3] / 2.0)
    y_centers = [box[1] + box[3] / 2.0 for box in boxes_sorted]
    y_groups = cluster_sorted_values(y_centers, rows)
    if len(y_groups) != rows:
        return None

    row_centers = [float(np.mean(group)) for group in y_groups]
    rows_data = [[] for _ in range(rows)]

    for box in boxes:
        center_y = box[1] + box[3] / 2.0
        row_index = min(range(rows), key=lambda idx: abs(center_y - row_centers[idx]))
        rows_data[row_index].append(box)

    ordered = []
    for row_boxes in rows_data:
        if len(row_boxes) < cols:
            return None
        row_boxes = sorted(row_boxes, key=lambda box: box[0] + box[2] / 2.0)
        if len(row_boxes) > cols:
            row_boxes = select_evenly_spaced_boxes(row_boxes, cols)
        ordered.extend(row_boxes)

    return ordered if len(ordered) == rows * cols else None


def detect_board_bbox(
    image_bgr: np.ndarray, rows: int, cols: int, boxes: list[tuple[int, int, int, int]]
) -> tuple[int, int, int, int]:
    height, width = image_bgr.shape[:2]
    if boxes:
        xs = [x for x, _, w, _ in boxes] + [x + w for x, _, w, _ in boxes]
        ys = [y for _, y, _, h in boxes] + [y + h for _, y, _, h in boxes]
        pad_x = max(4, int(np.median([box[2] for box in boxes]) * 0.15))
        pad_y = max(4, int(np.median([box[3] for box in boxes]) * 0.15))
        return (
            max(0, min(xs) - pad_x),
            max(0, min(ys) - pad_y),
            min(width, max(xs) + pad_x),
            min(height, max(ys) + pad_y),
        )

    mask = build_white_mask(image_bgr)
    coords = cv2.findNonZero(mask)
    if coords is None:
        return (0, 0, width, height)

    x, y, w, h = cv2.boundingRect(coords)
    pad_x = max(6, w // (cols * 4))
    pad_y = max(6, h // (rows * 4))
    return (
        max(0, x - pad_x),
        max(0, y - pad_y),
        min(width, x + w + pad_x),
        min(height, y + h + pad_y),
    )


def fallback_grid_boxes(
    image_bgr: np.ndarray, rows: int, cols: int, board_bbox: tuple[int, int, int, int]
) -> list[tuple[int, int, int, int]]:
    x0, y0, x1, y1 = board_bbox
    board_width = x1 - x0
    board_height = y1 - y0
    boxes = []

    for row in range(rows):
        for col in range(cols):
            left = x0 + int(round(col * board_width / cols))
            right = x0 + int(round((col + 1) * board_width / cols))
            top = y0 + int(round(row * board_height / rows))
            bottom = y0 + int(round((row + 1) * board_height / rows))
            boxes.append((left, top, max(1, right - left), max(1, bottom - top)))

    return boxes


def build_tile_alpha(cell_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv, (0, 0, 125), (179, 85, 255))
    green_mask = cv2.inRange(hsv, (30, 25, 25), (100, 255, 255))
    gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    black_mask = cv2.inRange(gray, 0, 105)
    opaque_mask = cv2.bitwise_or(white_mask, black_mask)
    opaque_mask = cv2.bitwise_and(opaque_mask, cv2.bitwise_not(green_mask))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opaque_mask = cv2.morphologyEx(opaque_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    opaque_mask = cv2.morphologyEx(opaque_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opaque_mask, connectivity=8)
    if num_labels > 1:
        largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        opaque_mask = np.uint8(labels == largest_label) * 255

    return opaque_mask


def crop_single_tile(image_bgr: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = box
    pad_x = max(2, int(round(w * 0.08)))
    pad_y = max(2, int(round(h * 0.08)))
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(image_bgr.shape[1], x + w + pad_x)
    y1 = min(image_bgr.shape[0], y + h + pad_y)

    cell = image_bgr[y0:y1, x0:x1]
    alpha = build_tile_alpha(cell)
    coords = cv2.findNonZero(alpha)
    if coords is None:
        rgba = cv2.cvtColor(cell, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = 255
        return rgba

    tx, ty, tw, th = cv2.boundingRect(coords)
    pad = 2
    tx0 = max(0, tx - pad)
    ty0 = max(0, ty - pad)
    tx1 = min(cell.shape[1], tx + tw + pad)
    ty1 = min(cell.shape[0], ty + th + pad)
    cropped = cell[ty0:ty1, tx0:tx1]
    cropped_alpha = alpha[ty0:ty1, tx0:tx1]

    rgba = cv2.cvtColor(cropped, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = cropped_alpha
    rgba[cropped_alpha == 0] = [255, 255, 255, 0]
    return rgba


def inner_trim_tile(tile: np.ndarray, trim_px: int = 5) -> np.ndarray:
    h, w = tile.shape[:2]
    if trim_px <= 0:
        return tile
    if h <= trim_px * 2 + 5 or w <= trim_px * 2 + 5:
        return tile
    return tile[trim_px:h - trim_px, trim_px:w - trim_px]


def crop_tiles(
    image_path: Path, rows: int, cols: int, output_dir: Path, tile_trim: int = 5
) -> tuple[list[Path], list[dict], str]:
    image_bgr = load_image(image_path, cv2.IMREAD_COLOR)
    temp_dir = output_dir / "temp"
    ensure_clean_dir(temp_dir)

    candidate_boxes = get_candidate_boxes(image_bgr)
    ordered_boxes = group_boxes_into_rows(candidate_boxes, rows, cols)
    method = "contour-detection"

    if ordered_boxes is None or len(ordered_boxes) != rows * cols:
        method = "fallback-grid"
        board_bbox = detect_board_bbox(image_bgr, rows, cols, candidate_boxes)
        ordered_boxes = fallback_grid_boxes(image_bgr, rows, cols, board_bbox)

    tile_paths = []
    tile_meta = []
    for index, box in enumerate(ordered_boxes, start=1):
        tile_rgba = crop_single_tile(image_bgr, box)
        tile_rgba = inner_trim_tile(tile_rgba, tile_trim)
        tile_path = temp_dir / f"{index:03d}.png"
        cv2.imwrite(str(tile_path), tile_rgba)
        tile_paths.append(tile_path)
        x, y, w, h = box
        tile_meta.append(
            {
                "index": index,
                "file": tile_path.name,
                "bbox": [int(x), int(y), int(w), int(h)],
                "tile_trim": int(tile_trim),
            }
        )

    expected = rows * cols
    if len(tile_paths) != expected:
        raise RuntimeError(f"Expected {expected} tiles, wrote {len(tile_paths)}")

    return tile_paths, tile_meta, method


def extract_digit_mask(image: np.ndarray) -> np.ndarray:
    image = rgba_on_white(image)
    if image.ndim == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    dark_mask = cv2.inRange(gray, 0, 105)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    coords = cv2.findNonZero(dark_mask)
    if coords is None:
        raise ValueError("No digit pixels found")

    x, y, w, h = cv2.boundingRect(coords)
    return dark_mask[y:y + h, x:x + w]


def normalize_digit_keep_ratio(digit_mask: np.ndarray, canvas_size: int = CANVAS_SIZE) -> np.ndarray:
    height, width = digit_mask.shape[:2]
    if height == 0 or width == 0:
        raise ValueError("Empty digit mask")

    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    max_digit_size = int(canvas_size * 0.75)
    scale = max_digit_size / max(height, width)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(digit_mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    x_offset = (canvas_size - new_width) // 2
    y_offset = (canvas_size - new_height) // 2
    canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized
    return canvas


def normalize_image_to_mask(image: np.ndarray) -> np.ndarray:
    return normalize_digit_keep_ratio(extract_digit_mask(image))


def count_holes(mask: np.ndarray) -> int:
    binary = np.uint8(mask > 0) * 255
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return 0

    hole_count = 0
    for idx in range(len(contours)):
        parent = hierarchy[0][idx][3]
        if parent != -1 and cv2.contourArea(contours[idx]) > 8:
            hole_count += 1
    return hole_count


def compare_masks(a: np.ndarray, b: np.ndarray) -> float:
    a_bin = (a > 0).astype(np.uint8)
    b_bin = (b > 0).astype(np.uint8)

    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    intersection = np.logical_and(a_bin, b_bin).sum()
    union = np.logical_or(a_bin, b_bin).sum()
    iou_penalty = 1.0 if union == 0 else 1.0 - (intersection / union)
    hole_penalty = abs(count_holes(a) - count_holes(b)) * 9000.0

    return float(mse + iou_penalty * 20000.0 + hole_penalty)


def load_templates(templates_dir: Path) -> dict[int, list[dict]]:
    templates = {}
    for digit in range(1, 10):
        paths = []
        single = templates_dir / f"{digit}.png"
        nested_dir = templates_dir / str(digit)
        if single.exists():
            paths.append(single)
        if nested_dir.exists() and nested_dir.is_dir():
            paths.extend(sorted(nested_dir.glob("*.png")))
        if not paths:
            raise FileNotFoundError(
                f"Missing templates for digit {digit} in {single} or {nested_dir}"
            )

        templates[digit] = []
        for path in paths:
            mask = normalize_image_to_mask(load_image(path, cv2.IMREAD_UNCHANGED))
            templates[digit].append(
                {
                    "path": str(path),
                    "mask": mask,
                    "holes": count_holes(mask),
                }
            )

    return templates


def recognize_one_tile(
    tile_path: Path, templates: dict[int, list[dict]]
) -> tuple[int, float, float, int]:
    tile_mask = normalize_image_to_mask(load_image(tile_path, cv2.IMREAD_UNCHANGED))
    tile_holes = count_holes(tile_mask)

    best_digit = None
    best_score = float("inf")
    second_best_score = float("inf")

    for digit, template_list in templates.items():
        for template in template_list:
            score = compare_masks(tile_mask, template["mask"])
            if score < best_score:
                second_best_score = best_score
                best_score = score
                best_digit = digit
            elif score < second_best_score:
                second_best_score = score

    if best_digit is None:
        raise ValueError(f"Could not recognize tile {tile_path.name}")

    if second_best_score == float("inf"):
        confidence = 1.0
    else:
        gap = second_best_score - best_score
        confidence = max(0.0, min(1.0, gap / max(second_best_score, 1.0)))

    return best_digit, confidence, float(best_score), tile_holes


def recognize_board(
    tile_paths: list[Path], templates_dir: Path, rows: int, cols: int
) -> tuple[list[int], list[list[int]], list[dict]]:
    templates = load_templates(templates_dir)
    flat_numbers = []
    recognition_details = []

    for index, tile_path in enumerate(tile_paths, start=1):
        digit, confidence, score, hole_count = recognize_one_tile(tile_path, templates)
        flat_numbers.append(digit)
        recognition_details.append(
            {
                "index": index,
                "file": tile_path.name,
                "digit": digit,
                "confidence": confidence,
                "score": score,
                "hole_count": hole_count,
            }
        )

    expected = rows * cols
    if len(flat_numbers) != expected:
        raise RuntimeError(f"Expected {expected} recognized numbers, got {len(flat_numbers)}")

    board = [flat_numbers[index:index + cols] for index in range(0, expected, cols)]
    return flat_numbers, board, recognition_details


def board_to_text(board: list[list[int]]) -> str:
    return "\n".join(" ".join(str(value) for value in row) for row in board)


def count_non_zero(board: list[list[int]]) -> int:
    return sum(1 for row in board for value in row if value != 0)


def cell_human(row: int, col: int) -> str:
    return f"R{row + 1}C{col + 1}"


def rect_human(r1: int, c1: int, r2: int, c2: int) -> str:
    return f"R{r1 + 1}C{c1 + 1}:R{r2 + 1}C{c2 + 1}"


def build_prefix_sum(board: list[list[int]]) -> np.ndarray:
    array = np.array(board, dtype=np.int32)
    prefix = np.pad(array, ((1, 0), (1, 0)), mode="constant")
    return prefix.cumsum(axis=0).cumsum(axis=1)


def build_prefix_count(board: list[list[int]]) -> np.ndarray:
    array = np.array((np.array(board, dtype=np.int32) > 0).astype(np.int32))
    prefix = np.pad(array, ((1, 0), (1, 0)), mode="constant")
    return prefix.cumsum(axis=0).cumsum(axis=1)


def rect_sum(prefix: np.ndarray, r1: int, c1: int, r2: int, c2: int) -> int:
    return int(
        prefix[r2 + 1, c2 + 1]
        - prefix[r1, c2 + 1]
        - prefix[r2 + 1, c1]
        + prefix[r1, c1]
    )


def rect_nonzero_count(prefix_count: np.ndarray, r1: int, c1: int, r2: int, c2: int) -> int:
    return rect_sum(prefix_count, r1, c1, r2, c2)


def classify_move(height: int, width: int, area: int) -> str:
    if area <= 6:
        return "easy"
    if height == 1 and width <= 4:
        return "easy"
    if width == 1 and height <= 4:
        return "easy"
    if height == 2 and width == 2:
        return "easy"
    return "large"


def find_valid_moves(board: list[list[int]]) -> list[dict]:
    rows = len(board)
    cols = len(board[0]) if rows else 0
    prefix_sum = build_prefix_sum(board)
    prefix_count = build_prefix_count(board)
    moves = []

    for r1 in range(rows):
        for c1 in range(cols):
            for r2 in range(r1, rows):
                for c2 in range(c1, cols):
                    total = rect_sum(prefix_sum, r1, c1, r2, c2)
                    if total > TARGET_SUM:
                        continue
                    if total != TARGET_SUM:
                        continue

                    non_zero_count = rect_nonzero_count(prefix_count, r1, c1, r2, c2)
                    if non_zero_count < 2:
                        continue

                    values = []
                    positions = []
                    for row in range(r1, r2 + 1):
                        for col in range(c1, c2 + 1):
                            value = board[row][col]
                            if value != 0:
                                values.append(value)
                                positions.append([row, col])

                    if len(values) < 2:
                        continue

                    height = r2 - r1 + 1
                    width = c2 - c1 + 1
                    area = height * width
                    category = classify_move(height, width, area)
                    moves.append(
                        {
                            "id": 0,
                            "display_id": "",
                            "rect": [r1, c1, r2, c2],
                            "rect_human": rect_human(r1, c1, r2, c2),
                            "values": values,
                            "positions": positions,
                            "remove_positions": positions[:],
                            "positions_human": [cell_human(row, col) for row, col in positions],
                            "sum": total,
                            "area": area,
                            "removed_count": len(values),
                            "height": height,
                            "width": width,
                            "category": category,
                            "color": "",
                        }
                    )

    return moves


def sort_moves_for_batch(moves: list[dict]) -> list[dict]:
    sorted_moves = sorted(
        moves,
        key=lambda move: (
            0 if move["category"] == "easy" else 1,
            move["area"],
            -move["removed_count"],
            max(move["height"], move["width"]),
            move["rect"][0],
            move["rect"][1],
            move["rect"][2],
            move["rect"][3],
        ),
    )

    easy_index = 0
    large_index = 0
    for idx, move in enumerate(sorted_moves, start=1):
        move["id"] = idx
        if move["category"] == "easy":
            easy_index += 1
            move["display_id"] = f"A{easy_index}"
        else:
            large_index += 1
            move["display_id"] = f"B{large_index}"
    return sorted_moves


def select_non_overlapping_batch(moves: list[dict]) -> list[dict]:
    selected = []
    used_positions = set()

    for move in sort_moves_for_batch(moves):
        remove_positions = {tuple(pos) for pos in move["remove_positions"]}
        if remove_positions & used_positions:
            continue
        selected.append(move.copy())
        used_positions.update(remove_positions)

    for index, move in enumerate(selected):
        move["color"] = BATCH_COLORS[index % len(BATCH_COLORS)]

    return selected


def apply_round(board: list[list[int]], batch_moves: list[dict]) -> list[list[int]]:
    new_board = clone_board(board)
    for move in batch_moves:
        for row, col in move["remove_positions"]:
            new_board[row][col] = 0
    return new_board


def format_move(move: dict, index_in_round: int | None = None) -> str:
    prefix = f"{index_in_round}. " if index_in_round is not None else f"{move['display_id']}. "
    values_text = " + ".join(str(value) for value in move["values"])
    return f"{prefix}{move['rect_human']} | {values_text} = {move['sum']}"


def save_board(
    output_dir: Path,
    rows: int,
    cols: int,
    flat_numbers: list[int],
    board: list[list[int]],
    recognition_details: list[dict],
    tile_meta: list[dict],
    crop_method: str,
):
    data = {
        "rows": rows,
        "cols": cols,
        "flat_numbers": flat_numbers,
        "board": board,
        "recognition_details": recognition_details,
        "crop_method": crop_method,
        "tiles": tile_meta,
    }
    (output_dir / "board.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    (output_dir / "board.txt").write_text(board_to_text(board) + "\n", encoding="utf-8")


def save_history(
    output_dir: Path,
    original_board: list[list[int]],
    current_board: list[list[int]],
    round_history: list[dict],
):
    history_json = {
        "original_board": original_board,
        "current_board": current_board,
        "round_count": len(round_history),
        "removed_count": count_non_zero(original_board) - count_non_zero(current_board),
        "remaining_count": count_non_zero(current_board),
        "rounds": round_history,
    }
    (output_dir / "history.json").write_text(json.dumps(history_json, indent=2), encoding="utf-8")

    lines = []
    if not round_history:
        lines.append("No rounds applied yet.")
        lines.append("")
    else:
        for round_item in round_history:
            lines.append(f"Round {round_item['round']}")
            for index, move in enumerate(round_item["moves"], start=1):
                values_text = " + ".join(str(value) for value in move["values"])
                cells_text = ", ".join(move["positions_human"])
                lines.append(
                    f"{index}. Select {move['rect_human']} | {values_text} = {move['sum']} | cells {cells_text}"
                )
            lines.append("")

    lines.append("Current board:")
    lines.append(board_to_text(current_board))
    lines.append("")
    lines.append(f"Removed count: {history_json['removed_count']}")
    lines.append(f"Remaining count: {history_json['remaining_count']}")
    (output_dir / "history.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


class GameHelperApp:
    def __init__(self, root: tk.Tk, original_board: list[list[int]], output_dir: Path):
        self.root = root
        self.root.title("Number Elimination Batch Helper")
        self.root.geometry("760x860")
        self.output_dir = output_dir
        self.original_board = clone_board(original_board)
        self.current_board = clone_board(original_board)
        self.round_history: list[dict] = []
        self.all_possible_moves: list[dict] = []
        self.current_batch: list[dict] = []
        self.status_var = tk.StringVar()

        self.rows = len(original_board)
        self.cols = len(original_board[0]) if self.rows else 0
        self.cell_size = 34
        self.left_margin = 52
        self.top_margin = 32
        self.board_canvas: tk.Canvas | None = None

        self.build_layout()
        self.recalculate_batch()

    def build_layout(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.grid(row=0, column=0, sticky="nsew")
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)

        info_frame = ttk.LabelFrame(main_frame, text="Board Status", padding=10)
        info_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        ttk.Label(
            info_frame,
            textvariable=self.status_var,
            justify="left",
            font=("Menlo", 11),
        ).pack(anchor="w")

        board_frame = ttk.LabelFrame(main_frame, text="Current Round Board", padding=10)
        board_frame.grid(row=1, column=0, sticky="nsew")
        board_frame.columnconfigure(0, weight=1)
        board_frame.rowconfigure(0, weight=1)

        canvas_width = self.left_margin + self.cols * self.cell_size + 20
        canvas_height = self.top_margin + self.rows * self.cell_size + 20
        self.board_canvas = tk.Canvas(
            board_frame,
            width=canvas_width,
            height=canvas_height,
            bg="#f7f7f7",
            highlightthickness=0,
        )
        self.board_canvas.grid(row=0, column=0, sticky="nsew")
        button_frame = ttk.Frame(main_frame, padding=(0, 10, 0, 0))
        button_frame.grid(row=2, column=0, sticky="ew")
        for column in range(5):
            button_frame.columnconfigure(column, weight=1)

        ttk.Button(button_frame, text="Apply This Round", command=self.apply_this_round).grid(
            row=0, column=0, padx=4, pady=4, sticky="ew"
        )
        ttk.Button(button_frame, text="Recalculate Batch", command=self.recalculate_batch).grid(
            row=0, column=1, padx=4, pady=4, sticky="ew"
        )
        ttk.Button(button_frame, text="Undo Round", command=self.undo_round).grid(
            row=0, column=2, padx=4, pady=4, sticky="ew"
        )
        ttk.Button(button_frame, text="Reset", command=self.reset_board).grid(
            row=0, column=3, padx=4, pady=4, sticky="ew"
        )
        ttk.Button(button_frame, text="Save History", command=self.save_history_now).grid(
            row=0, column=4, padx=4, pady=4, sticky="ew"
        )

    def refresh_status(self):
        round_number = len(self.round_history) + 1
        remaining_count = count_non_zero(self.current_board)
        removed_count = count_non_zero(self.original_board) - remaining_count
        easy_count = sum(1 for move in self.current_batch if move["category"] == "easy")
        large_count = len(self.current_batch) - easy_count
        self.status_var.set(
            f"Round: {round_number}\n"
            f"Remaining numbers: {remaining_count}\n"
            f"Removed numbers: {removed_count}\n"
            f"Current batch size: {len(self.current_batch)}\n"
            f"Easy batch moves: {easy_count}\n"
            f"Larger batch moves: {large_count}\n"
            f"All valid moves now: {len(self.all_possible_moves)}"
        )

    def draw_batch_on_board(self):
        canvas = self.board_canvas
        canvas.delete("all")

        for col in range(self.cols):
            x = self.left_margin + col * self.cell_size + self.cell_size / 2
            canvas.create_text(x, 16, text=f"C{col + 1}", font=("Menlo", 9, "bold"))

        for row in range(self.rows):
            y = self.top_margin + row * self.cell_size + self.cell_size / 2
            canvas.create_text(22, y, text=f"R{row + 1}", font=("Menlo", 9, "bold"))

        cell_fill_map = {}
        cell_tag_map = {}
        for index, move in enumerate(self.current_batch, start=1):
            color = move["color"]
            for row, col in move["remove_positions"]:
                cell_fill_map[(row, col)] = color
                cell_tag_map[(row, col)] = str(index)

        for row in range(self.rows):
            for col in range(self.cols):
                x1 = self.left_margin + col * self.cell_size
                y1 = self.top_margin + row * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                value = self.current_board[row][col]

                if value == 0:
                    fill = "#d6d6d6"
                    text = ""
                    text_color = "#666666"
                else:
                    fill = cell_fill_map.get((row, col), "#ffffff")
                    text = str(value)
                    text_color = "#111111"

                canvas.create_rectangle(x1, y1, x2, y2, fill=fill, outline="#b5b5b5", width=1)
                if text:
                    canvas.create_text(
                        (x1 + x2) / 2,
                        (y1 + y2) / 2 + 1,
                        text=text,
                        font=("Menlo", 11, "bold"),
                        fill=text_color,
                    )
                if (row, col) in cell_tag_map:
                    canvas.create_text(
                        x1 + 8,
                        y1 + 8,
                        text=cell_tag_map[(row, col)],
                        font=("Menlo", 7, "bold"),
                        fill="#222222",
                    )

        for index, move in enumerate(self.current_batch, start=1):
            r1, c1, r2, c2 = move["rect"]
            x1 = self.left_margin + c1 * self.cell_size + 2
            y1 = self.top_margin + r1 * self.cell_size + 2
            x2 = self.left_margin + (c2 + 1) * self.cell_size - 2
            y2 = self.top_margin + (r2 + 1) * self.cell_size - 2
            canvas.create_rectangle(x1, y1, x2, y2, outline=move["color"], width=3)
            canvas.create_text(
                x1 + 10,
                y1 + 12,
                text=f"{index}",
                font=("Menlo", 8, "bold"),
                fill="#111111",
            )

    def refresh_ui(self):
        self.refresh_status()
        self.draw_batch_on_board()
        save_history(self.output_dir, self.original_board, self.current_board, self.round_history)

    def recalculate_batch(self):
        self.all_possible_moves = find_valid_moves(self.current_board)
        self.current_batch = select_non_overlapping_batch(self.all_possible_moves)
        self.refresh_ui()

    def apply_this_round(self):
        if not self.current_batch:
            messagebox.showinfo("No Moves", "There is no current batch to apply.")
            return

        round_number = len(self.round_history) + 1
        board_before = clone_board(self.current_board)
        board_after = apply_round(self.current_board, self.current_batch)
        self.round_history.append(
            {
                "round": round_number,
                "board_before": board_before,
                "board_after": clone_board(board_after),
                "moves": [move.copy() for move in self.current_batch],
            }
        )
        self.current_board = board_after
        self.recalculate_batch()

        if not self.current_batch:
            messagebox.showinfo("No More Moves", "No valid moves remain after this round.")

    def undo_round(self):
        if not self.round_history:
            messagebox.showinfo("Undo Round", "There is no round to undo.")
            return

        self.round_history.pop()
        if self.round_history:
            self.current_board = clone_board(self.round_history[-1]["board_after"])
        else:
            self.current_board = clone_board(self.original_board)
        self.recalculate_batch()

    def reset_board(self):
        if not messagebox.askyesno("Reset", "Reset the board to the original recognized state?"):
            return
        self.current_board = clone_board(self.original_board)
        self.round_history = []
        self.recalculate_batch()

    def save_history_now(self):
        save_history(self.output_dir, self.original_board, self.current_board, self.round_history)
        messagebox.showinfo("Saved", f"Saved history to {self.output_dir}")


def start_tkinter_gui(original_board: list[list[int]], output_dir: Path):
    root = tk.Tk()
    GameHelperApp(root, original_board, output_dir)
    root.mainloop()


def main():
    args = parse_args()
    image_path = Path(args.image_path)
    templates_dir = Path(args.templates)
    output_dir = Path(args.output)

    if not image_path.exists():
        print(f"Error: input image not found: {image_path}")
        sys.exit(1)

    if not templates_dir.exists():
        print(f"Error: templates folder not found: {templates_dir}")
        sys.exit(1)

    ensure_dir(output_dir)

    try:
        tile_paths, tile_meta, crop_method = crop_tiles(
            image_path,
            args.rows,
            args.cols,
            output_dir,
            tile_trim=args.tile_trim,
        )
    except Exception as exc:
        print(f"Error during cropping: {exc}")
        sys.exit(1)

    print(
        f"Cropped {len(tile_paths)} tiles to {output_dir / 'temp'} "
        f"using {crop_method} with tile trim {args.tile_trim}px."
    )

    try:
        flat_numbers, board, recognition_details = recognize_board(
            tile_paths, templates_dir, args.rows, args.cols
        )
    except Exception as exc:
        print(f"Error during recognition: {exc}")
        sys.exit(1)

    print("\nRecognized board")
    print("----------------")
    print(board_to_text(board))

    low_confidence = [
        detail for detail in recognition_details if detail["confidence"] < LOW_CONFIDENCE_THRESHOLD
    ]
    if low_confidence:
        print("\nLow confidence warnings")
        print("-----------------------")
        for detail in low_confidence:
            print(
                f"{detail['file']}: digit={detail['digit']}, confidence={detail['confidence']:.3f}, "
                f"score={detail['score']:.1f}, holes={detail['hole_count']}"
            )

    all_moves = find_valid_moves(board)
    batch = select_non_overlapping_batch(all_moves)
    print(f"\nAll valid moves: {len(all_moves)}")
    print(f"Initial batch moves: {len(batch)}")
    for index, move in enumerate(batch[:20], start=1):
        print(format_move(move, index))

    try:
        save_board(
            output_dir,
            args.rows,
            args.cols,
            flat_numbers,
            board,
            recognition_details,
            tile_meta,
            crop_method,
        )
        save_history(output_dir, board, board, [])
    except Exception as exc:
        print(f"Error saving output files: {exc}")
        sys.exit(1)

    print("\nOpening tkinter GUI for batch rounds...")

    try:
        start_tkinter_gui(board, output_dir)
    except tk.TclError as exc:
        print(f"Error starting tkinter GUI: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
