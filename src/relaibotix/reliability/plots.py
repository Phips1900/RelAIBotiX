"""Dependency-free publication plots for reliability outputs."""

from __future__ import annotations

from html import escape
import math
from pathlib import Path

import pandas as pd


def write_sensitivity_spider_svg(
    sensitivity: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Write a radar chart of the system probability ratio for each component."""

    required = {"component", "requested_factor", "system_probability_ratio"}
    missing = required - set(sensitivity.columns)
    if missing:
        raise ValueError(f"Sensitivity plot is missing columns: {sorted(missing)}")
    if sensitivity.empty:
        raise ValueError("Sensitivity plot requires at least one component.")

    values = sensitivity.sort_values("component", ignore_index=True)
    labels = values["component"].astype(str).tolist()
    ratios = values["system_probability_ratio"].astype(float).tolist()
    if any(not math.isfinite(value) or value < 0.0 for value in ratios):
        raise ValueError("Sensitivity ratios must be finite and non-negative.")

    width, height = 900, 760
    center_x, center_y, radius = 450.0, 375.0, 255.0
    maximum = max(max(ratios), 1.0)
    angles = [
        (-math.pi / 2.0) + 2.0 * math.pi * index / len(labels)
        for index in range(len(labels))
    ]

    def point(angle: float, value: float) -> tuple[float, float]:
        scaled = radius * value / maximum
        return center_x + scaled * math.cos(angle), center_y + scaled * math.sin(angle)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#263238}'
        '.label{font-size:14px}.tick{font-size:12px;fill:#667085}'
        '.title{font-size:20px;font-weight:600}</style>',
    ]
    factor = float(values["requested_factor"].iloc[0])
    lines.append(
        f'<text class="title" x="{center_x:g}" y="30" text-anchor="middle">'
        f'Sensitivity: system failure-probability ratio after ×{factor:g} perturbation</text>'
    )
    for fraction in (0.25, 0.5, 0.75, 1.0):
        ring = [point(angle, maximum * fraction) for angle in angles]
        coordinates = " ".join(f"{x:.2f},{y:.2f}" for x, y in ring)
        lines.append(
            f'<polygon points="{coordinates}" fill="none" stroke="#d0d5dd" '
            'stroke-width="1"/>'
        )
        lines.append(
            f'<text class="tick" x="{center_x + 5:g}" '
            f'y="{center_y - radius * fraction + 14:g}">{maximum * fraction:.3g}×</text>'
        )
    for angle, label in zip(angles, labels):
        axis_x, axis_y = point(angle, maximum)
        label_x = center_x + (radius + 34.0) * math.cos(angle)
        label_y = center_y + (radius + 34.0) * math.sin(angle)
        anchor = "middle" if abs(math.cos(angle)) < 0.2 else (
            "start" if math.cos(angle) > 0.0 else "end"
        )
        lines.append(
            f'<line x1="{center_x:g}" y1="{center_y:g}" x2="{axis_x:.2f}" '
            f'y2="{axis_y:.2f}" stroke="#d0d5dd" stroke-width="1"/>'
        )
        lines.append(
            f'<text class="label" x="{label_x:.2f}" y="{label_y:.2f}" '
            f'text-anchor="{anchor}" dominant-baseline="middle">'
            f'{escape(label.replace("_", " "))}</text>'
        )
    polygon = " ".join(
        f"{x:.2f},{y:.2f}"
        for x, y in (point(angle, ratio) for angle, ratio in zip(angles, ratios))
    )
    lines.extend((
        f'<polygon points="{polygon}" fill="#2563eb" fill-opacity="0.2" '
        'stroke="#2563eb" stroke-width="3"/>',
        '</svg>',
    ))
    destination = Path(output_path)
    destination.write_text("\n".join(lines) + "\n")
    return destination
