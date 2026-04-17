"""Verification/demo script for the minicv milestone-one package."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import minicv as cv

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def build_synthetic_images() -> tuple[np.ndarray, np.ndarray]:
    h, w = 256, 256
    y = np.linspace(0, 255, h, dtype=np.float32)[:, None]
    x = np.linspace(0, 255, w, dtype=np.float32)[None, :]
    base = 0.45 * x + 0.55 * y
    gray = np.clip(base, 0, 255).astype(np.float32)
    yy, xx = np.mgrid[:h, :w]
    circle = (xx - 70) ** 2 + (yy - 80) ** 2 <= 35**2
    square = (xx > 150) & (xx < 220) & (yy > 40) & (yy < 110)
    diagonal = np.abs(yy - (0.8 * xx + 20)) < 3
    gray[circle] = 220
    gray[square] = 40
    gray[diagonal] = 255
    rgb = np.stack([gray, np.roll(gray, 25, axis=1), np.roll(gray, -25, axis=0)], axis=-1)
    return gray, rgb


def save_panel(filename: str, images: list[np.ndarray], titles: list[str], cmap: str | None = None) -> None:
    cols = min(3, len(images))
    rows = int(np.ceil(len(images) / cols))
    plt.figure(figsize=(5 * cols, 4 * rows))
    for i, (img, title) in enumerate(zip(images, titles), start=1):
        plt.subplot(rows, cols, i)
        if img.ndim == 2:
            plt.imshow(img, cmap=cmap or "gray")
        else:
            plt.imshow(np.clip(img, 0, 255).astype(np.uint8))
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(RESULTS / filename, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    gray, rgb = build_synthetic_images()
    cv.save_image(RESULTS / "input_gray.png", gray)
    cv.save_image(RESULTS / "input_rgb.png", rgb)

    gray_loaded = cv.read_image(RESULTS / "input_gray.png", mode="grayscale")
    rgb_loaded = cv.read_image(RESULTS / "input_rgb.png", mode="rgb")
    gray_from_rgb = cv.rgb_to_grayscale(rgb_loaded)
    rgb_from_gray = cv.grayscale_to_rgb(gray_loaded)
    save_panel("01_io_color.png", [gray_loaded, rgb_loaded, gray_from_rgb, rgb_from_gray], ["gray loaded", "rgb loaded", "rgb to gray", "gray to rgb"])

    minmax = cv.normalize_image(gray, mode="minmax")
    zscore = cv.normalize_image(gray, mode="zscore")
    unit = cv.normalize_image(gray, mode="unit") * 255.0
    padded = cv.pad_image(gray, 10, mode="reflect")
    kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    conv = cv.convolve2d(gray, kernel)
    save_panel("02_core_ops.png", [minmax, zscore, unit, padded, conv], ["normalize minmax", "normalize zscore", "normalize unit", "padding reflect", "convolution"])

    mean_img = cv.mean_filter(gray, 5)
    gauss_img = cv.gaussian_filter(gray, 7, 1.4)
    median_img = cv.median_filter(gray, 5)
    t_global = cv.global_threshold(gray, 120)
    otsu_t, t_otsu = cv.otsu_threshold(gray)
    t_adapt = cv.adaptive_threshold(gray, block_size=15, c=5, method="gaussian")
    _, _, mag, _ = cv.sobel_gradients(gray)
    eq = cv.histogram_equalization(gray)
    sharp = cv.unsharp_mask(gray, kernel_size=7, sigma=1.2, amount=1.2)
    save_panel("03_filters.png", [mean_img, gauss_img, median_img, t_global, t_otsu, t_adapt, mag, eq, sharp], ["mean", "gaussian", "median", "global threshold", f"otsu ({otsu_t:.0f})", "adaptive threshold", "sobel magnitude", "hist equalization", "unsharp mask"])

    planes = cv.bit_plane_slicing(gray)
    save_panel("04_bit_planes.png", [planes[i] * 255 for i in range(8)], [f"bit plane {i}" for i in range(8)])
    hist, edges = cv.histogram(gray)
    plt.figure(figsize=(7, 4))
    plt.bar(edges[:-1], hist, width=np.diff(edges), align="edge")
    plt.title("Grayscale histogram")
    plt.xlabel("Intensity")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(RESULTS / "05_histogram.png", dpi=150)
    plt.close()

    resized_nn = cv.resize(rgb, (180, 180), interpolation="nearest")
    resized_bl = cv.resize(rgb, (180, 180), interpolation="bilinear")
    rotated = cv.rotate(rgb, 30, interpolation="bilinear")
    translated = cv.translate(rgb, tx=25, ty=35, interpolation="bilinear")
    save_panel("06_transforms.png", [rgb, resized_nn, resized_bl, rotated, translated], ["original rgb", "resize nearest", "resize bilinear", "rotate 30", "translate"])

    feature_tables = {
        "intensity_statistics": cv.intensity_statistics(gray),
        "hist_descriptor": cv.grayscale_histogram_descriptor(gray, bins=16),
        "gradient_statistics": cv.gradient_statistics(gray),
        "hog_descriptor": cv.hog_descriptor(gray, cell_size=16, bins=9),
    }
    pd.concat(feature_tables, axis=0).to_csv(RESULTS / "feature_summary.csv", header=["value"])

    canvas = cv.create_canvas(300, 420, channels=3, color=(20, 20, 20))
    canvas = cv.draw_point(canvas, 40, 40, color=(255, 0, 0), thickness=7)
    canvas = cv.draw_line(canvas, (20, 260), (180, 40), color=(0, 255, 0), thickness=4)
    canvas = cv.draw_rectangle(canvas, (210, 30), (380, 120), color=(255, 255, 0), thickness=3)
    canvas = cv.draw_rectangle(canvas, (220, 150), (390, 260), color=(30, 144, 255), filled=True)
    canvas = cv.draw_polygon(canvas, [(70, 160), (140, 110), (180, 180), (100, 240)], color=(255, 0, 255), thickness=3)
    canvas = cv.draw_polygon(canvas, [(250, 180), (300, 130), (360, 160), (340, 230), (270, 240)], color=(255, 128, 0), filled=True)
    canvas = cv.put_text(canvas, "MINICV DEMO", (20, 10), font_scale=2, color=(255, 255, 255), spacing=2)
    cv.save_image(RESULTS / "07_drawing_text.png", canvas)
    save_panel("08_overview.png", [gray, gray_from_rgb, mag, eq, rotated, canvas], ["input", "gray from rgb", "sobel magnitude", "equalized", "rotated", "drawing canvas"])
    print("MiniCV demo completed successfully.")


if __name__ == "__main__":
    main()
