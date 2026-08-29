#!/usr/bin/env python3
"""
generate_aruco.py

Generate a print-ready PDF sheet of ArUco markers, sized to physical
dimensions (mm), for gluing onto objects (e.g. the 6 faces of a hexagonal
prism used as an orientation fiducial for tracking miniatures).

Example
-------
Six 20mm markers (one hexagon's worth), 3 copies (3 hexagons), on A4:

    python generate_aruco.py --size 20 --count 6 --copies 3 --out markers.pdf

Marker IDs used are 0..count-1, repeated identically for each copy, so that
every physical hexagon uses the *same* 6 IDs in the *same* order — this is
what you want if your post-processing code identifies faces by ID and infers
orientation from which face(s) are visible.

Use --unique-ids instead if you want every printed marker to have a distinct
ID (e.g. for tracking several different miniatures at once).
"""

import argparse
import io
import sys

import cv2.aruco as aruco
from PIL import Image
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

DICTIONARIES = {
    "4X4_50": aruco.DICT_4X4_50,
    "4X4_100": aruco.DICT_4X4_100,
    "4X4_250": aruco.DICT_4X4_250,
    "5X5_50": aruco.DICT_5X5_50,
    "5X5_100": aruco.DICT_5X5_100,
    "5X5_250": aruco.DICT_5X5_250,
    "6X6_50": aruco.DICT_6X6_50,
    "6X6_100": aruco.DICT_6X6_100,
    "6X6_250": aruco.DICT_6X6_250,
    "ORIGINAL": aruco.DICT_ARUCO_ORIGINAL,
}

PAGE_SIZES = {"a4": A4, "letter": letter}

PX_PER_MM = 12  # marker raster resolution before embedding in the PDF
LABEL_HEIGHT_MM = 4.5  # space reserved below each marker for its ID text
GAP_MM = 4  # gap between adjacent marker cells
PAGE_MARGIN_MM = 10


def generate_marker_image(dictionary, marker_id: int, size_mm: float) -> Image.Image:
    """Render one ArUco marker (with a 1-module quiet-zone border) as a PIL image."""
    side_px = max(int(round(size_mm * PX_PER_MM)), 100)
    marker_px = aruco.generateImageMarker(dictionary, marker_id, side_px)
    img = Image.fromarray(marker_px).convert("L")
    # Add a white quiet-zone border (~10% of marker size on each side) so the
    # marker is reliably detectable right up to where it's cut/glued.
    border = max(int(side_px * 0.12), 4)
    bordered = Image.new("L", (side_px + 2 * border, side_px + 2 * border), 255)
    bordered.paste(img, (border, border))
    return bordered


def build_pdf(
    out_path: str,
    size_mm: float,
    count: int,
    copies: int,
    dict_name: str,
    unique_ids: bool,
    page_size_name: str,
    start_id: int,
):
    dictionary = aruco.getPredefinedDictionary(DICTIONARIES[dict_name])
    page_w, page_h = PAGE_SIZES[page_size_name]

    # Build the ordered list of (copy_index, marker_id) to place on the sheet.
    if unique_ids:
        total = count * copies
        ids = list(range(start_id, start_id + total))
        jobs = [(i // count, mid) for i, mid in enumerate(ids)]
    else:
        ids = list(range(start_id, start_id + count))
        jobs = [(c, mid) for c in range(copies) for mid in ids]

    cell_mm = size_mm + GAP_MM
    cell_h_mm = size_mm + LABEL_HEIGHT_MM + GAP_MM
    usable_w = (page_w / mm) - 2 * PAGE_MARGIN_MM
    usable_h = (page_h / mm) - 2 * PAGE_MARGIN_MM
    cols = max(int(usable_w // cell_mm), 1)
    rows = max(int(usable_h // cell_h_mm), 1)
    per_page = cols * rows

    c = canvas.Canvas(out_path, pagesize=PAGE_SIZES[page_size_name])

    for idx, (copy_idx, marker_id) in enumerate(jobs):
        pos_on_page = idx % per_page
        if idx > 0 and pos_on_page == 0:
            c.showPage()
        row = pos_on_page // cols
        col = pos_on_page % cols

        x_mm = PAGE_MARGIN_MM + col * cell_mm
        # y measured from top of page down, then flipped for reportlab's
        # bottom-left origin
        y_from_top_mm = PAGE_MARGIN_MM + row * cell_h_mm
        y_mm = (page_h / mm) - y_from_top_mm - size_mm

        img = generate_marker_image(dictionary, marker_id, size_mm)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        c.drawImage(
            ImageReader(buf),
            x_mm * mm,
            y_mm * mm,
            width=size_mm * mm,
            height=size_mm * mm,
        )

        # Cut guide around the marker
        c.setLineWidth(0.2)
        c.setStrokeColorRGB(0.6, 0.6, 0.6)
        c.setDash(2, 2)
        c.rect(x_mm * mm, y_mm * mm, size_mm * mm, size_mm * mm, stroke=1, fill=0)
        c.setDash()

        # Label: marker ID (+ copy/face number when reusing IDs per copy)
        label = f"ID {marker_id}"
        if not unique_ids and copies > 1:
            label += f"  ({dict_name}, set {copy_idx + 1})"
        c.setFont("Helvetica", 6)
        c.setFillColorRGB(0, 0, 0)
        c.drawCentredString(
            (x_mm + size_mm / 2) * mm,
            (y_mm - LABEL_HEIGHT_MM + 1) * mm,
            label,
        )

    c.save()

    n_pages = max(1, -(-len(jobs) // per_page))
    print(
        f"Wrote {len(jobs)} marker(s) ({dict_name}, {size_mm:.1f}mm each) "
        f"across {n_pages} page(s) -> {out_path}"
    )


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Generate a print-ready PDF sheet of ArUco markers at exact physical size."
    )
    p.add_argument(
        "--size", type=float, required=True, help="Marker side length in mm (print size)."
    )
    p.add_argument(
        "--count",
        type=int,
        required=True,
        help="Number of markers per set (e.g. 6 for a hexagon's faces).",
    )
    p.add_argument(
        "--copies",
        type=int,
        default=1,
        help="How many copies of the set to print (e.g. one per physical hexagon). Default: 1.",
    )
    p.add_argument(
        "--dict",
        dest="dict_name",
        default="4X4_50",
        choices=sorted(DICTIONARIES.keys()),
        help="ArUco dictionary to use. Default: 4X4_50.",
    )
    p.add_argument(
        "--start-id", type=int, default=0, help="First marker ID to use. Default: 0."
    )
    p.add_argument(
        "--unique-ids",
        action="store_true",
        help="Give every printed marker a distinct ID instead of repeating the same "
        "count IDs for each copy (use this if you're tracking several different "
        "objects and need every marker to be globally unique).",
    )
    p.add_argument(
        "--page",
        dest="page_size_name",
        default="a4",
        choices=sorted(PAGE_SIZES.keys()),
        help="Page size. Default: a4.",
    )
    p.add_argument(
        "--out", default="markers.pdf", help="Output PDF path. Default: markers.pdf"
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    max_id = 50
    if "50" in args.dict_name:
        max_id = 50
    elif "100" in args.dict_name:
        max_id = 100
    elif "250" in args.dict_name:
        max_id = 250
    if not args.unique_ids and args.start_id + args.count > max_id:
        print(
            f"Error: {args.dict_name} only has IDs 0-{max_id - 1}, but "
            f"--start-id {args.start_id} + --count {args.count} exceeds that. "
            "Pick a larger dictionary (e.g. --dict 4X4_250) or reduce --count.",
            file=sys.stderr,
        )
        return 1
    build_pdf(
        out_path=args.out,
        size_mm=args.size,
        count=args.count,
        copies=args.copies,
        dict_name=args.dict_name,
        unique_ids=args.unique_ids,
        page_size_name=args.page_size_name,
        start_id=args.start_id,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
