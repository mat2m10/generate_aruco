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
# White quiet zone kept around each marker, as a fraction of the marker side.
# It is left blank on the page (inside the cut guide) rather than baked into
# the marker bitmap, so the black marker itself prints at exactly --size mm.
QUIET_ZONE_FRAC = 0.12


def _positive(type_):
    def check(value):
        v = type_(value)
        if not v > 0:
            raise argparse.ArgumentTypeError(f"must be > 0, got {value}")
        return v

    return check


def _non_negative_int(value):
    v = int(value)
    if v < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {value}")
    return v


def generate_marker_image(dictionary, marker_id: int, size_mm: float) -> Image.Image:
    """Render one ArUco marker (black outer edge to black outer edge, no
    quiet zone) as a PIL image. The caller draws it at exactly size_mm."""
    side_px = max(int(round(size_mm * PX_PER_MM)), 100)
    marker_px = aruco.generateImageMarker(dictionary, marker_id, side_px)
    return Image.fromarray(marker_px).convert("L")


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

    # Each cell holds the marker plus its white quiet zone on every side (the
    # area inside the cut guide), then the label and the gap to the next cell.
    quiet_mm = size_mm * QUIET_ZONE_FRAC
    outer_mm = size_mm + 2 * quiet_mm  # cut-guide square
    cell_mm = outer_mm + GAP_MM
    cell_h_mm = outer_mm + LABEL_HEIGHT_MM + GAP_MM
    usable_w = (page_w / mm) - 2 * PAGE_MARGIN_MM
    usable_h = (page_h / mm) - 2 * PAGE_MARGIN_MM
    # One cell without its trailing gap must fit, or the marker would run
    # off the page.
    if cell_mm - GAP_MM > usable_w or cell_h_mm - GAP_MM > usable_h:
        raise ValueError(
            f"a {size_mm:g}mm marker (plus label) does not fit in the "
            f"{usable_w:.0f}x{usable_h:.0f}mm printable area of "
            f"{page_size_name.upper()} paper. Use a smaller --size."
        )
    cols = max(int(usable_w // cell_mm), 1)
    rows = max(int(usable_h // cell_h_mm), 1)
    per_page = cols * rows

    c = canvas.Canvas(out_path, pagesize=PAGE_SIZES[page_size_name])

    # Available horizontal space for each marker's label, in points. Labels
    # must never be wider than this or they run into the neighbouring
    # marker's label (this is what caused overlapping text at small sizes).
    avail_label_w_pt = (cell_mm - 1) * mm

    def fit_label(candidates):
        """Return (text, font_size) for the most informative candidate string
        that fits avail_label_w_pt, shrinking the font down to 3pt before
        falling back to a shorter candidate."""
        for text in candidates:
            size = 6.0
            while size >= 3.0:
                if c.stringWidth(text, "Helvetica", size) <= avail_label_w_pt:
                    return text, size
                size -= 0.5
        # Nothing fit even at 3pt (extremely tiny marker) - truncate the
        # shortest candidate as a last resort.
        text = candidates[-1]
        while len(text) > 1 and c.stringWidth(text, "Helvetica", 3.0) > avail_label_w_pt:
            text = text[:-1]
        return text, 3.0

    header = f"{dict_name}, {size_mm:.1f}mm markers"
    if not unique_ids and copies > 1:
        header += f"  (IDs 0-{count - 1} repeat per set; set number shown as ·Sn)"

    def draw_header():
        c.setFont("Helvetica", 7)
        c.setFillColorRGB(0.3, 0.3, 0.3)
        c.drawString(PAGE_MARGIN_MM * mm, (page_h / mm - PAGE_MARGIN_MM + 3) * mm, header)

    draw_header()

    for idx, (copy_idx, marker_id) in enumerate(jobs):
        pos_on_page = idx % per_page
        if idx > 0 and pos_on_page == 0:
            c.showPage()
            draw_header()
        row = pos_on_page // cols
        col = pos_on_page % cols

        # (x_mm, y_mm) is the bottom-left corner of the black marker itself.
        x_mm = PAGE_MARGIN_MM + col * cell_mm + quiet_mm
        # y measured from top of page down, then flipped for reportlab's
        # bottom-left origin
        y_from_top_mm = PAGE_MARGIN_MM + row * cell_h_mm + quiet_mm
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

        # Cut guide around the marker, enclosing its white quiet zone
        c.setLineWidth(0.2)
        c.setStrokeColorRGB(0.6, 0.6, 0.6)
        c.setDash(2, 2)
        c.rect(
            (x_mm - quiet_mm) * mm,
            (y_mm - quiet_mm) * mm,
            outer_mm * mm,
            outer_mm * mm,
            stroke=1,
            fill=0,
        )
        c.setDash()

        # Label: marker ID (+ compact set indicator when reusing IDs per
        # copy). Candidates are ordered most-to-least informative; fit_label
        # picks the first that fits the cell, shrinking font size first and
        # falling back to a shorter string only if it still doesn't fit.
        candidates = []
        if not unique_ids and copies > 1:
            candidates.append(f"ID {marker_id} \u00b7S{copy_idx + 1}")
        candidates.append(f"ID {marker_id}")
        candidates.append(f"{marker_id}")
        label, font_size = fit_label(candidates)

        c.setFont("Helvetica", font_size)
        c.setFillColorRGB(0, 0, 0)
        c.drawCentredString(
            (x_mm + size_mm / 2) * mm,
            (y_mm - quiet_mm - LABEL_HEIGHT_MM + 1) * mm,
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
        "--size", type=_positive(float), required=True, help="Marker side length in mm (print size)."
    )
    p.add_argument(
        "--count",
        type=_positive(int),
        required=True,
        help="Number of markers per set (e.g. 6 for a hexagon's faces).",
    )
    p.add_argument(
        "--copies",
        type=_positive(int),
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
        "--start-id", type=_non_negative_int, default=0, help="First marker ID to use. Default: 0."
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
    # Number of IDs the dictionary really holds (50/100/250, 1024 for ORIGINAL).
    dictionary = aruco.getPredefinedDictionary(DICTIONARIES[args.dict_name])
    n_ids = len(dictionary.bytesList)
    # With --unique-ids every printed marker gets its own ID.
    n_needed = args.count * (args.copies if args.unique_ids else 1)
    if args.start_id + n_needed > n_ids:
        what = (
            f"--count {args.count} x --copies {args.copies} (--unique-ids)"
            if args.unique_ids
            else f"--count {args.count}"
        )
        print(
            f"Error: {args.dict_name} only has IDs 0-{n_ids - 1}, but "
            f"--start-id {args.start_id} + {what} needs IDs up to "
            f"{args.start_id + n_needed - 1}. Pick a dictionary with more IDs "
            "(*_100, *_250 or ORIGINAL) or use fewer IDs.",
            file=sys.stderr,
        )
        return 1
    try:
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
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())