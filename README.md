# generate_aruco

Generate a print-ready PDF sheet of [ArUco](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html)
markers at exact physical size (mm), so you can print, cut, and glue them onto
objects.

This was built to make marker sets for a **hexagonal fiducial** — six
markers glued onto the six faces of a small hexagonal prism, placed under or
around a miniature. Filming from above/at an angle, whichever face(s) are
visible tells your post-processing pipeline (e.g. `cv2.aruco.detectMarkers`)
which way the miniature is oriented.

## Install

```bash
pip install -r requirements.txt
```

## Usage

```bash
python generate_aruco.py --size <mm> --count <n> --copies <n> --out <file.pdf>
```

| Flag | Meaning | Default |
|---|---|---|
| `--size` | Marker side length in mm (prints at true scale — no "fit to page") | *required* |
| `--count` | Markers per set (e.g. `6` for one hexagon) | *required* |
| `--copies` | How many full sets to print (e.g. one per physical hexagon) | `1` |
| `--dict` | ArUco dictionary: `4X4_50`, `5X5_50`, `6X6_50`, `4X4_250`, ... | `4X4_50` |
| `--start-id` | First marker ID to use | `0` |
| `--unique-ids` | Give every printed marker a distinct ID instead of reusing the same `count` IDs per copy | off |
| `--page` | `a4` or `letter` | `a4` |
| `--out` | Output PDF path | `markers.pdf` |

The black marker square measures exactly `--size` mm (this is the marker
length to give your pose-estimation code). Each marker also gets a white
quiet zone of 12% of `--size` on every side, a dashed cut guide around that
quiet zone (so the cut-out square is about `1.24 × --size`), and a small ID
label underneath. **Print at 100% scale** (disable "fit to page"
or "scale to fit" in your print dialog) or the physical size will be wrong.

## Example: one hexagon, 20mm faces

```bash
python generate_aruco.py --size 20 --count 6 --copies 1 --out example_hexagon_set.pdf
```

This produces 6 markers, IDs `0`–`5`, laid out on one A4 page:

![example output](example_preview.png)

Cut out each square along the dashed line and glue one to each face of the
hexagonal prism, in a fixed, known order (e.g. ID 0 = "front", going
clockwise). Because every hexagon printed this way reuses IDs `0`–`5`, your
detection code can hard-code that mapping once and reuse it for every
miniature base.

### Multiple hexagons

If you're tracking three miniatures, print three sets of the same 6 IDs:

```bash
python generate_aruco.py --size 20 --count 6 --copies 3 --out markers.pdf
```

Each hexagon still uses IDs `0`–`5` — you'd distinguish *which* hexagon is
which by pairing it with a video-tracking ID (e.g. from `cv2.Tracker` or
just spatial continuity between frames), not from the ArUco ID itself.

If instead you need every glued marker to be globally unique (e.g. because
your pipeline identifies *which* miniature it is directly from the marker
ID, not just its orientation), add `--unique-ids`:

```bash
python generate_aruco.py --size 20 --count 6 --copies 3 --unique-ids --out markers.pdf
# hexagon 1 -> IDs 0-5, hexagon 2 -> IDs 6-11, hexagon 3 -> IDs 12-17
```

## Choosing a dictionary and size

- Smaller/closer markers (small hexagon faces, or camera close to the
  table) → a smaller dictionary like `4X4_50` needs fewer pixels to decode
  reliably.
- Markers likely to be partially occluded or filmed at a distance → a
  larger dictionary (`5X5_50`, `6X6_50`) trades marker complexity for
  robustness, but needs more resolution to read.
- Test detection at your actual filming distance and resolution before
  committing to a size — `cv2.aruco.detectMarkers` will tell you quickly
  whether a given size/dictionary combination is reliable.

## Orientation tracking notes

With markers glued flat on the 6 vertical faces of a straight hexagonal
prism, a camera looking straight down will only see the marker(s) nearly
edge-on or not at all. Two options:

- Angle the camera so it's looking somewhat down *and* across the table
  (most practical setups already do this).
- Use a truncated hexagonal **pyramid** instead of a straight prism, so
  each face tilts outward toward the camera and stays visible across a
  wider range of table positions.
