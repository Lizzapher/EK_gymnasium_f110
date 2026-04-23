import argparse
import json
import os
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description="Click-to-JSON for F1Tenth: adjust for image-origin vs. world-origin."
    )
    parser.add_argument("--image", required=True,
                        help="Path to the track PNG.")
    parser.add_argument("--n_checkpoints", type=int, required=True,
                        help="Number of checkpoint segments (2 clicks each).")
    parser.add_argument("--resolution", type=float, default=0.025,
                        help="Meters per pixel.")
    parser.add_argument("--origin", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                        help="World (x, y, θ) that corresponds to the bottom-left corner of the image.")
    parser.add_argument("--name", type=str, default="interactive_map",
                        help="Track name in JSON.")
    parser.add_argument("--output", required=True,
                        help="Where to write the output JSON.")
    args = parser.parse_args()

    # Load the image to get its height in pixels
    img_arr = plt.imread(args.image)
    H_px = img_arr.shape[0]   # number of rows = image height in pixels

    # Display the image and collect raw pixel clicks
    fig, ax = plt.subplots()
    ax.imshow(img_arr)
    ax.set_title(f"Click {2 * args.n_checkpoints} points (2 per segment)")
    pts = plt.ginput(2 * args.n_checkpoints, timeout=0)
    plt.close(fig)

    if len(pts) != 2 * args.n_checkpoints:
        print(f"Error: expected {2 * args.n_checkpoints} clicks but got {len(pts)}.")
        return

    origin_x, origin_y, _ = args.origin
    res = args.resolution

    # Convert each clicked pixel-coordinate into world-meter coordinates
    segments = []
    midpoints = []
    for i in range(args.n_checkpoints):
        x1_pix, y1_pix = pts[2 * i]
        x2_pix, y2_pix = pts[2 * i + 1]

        # Flip y coordinate and scale
        x1_m = origin_x + x1_pix * res
        y1_m = origin_y + (H_px - y1_pix) * res
        x2_m = origin_x + x2_pix * res
        y2_m = origin_y + (H_px - y2_pix) * res

        segments.append([[float(x1_m), float(y1_m)], [float(x2_m), float(y2_m)]])
        midpoints.append([ (x1_m + x2_m) / 2.0, (y1_m + y2_m) / 2.0 ])

    # Build JSON data structure
    track_json = {
        "track_info": {
            "name": args.name,
            "resolution": args.resolution,
            "origin": [origin_x, origin_y, 0.0]
        },
        # Minimal centerline
        "centerline": midpoints,
        # Checkpoints: list of 2-point segments in world coords
        "checkpoints": segments
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(track_json, f, indent=4)

    print(f"Wrote JSON with {len(segments)} checkpoints to {args.output}")

    with open(f"RaceTracks/{args.name}.json", "r") as f:
        data = json.load(f)

    origin_x, origin_y, _ = data["track_info"]["origin"]
    res = data["track_info"]["resolution"]

    img = plt.imread(f"RaceTracks/{args.name}.png")
    H_px = img.shape[0]

    pixel_points = []
    for (x_w, y_w) in data["centerline"]:
        x_pix = (x_w - origin_x) / res
        y_pix = H_px - ((y_w - origin_y) / res)
        pixel_points.append((x_pix, y_pix))

    for seg in data["checkpoints"]:
        for (x_w, y_w) in seg:
            x_pix = (x_w - origin_x) / res
            y_pix = H_px - ((y_w - origin_y) / res)
            pixel_points.append((x_pix, y_pix))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img)
    xs, ys = zip(*pixel_points)
    ax.scatter(xs, ys, c="red", s=30, marker="x")
    ax.set_title("Red X Correspond to User Selected Checkpoints with Calcualted Centre Points")
    os.makedirs('./RaceTracks', exist_ok=True)
    plt.savefig(f"./RaceTracks/{args.name}_checkpoints.png")

if __name__ == "__main__":
    main()