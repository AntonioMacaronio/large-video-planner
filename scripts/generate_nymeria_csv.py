"""
Generate metadata CSV for Nymeria dataset.

This script scans the Nymeria data directory for h5/mp4 file pairs and generates
a CSV file with the required columns for training.

Usage:
    conda activate ei_world_model
    python scripts/generate_nymeria_csv.py --data-dir /data/nymeria --output /data/nymeria/metadata.csv

The output CSV will have columns:
    - video_path: filename of the mp4 file (relative to data_root)
    - caption: the atomic_action from the h5 file
    - height, width, fps, n_frames: video metadata
    - split: "training" or "validation" (90/10 split by default)
"""

from dataclasses import dataclass
from pathlib import Path

import h5py
import hdf5plugin  # Required for LZ4-compressed HDF5 files
import pandas as pd
import tyro
from decord import VideoReader, cpu
from tqdm import tqdm


def get_video_metadata(mp4_path: Path) -> dict:
    """Get video metadata from mp4 file using decord."""
    vr = VideoReader(str(mp4_path), ctx=cpu(0))
    return {
        "n_frames": len(vr),
        "height": vr[0].shape[0],
        "width": vr[0].shape[1],
        "fps": vr.get_avg_fps(),
    }


def get_h5_metadata(h5_path: Path) -> dict:
    """Get metadata from h5 file."""
    with h5py.File(h5_path, "r") as f:
        return {
            "atomic_action": f.attrs.get("atomic_action", ""),
            "num_frames": f.attrs.get("num_frames", 0),
            "sequence_name": f.attrs.get("sequence_name", ""),
            "egoview_mp4_filename": f.attrs.get("egoview_mp4_filename", ""),
        }


def generate_csv(
    data_dir: Path,
    output_path: Path,
    test_percentage: float = 0.1,
    use_h5_frames: bool = True,
    sample_video_metadata: bool = True,
):
    """
    Generate metadata CSV for Nymeria dataset.

    Args:
        data_dir: Directory containing h5 and mp4 files
        output_path: Path to save the output CSV
        test_percentage: Percentage of data to use for validation (default: 10%)
        use_h5_frames: If True, use num_frames from h5 file; if False, read from mp4
        sample_video_metadata: If True, sample one video to get height/width/fps
                               (faster, assumes all videos have same dimensions)
    """
    # Find all h5 files
    h5_files = sorted(data_dir.glob("*.h5"))
    print(f"Found {len(h5_files)} h5 files in {data_dir}")

    if len(h5_files) == 0:
        raise ValueError(f"No h5 files found in {data_dir}")

    # Sample video metadata from first file (assuming all are same dimensions)
    if sample_video_metadata:
        first_mp4 = h5_files[0].with_suffix(".mp4")
        if first_mp4.exists():
            sample_meta = get_video_metadata(first_mp4)
            default_height = sample_meta["height"]
            default_width = sample_meta["width"]
            default_fps = sample_meta["fps"]
            print(f"Sampled video metadata: {default_height}x{default_width} @ {default_fps} fps")
        else:
            raise ValueError(f"Sample mp4 not found: {first_mp4}")

    records = []
    skipped = 0

    for h5_path in tqdm(h5_files, desc="Processing files"):
        mp4_path = h5_path.with_suffix(".mp4")

        # Skip if mp4 doesn't exist
        if not mp4_path.exists():
            skipped += 1
            continue

        # Get h5 metadata
        h5_meta = get_h5_metadata(h5_path)

        # Get video metadata
        if sample_video_metadata:
            height, width, fps = default_height, default_width, default_fps
        else:
            video_meta = get_video_metadata(mp4_path)
            height, width, fps = video_meta["height"], video_meta["width"], video_meta["fps"]

        # Use n_frames from h5 or mp4
        if use_h5_frames:
            n_frames = h5_meta["num_frames"]
        else:
            if sample_video_metadata:
                # Need to read this specific file
                video_meta = get_video_metadata(mp4_path)
            n_frames = video_meta["n_frames"]

        record = {
            "video_path": mp4_path.name,  # Just the filename, relative to data_root
            "caption": h5_meta["atomic_action"],
            "height": height,
            "width": width,
            "fps": fps,
            "n_frames": n_frames,
            "sequence_name": h5_meta["sequence_name"],
        }
        records.append(record)

    print(f"Processed {len(records)} files, skipped {skipped} (missing mp4)")

    # Create DataFrame and assign splits
    df = pd.DataFrame(records)

    # Shuffle with fixed seed for reproducibility
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Assign splits
    n_validation = int(len(df) * test_percentage)
    df["split"] = "training"
    df.loc[df.index[-n_validation:], "split"] = "validation"

    print(f"Split: {len(df) - n_validation} training, {n_validation} validation")

    # Save CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved CSV to {output_path}")

    return df


@dataclass
class Config:
    """Generate Nymeria metadata CSV."""

    data_dir: Path = Path("/data/nymeria")
    """Directory containing h5 and mp4 files."""

    output: Path = Path("/data/nymeria/metadata.csv")
    """Output CSV path."""

    test_percentage: float = 0.1
    """Percentage of data for validation."""

    read_all_videos: bool = False
    """Read metadata from each video file (slower but more accurate)."""


def main(cfg: Config):
    generate_csv(
        data_dir=cfg.data_dir,
        output_path=cfg.output,
        test_percentage=cfg.test_percentage,
        sample_video_metadata=not cfg.read_all_videos,
    )


if __name__ == "__main__":
    main(tyro.cli(Config))
