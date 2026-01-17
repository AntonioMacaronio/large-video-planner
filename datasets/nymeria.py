"""
Nymeria Video Dataset for finetuning video prediction models.

This dataset loads Nymeria egocentric video data which consists of:
- MP4 video files (egocentric RGB views, typically 1408x1408)
- HDF5 files containing motion data and atomic action labels

The atomic_action from the HDF5 files is used as the caption/prompt for video generation.

Usage:
    1. First generate the metadata CSV using scripts/generate_nymeria_csv.py:
       python scripts/generate_nymeria_csv.py --data-dir /path/to/nymeria --output /path/to/nymeria/metadata.csv

    2. Optionally precompute T5 embeddings using scripts/precompute_embeddings.py:
       python scripts/precompute_embeddings.py --csv-path /path/to/nymeria/metadata.csv --output-dir /path/to/nymeria/embeddings

    3. Train with:
       python main.py dataset=nymeria dataset.data_root=/path/to/nymeria algorithm=wan_i2v
"""

from pathlib import Path
from typing import Any, Dict

from .video_base import VideoDataset


class NymeriaVideoDataset(VideoDataset):
    """
    Nymeria Video Dataset for egocentric human motion video prediction.

    Extends the base VideoDataset to handle Nymeria-specific data format.
    Expects a metadata CSV with columns:
        - video_path: relative path to mp4 file
        - caption: atomic action description from h5 file
        - height, width, fps, n_frames: video metadata
        - split: "training" or "validation"
        - prompt_embed_path (optional): path to precomputed T5 embeddings
    """

    def preprocess_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess a record before loading.

        For Nymeria, we may want to override fps if the config specifies fps_override,
        since Nymeria videos may have varying framerates.
        """
        # Override fps if specified in config (useful for normalizing variable fps videos)
        if hasattr(self.cfg, 'fps_override') and self.cfg.fps_override is not None:
            record["fps"] = self.cfg.fps_override

        return record

    def download(self):
        """
        Nymeria data download is not automated.
        Users should prepare data and generate metadata CSV manually.
        """
        raise NotImplementedError(
            "Nymeria dataset download is not automated. "
            "Please prepare your data directory with h5/mp4 file pairs, then run:\n"
            "  python scripts/generate_nymeria_csv.py --data-dir /path/to/nymeria --output /path/to/nymeria/metadata.csv\n"
            "See datasets/nymeria.py docstring for full instructions."
        )
