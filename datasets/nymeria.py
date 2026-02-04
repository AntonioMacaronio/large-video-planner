"""
Nymeria Video Dataset for finetuning video prediction models.

This dataset loads Nymeria egocentric video data which consists of:
- MP4 video files (egocentric RGB views, typically 1408x1408)
- HDF5 files containing motion data and atomic action labels

The atomic_action from the HDF5 files is used as the caption/prompt for video generation.

Usage:
    1. First generate the metadata CSV using scripts/generate_nymeria_csv.py:
       python scripts/generate_nymeria_csv.py --data-dir /path/to/nymeria --output /path/to/nymeria/metadata.csv
       python scripts/generate_nymeria_csv.py --data-dir /data/nymeria --output /data/nymeria/metadata.csv

    2. Optionally precompute T5 embeddings using scripts/precompute_embeddings.py:
       python scripts/precompute_embeddings.py --csv-path /path/to/nymeria/metadata.csv --output-dir /path/to/nymeria/embeddings

    3. Train with:
       python main.py dataset=nymeria dataset.data_root=/path/to/nymeria algorithm=wan_i2v
"""

from pathlib import Path
from typing import Any, Dict

import h5py
import hdf5plugin  # Required for reading LZ4-compressed HDF5 files
import numpy as np
import torch

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

    When load_actions=True, also loads motion data from corresponding HDF5 files:
        - root_translation: (n_frames, 3)
        - root_orientation: (n_frames, 3, 3)
        - cpf_translation: (n_frames, 3)
        - cpf_orientation: (n_frames, 3, 3)
        - joint_translation: (n_frames, 23, 3)
        - joint_orientation: (n_frames, 23, 3, 3)
        - contact_information: (n_frames, 4)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Check if load_actions is configured (defaults to False)
        self.load_actions = getattr(self.cfg, 'load_actions', False)

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

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single item from the dataset.

        Extends base class to optionally load actions from HDF5 files.
        """
        # Get base output from parent class
        output = super().__getitem__(idx)
        # Example output from VideoDataset for a Nymeria video (20230607_s0_james_johnson_act0_e72nhq_00000.mp4):
        # Original video: 1408x1408 @ 30fps, 128 frames
        # After processing with config (n_frames=49, height=480, width=832):
        # {
        #     "videos": torch.Tensor,       # shape: (49, 3, 480, 832), float32, normalized to [-1, 1]
        #     "video_metadata": {
        #         "num_frames": 49,
        #         "height": 480,
        #         "width": 832,
        #         "has_caption": True,
        #     },
        #     "bbox_render": torch.Tensor,  # shape: (2, 480, 832), float32
        #     "has_bbox": torch.Tensor,     # shape: (2,), bool, typically [False, False]
        #     "video_path": "/data/nymeria/20230607_s0_james_johnson_act0_e72nhq_00000.mp4",
        #     "prompts": "C is standing in the foyer while talking to her peer. C walks towards...",
        #     # Optional (if load_prompt_embed=True):
        #     # "prompt_embeds": torch.Tensor,  # shape: (max_text_tokens, 4096)
        #     # "prompt_embed_len": int,
        # }

        # Optionally load actions from HDF5 file
        if self.load_actions:
            record = self.records[idx]
            actions = self._load_actions(record)
            output["actions"] = actions
            # Example actions for 20230607_s0_james_johnson_act0_e72nhq_00000.h5:
            # HDF5 contains 128 frames of motion data, with config n_frames=49:
            # {
            #     "root_translation": torch.Tensor,     # shape: (49, 3), float32 - body root position
            #     "root_orientation": torch.Tensor,     # shape: (49, 3, 3), float32 - body root rotation matrix
            #     "cpf_translation": torch.Tensor,      # shape: (49, 3), float32 - camera pose (CPF) position
            #     "cpf_orientation": torch.Tensor,      # shape: (49, 3, 3), float32 - camera pose rotation matrix
            #     "joint_translation": torch.Tensor,    # shape: (49, 23, 3), float32 - 23 body joint positions
            #     "joint_orientation": torch.Tensor,    # shape: (49, 23, 3, 3), float32 - 23 body joint rotations
            #     "contact_information": torch.Tensor,  # shape: (49, 4), float32 - foot contact labels
            #     "padding_mask": torch.Tensor,         # shape: (49,), float32 - [1,1,1,...,1,0,0,...] if padded
            #     "valid_length": torch.Tensor,         # scalar, e.g. tensor(49) - actual frames before padding
            # }

        return output

    def _load_actions(self, record: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Load motion data from the HDF5 file corresponding to this video.

        The HDF5 file is expected to be at the same path as the video but with .h5 extension.
        Actions are loaded at their original length, then zero-padded to n_frames if shorter
        or trimmed if longer. A padding_mask indicates valid vs padded frames.

        Args:
            record: Dictionary containing video metadata including video_path

        Returns:
            Dictionary containing motion tensors, all with shape (n_frames, ...):
                - root_translation: (n_frames, 3)
                - root_orientation: (n_frames, 3, 3)
                - cpf_translation: (n_frames, 3)
                - cpf_orientation: (n_frames, 3, 3)
                - joint_translation: (n_frames, 23, 3)
                - joint_orientation: (n_frames, 23, 3, 3)
                - contact_information: (n_frames, 4)
                - padding_mask: (n_frames,) - 1.0 for valid frames, 0.0 for padded
                - valid_length: scalar tensor - number of valid (non-padded) frames
        """
        # Derive HDF5 path from video path
        video_path = self.data_root / record["video_path"]
        h5_path = video_path.with_suffix('.h5')

        if not h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

        T = self.n_frames  # Target sequence length

        # Load motion data from HDF5
        with h5py.File(h5_path, 'r') as f:
            # Get the actual number of frames in the HDF5 file
            num_frames_in_h5 = f.attrs.get('num_frames', f['root_translation'].shape[0])

            # Determine how many frames to load (min of available and target)
            valid_length = min(num_frames_in_h5, T)

            # Load only valid frames (up to T)
            root_translation = f['root_translation'][:valid_length]        # (valid_length, 3)
            root_orientation = f['root_orientation'][:valid_length]        # (valid_length, 3, 3)
            cpf_translation = f['cpf_translation'][:valid_length]          # (valid_length, 3)
            cpf_orientation = f['cpf_orientation'][:valid_length]          # (valid_length, 3, 3)
            joint_translation = f['joint_translation'][:valid_length]      # (valid_length, 23, 3)
            joint_orientation = f['joint_orientation'][:valid_length]      # (valid_length, 23, 3, 3)
            contact_information = f['contact_information'][:valid_length]  # (valid_length, 4)

        def pad_to_length(arr: np.ndarray, target_len: int) -> np.ndarray:
            """Zero-pad array to target length along first dimension."""
            if len(arr) >= target_len:
                return arr[:target_len]
            pad_shape = (target_len - len(arr),) + arr.shape[1:]
            padding = np.zeros(pad_shape, dtype=arr.dtype)
            return np.concatenate([arr, padding], axis=0)

        # Create padding mask: 1.0 for valid frames, 0.0 for padded
        padding_mask = np.zeros(T, dtype=np.float32)
        padding_mask[:valid_length] = 1.0

        # Convert to torch tensors with zero-padding
        actions = {
            'root_translation': torch.from_numpy(pad_to_length(root_translation, T).astype(np.float32)),
            'root_orientation': torch.from_numpy(pad_to_length(root_orientation, T).astype(np.float32)),
            'cpf_translation': torch.from_numpy(pad_to_length(cpf_translation, T).astype(np.float32)),
            'cpf_orientation': torch.from_numpy(pad_to_length(cpf_orientation, T).astype(np.float32)),
            'joint_translation': torch.from_numpy(pad_to_length(joint_translation, T).astype(np.float32)),
            'joint_orientation': torch.from_numpy(pad_to_length(joint_orientation, T).astype(np.float32)),
            'contact_information': torch.from_numpy(pad_to_length(contact_information, T).astype(np.float32)),
            'padding_mask': torch.from_numpy(padding_mask),  # (n_frames,)
            'valid_length': torch.tensor(valid_length, dtype=torch.long),  # scalar
        }

        return actions

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


def nymeria_collate_fn(batch: list) -> Dict[str, Any]:
    """
    Custom collate function for NymeriaVideoDataset that properly handles
    the nested 'actions' dictionary with padding masks.

    Since all tensors are already padded to the same length (n_frames),
    this function stacks them into batched tensors.

    Args:
        batch: List of dictionaries from NymeriaVideoDataset.__getitem__

    Returns:
        Dictionary with batched tensors:
            - videos:               (B, n_frames, 3, H, W)
            - video_metadata:       list of dicts
            - video_path:           list of strings
            - actions (if present):
                - root_translation:     (B, n_frames, 3)
                - root_orientation:     (B, n_frames, 3, 3)
                - cpf_translation:      (B, n_frames, 3)
                - cpf_orientation:      (B, n_frames, 3, 3)
                - joint_translation:    (B, n_frames, 23, 3)
                - joint_orientation:    (B, n_frames, 23, 3, 3)
                - contact_information:  (B, n_frames, 4)
                - padding_mask:         (B, n_frames)
                - valid_length:         (B,)
    """
    # Separate keys that need special handling
    collated = {}

    # Stack tensor fields
    tensor_keys = ['videos', 'bbox_render', 'has_bbox']
    for key in tensor_keys:
        if key in batch[0]:
            collated[key] = torch.stack([item[key] for item in batch])

    # Keep list fields as lists
    list_keys = ['video_metadata', 'video_path']
    for key in list_keys:
        if key in batch[0]:
            collated[key] = [item[key] for item in batch]

    # Handle optional tensor fields (prompts, embeddings, latents)
    optional_tensor_keys = ['prompt_embeds', 'prompt_embed_len', 'image_latents', 'video_latents']
    for key in optional_tensor_keys:
        if key in batch[0] and batch[0][key] is not None:
            collated[key] = torch.stack([item[key] for item in batch])

    # Handle prompts (strings)
    if 'prompts' in batch[0]:
        collated['prompts'] = [item['prompts'] for item in batch]

    # Handle actions dictionary
    if 'actions' in batch[0]:
        actions_batch = {}
        action_keys = batch[0]['actions'].keys()
        for key in action_keys:
            values = [item['actions'][key] for item in batch]
            if isinstance(values[0], torch.Tensor):
                actions_batch[key] = torch.stack(values)
            else:
                # For non-tensor values like valid_length if stored as int
                actions_batch[key] = torch.tensor(values)
        collated['actions'] = actions_batch

    return collated
