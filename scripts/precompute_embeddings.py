"""
Precompute T5 text embeddings for Nymeria dataset.

This script loads the UMT5-XXL text encoder and precomputes embeddings for all
captions in the metadata CSV, saving them as .pt files.

Usage:
    conda activate ei_world_model
    python scripts/precompute_embeddings.py \
        --csv-path /data/nymeria/metadata.csv \
        --output-dir /data/nymeria/embeddings \
        --t5-ckpt data/ckpts/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth

This will:
    1. Load captions from the CSV
    2. Encode them with UMT5-XXL
    3. Save each embedding as a .pt file
    4. Update the CSV with prompt_embed_path column
"""

import sys
from dataclasses import dataclass
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import torch
import tyro
from tqdm import tqdm

from algorithms.wan.modules.t5 import umt5_xxl
from algorithms.wan.modules.tokenizers import HuggingfaceTokenizer


def load_text_encoder(ckpt_path: str, device: str = "cuda"):
    """Load the UMT5-XXL text encoder."""
    print(f"Loading text encoder from {ckpt_path}...")

    text_encoder = umt5_xxl(
        encoder_only=True,
        return_tokenizer=False,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    ).eval().requires_grad_(False)

    if ckpt_path and Path(ckpt_path).exists():
        text_encoder.load_state_dict(
            torch.load(ckpt_path, map_location="cpu", weights_only=True)
        )

    text_encoder = text_encoder.to(device)
    print("Text encoder loaded successfully!")
    return text_encoder


def load_tokenizer(name: str = "google/umt5-xxl", seq_len: int = 512):
    """Load the tokenizer."""
    return HuggingfaceTokenizer(
        name=name,
        seq_len=seq_len,
        clean="whitespace",
    )


@torch.no_grad()
def encode_texts(texts: list, text_encoder, tokenizer, device: str = "cuda"):
    """Encode a batch of texts and return embeddings."""
    ids, mask = tokenizer(texts, return_mask=True, add_special_tokens=True)
    ids = ids.to(device)
    mask = mask.to(device)

    # Get sequence lengths (before padding)
    seq_lens = mask.gt(0).sum(dim=1).long()

    # Encode
    context = text_encoder(ids, mask)

    # Return list of embeddings, each trimmed to actual sequence length
    embeddings = [u[:v].cpu() for u, v in zip(context, seq_lens)]
    return embeddings


def precompute_embeddings(
    csv_path: Path,
    output_dir: Path,
    t5_ckpt: str,
    batch_size: int = 8,
    device: str = "cuda",
    tokenizer_name: str = "google/umt5-xxl",
    seq_len: int = 512,
):
    """
    Precompute embeddings for all captions in the CSV.

    Args:
        csv_path: Path to the input CSV with captions
        output_dir: Directory to save embedding .pt files
        t5_ckpt: Path to T5 checkpoint
        batch_size: Batch size for encoding
        device: Device to use for encoding
        tokenizer_name: HuggingFace tokenizer name
        seq_len: Maximum sequence length
    """
    # Load CSV
    print(f"Loading CSV from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Found {len(df)} records")

    # Check for caption column
    if "caption" not in df.columns:
        raise ValueError("CSV must have a 'caption' column")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving embeddings to {output_dir}")

    # Load model and tokenizer
    text_encoder = load_text_encoder(t5_ckpt, device)
    tokenizer = load_tokenizer(tokenizer_name, seq_len)

    # Process in batches
    embed_paths = []
    captions = df["caption"].tolist()

    # Use video_path to create unique embedding filenames
    video_paths = df["video_path"].tolist()

    for i in tqdm(range(0, len(captions), batch_size), desc="Encoding"):
        batch_captions = captions[i:i + batch_size]
        batch_video_paths = video_paths[i:i + batch_size]

        # Encode batch
        embeddings = encode_texts(batch_captions, text_encoder, tokenizer, device)

        # Save each embedding
        for j, (emb, video_path) in enumerate(zip(embeddings, batch_video_paths)):
            # Create filename based on video path (replace .mp4 with .pt)
            video_name = Path(video_path).stem
            embed_filename = f"{video_name}.pt"
            embed_path = output_dir / embed_filename

            # Save embedding
            torch.save(emb, embed_path)

            # Store relative path (relative to data_root)
            embed_paths.append(f"embeddings/{embed_filename}")

    # Update CSV with embedding paths
    df["prompt_embed_path"] = embed_paths

    # Save updated CSV
    output_csv = csv_path.parent / f"{csv_path.stem}_with_embeddings.csv"
    df.to_csv(output_csv, index=False)
    print(f"Saved updated CSV to {output_csv}")

    # Also update the original CSV
    df.to_csv(csv_path, index=False)
    print(f"Updated original CSV at {csv_path}")

    print(f"\nDone! Precomputed {len(embed_paths)} embeddings")
    print(f"Embedding shape example: {embeddings[0].shape}")

    return df


@dataclass
class Config:
    """Precompute T5 text embeddings for video dataset."""

    csv_path: Path = Path("/data/nymeria/metadata.csv")
    """Path to metadata CSV with captions."""

    output_dir: Path = Path("/data/nymeria/embeddings")
    """Directory to save embedding .pt files."""

    t5_ckpt: Path = Path("data/ckpts/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth")
    """Path to T5 checkpoint."""

    batch_size: int = 8
    """Batch size for encoding."""

    device: str = "cuda"
    """Device to use (cuda or cpu)."""

    tokenizer: str = "google/umt5-xxl"
    """HuggingFace tokenizer name."""

    seq_len: int = 512
    """Maximum sequence length."""


def main(cfg: Config):
    precompute_embeddings(
        csv_path=cfg.csv_path,
        output_dir=cfg.output_dir,
        t5_ckpt=str(cfg.t5_ckpt),
        batch_size=cfg.batch_size,
        device=cfg.device,
        tokenizer_name=cfg.tokenizer,
        seq_len=cfg.seq_len,
    )


if __name__ == "__main__":
    main(tyro.cli(Config))
