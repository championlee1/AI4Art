#!/usr/bin/env python3
"""
FASTA -> abstract image pipeline (first runnable MVP for AI4Art).

Features:
1) Optional FASTA download from NCBI E-utilities (by accession IDs)
2) FASTA parsing
3) Sequence feature extraction (entropy profile + k-mer histogram)
4) Artistic rendering into high-res PNG
5) Metadata JSON export for reproducibility
"""

from __future__ import annotations

import argparse
import json
import math
import random
import textwrap
from collections import Counter
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode
from urllib.request import urlopen

import matplotlib.pyplot as plt
import numpy as np

try:
    import torch
    from esm import pretrained as esm_pretrained
except ImportError:  # optional runtime dependency
    torch = None
    esm_pretrained = None


ALPHABET_DNA = set("ACGTN")
ALPHABET_PROTEIN = set("ACDEFGHIKLMNPQRSTVWYBXZJUO")


@dataclass
class ArtMetadata:
    run_utc: str
    input_fasta: str
    sequence_id: str
    accession: str
    sequence_length: int
    sequence_type: str
    ncbi_db: str
    k: int
    patch: int
    temperature: float
    noise_scale: float
    top_n_kmers: int
    seed: int
    esm2_model_path: str | None
    top_kmers: list[tuple[str, int]]
    output_image: str


def download_ncbi_fasta(accessions: list[str], out_fasta: Path, db: str = "nuccore") -> None:
    ids = ",".join(accessions)
    params = urlencode({"db": db, "id": ids, "rettype": "fasta", "retmode": "text"})
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?{params}"
    with urlopen(url, timeout=60) as resp:  # noqa: S310
        text = resp.read().decode("utf-8", errors="replace")
    if ">" not in text:
        raise RuntimeError("NCBI efetch returned no FASTA records. Check accession IDs/db.")
    out_fasta.parent.mkdir(parents=True, exist_ok=True)
    out_fasta.write_text(text, encoding="utf-8")


def parse_fasta(path: Path) -> Iterable[tuple[str, str]]:
    header = None
    seq_parts: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                yield header, "".join(seq_parts).upper()
            header = line[1:].strip()
            seq_parts = []
        else:
            seq_parts.append(line)
    if header is not None:
        yield header, "".join(seq_parts).upper()




def parse_accession(header: str) -> str:
    first = header.split()[0]
    if "|" in first:
        parts = [x for x in first.split("|") if x]
        if len(parts) >= 2:
            return parts[1]
    return first

def infer_seq_type(seq: str) -> str:
    s = set(seq)
    if s <= ALPHABET_DNA:
        return "dna"
    if s <= ALPHABET_PROTEIN:
        return "protein"
    return "mixed"


def shannon_entropy(window: str) -> float:
    c = Counter(window)
    n = len(window)
    if n == 0:
        return 0.0
    return -sum((v / n) * math.log2(v / n) for v in c.values())


def entropy_profile(seq: str, patch: int) -> np.ndarray:
    chunks = [seq[i : i + patch] for i in range(0, len(seq), patch)]
    ent = np.array([shannon_entropy(ch) for ch in chunks], dtype=float)
    if ent.max(initial=0.0) > 0:
        ent = ent / ent.max()
    return ent


def kmer_hist(seq: str, k: int, top_n: int) -> list[tuple[str, int]]:
    if len(seq) < k:
        return []
    kmers = Counter(seq[i : i + k] for i in range(len(seq) - k + 1))
    return kmers.most_common(top_n)


def resolve_esm2_model_path(model_dir: Path) -> Path:
    if model_dir.is_file():
        return model_dir
    candidates = sorted(model_dir.glob("*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No .pt checkpoint found under: {model_dir}")
    return candidates[0]


def esm2_embedding(seq: str, model_path: Path, device: str = "cpu") -> np.ndarray:
    if torch is None or esm_pretrained is None:
        raise RuntimeError("ESM2 dependencies missing. Install `torch` and `fair-esm` first.")
    model, alphabet = esm_pretrained.load_model_and_alphabet_local(str(model_path))
    model.eval()
    model = model.to(device)
    batch_converter = alphabet.get_batch_converter()
    _, _, batch_tokens = batch_converter([("query", seq)])
    batch_tokens = batch_tokens.to(device)
    with torch.no_grad():
        result = model(batch_tokens, repr_layers=[33], return_contacts=False)
    token_repr = result["representations"][33][0, 1 : len(seq) + 1]
    emb = token_repr.mean(0).detach().cpu().numpy().astype(np.float32)
    return emb


def render_art(
    seq: str,
    ent: np.ndarray,
    kmers: list[tuple[str, int]],
    temperature: float,
    noise_scale: float,
    seed: int,
    max_grid_size: int,
) -> np.ndarray:
    random.seed(seed)
    np.random.seed(seed)

    n = max(64, int(np.sqrt(len(seq))) * 4)
    n = min(n, max_grid_size)
    grid = np.zeros((n, n, 3), dtype=float)

    # Base color field from entropy stripes
    for i in range(n):
        val = ent[i % max(1, len(ent))] if len(ent) else 0.0
        grid[i, :, 0] = 0.25 + 0.75 * val
        grid[i, :, 1] = 0.15 + 0.85 * (1 - val)

    # K-mer pulses as vertical bars
    if kmers:
        counts = np.array([c for _, c in kmers], dtype=float)
        counts = counts / counts.max()
        for idx, amp in enumerate(counts):
            x = int((idx + 1) * n / (len(kmers) + 1))
            width = max(1, int((0.01 + 0.03 * amp) * n))
            color = np.array([amp, 0.2 + 0.6 * (1 - amp), 0.5 + 0.4 * amp])
            grid[:, max(0, x - width) : min(n, x + width), :] += color * (0.4 + 0.6 * temperature)

    # Noise hallucination
    noise = np.random.normal(0, noise_scale * temperature, size=grid.shape)
    grid = np.clip(grid + noise, 0, 1)

    # Soft channel mixing by temperature
    mix = min(0.95, max(0.0, (temperature - 0.7) / 2.0))
    grid = (1 - mix) * grid + mix * grid[..., [2, 0, 1]]
    return np.clip(grid, 0, 1)


def render_embedding_art(embedding: np.ndarray, n: int, temperature: float, noise_scale: float, seed: int) -> np.ndarray:
    random.seed(seed)
    np.random.seed(seed)
    y, x = np.mgrid[-1:1:complex(0, n), -1:1:complex(0, n)]
    r = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)

    harmonics = np.abs(embedding[:12]) if embedding.size >= 12 else np.pad(np.abs(embedding), (0, max(0, 12 - embedding.size)))
    harmonics = harmonics / max(1e-6, harmonics.max(initial=1.0))
    field = np.zeros_like(r)
    for i, amp in enumerate(harmonics, start=1):
        freq = i * (1.0 + 0.7 * temperature)
        phase = (embedding[i % embedding.size] if embedding.size else 0.0) * math.pi
        field += amp * np.sin(freq * theta + (freq * 2.5) * r + phase)
    field = (field - field.min()) / max(1e-6, field.max() - field.min())

    palette = np.array(
        [
            0.55 + 0.45 * np.sin(2 * math.pi * field + temperature),
            0.50 + 0.50 * np.sin(2 * math.pi * field + 2.1),
            0.45 + 0.55 * np.cos(2 * math.pi * field + 4.3),
        ]
    ).transpose(1, 2, 0)

    noise = np.random.normal(0, noise_scale * max(0.3, temperature), size=palette.shape)
    vignette = np.clip(1.2 - r**1.7, 0, 1)[..., None]
    art = np.clip((palette + noise) * vignette + 0.08, 0, 1)
    return art


def save_figure(arr: np.ndarray, output: Path, title: str, annotation: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 10), dpi=300)
    plt.imshow(arr)
    plt.axis("off")
    plt.title("\n".join(textwrap.wrap(title, width=60)), fontsize=9)
    plt.figtext(0.01, 0.01, annotation, ha="left", va="bottom", fontsize=7, color="white", bbox={"facecolor": "black", "alpha": 0.45, "pad": 3})
    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight", pad_inches=0)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate abstract art from FASTA sequences.")
    parser.add_argument("--fasta", type=Path, help="Input FASTA file path.")
    parser.add_argument("--accessions", nargs="*", default=[], help="NCBI accession list for auto-download.")
    parser.add_argument("--db", default="nuccore", help="NCBI database: nuccore/protein.")
    parser.add_argument("--download-to", type=Path, default=Path("artifacts/input/ncbi_download.fasta"))
    parser.add_argument("--outdir", type=Path, default=Path("artifacts/images/fasta_art"))
    parser.add_argument("--meta-dir", type=Path, default=Path("artifacts/metadata"))
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--patch", type=int, default=64)
    parser.add_argument("--top-n-kmers", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.4)
    parser.add_argument("--noise-scale", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/metadata/art_manifest.tsv"))
    parser.add_argument("--max-grid-size", type=int, default=2048, help="Maximum render grid width/height to control memory use.")
    parser.add_argument("--use-esm2", action="store_true", help="Use ESM2 embedding-driven renderer.")
    parser.add_argument("--esm2-model-dir", type=Path, default=None, help="Path to ESM2 checkpoint dir or .pt file.")
    parser.add_argument("--embedding-device", default="cpu", help="Torch device for ESM2 embedding, e.g. cpu/cuda.")
    args = parser.parse_args()

    if args.max_grid_size < 64:
        raise SystemExit("--max-grid-size must be >= 64.")
    if args.use_esm2 and args.esm2_model_dir is None:
        raise SystemExit("When --use-esm2 is set, provide --esm2-model-dir.")

    fasta_path = args.fasta
    if args.accessions:
        download_ncbi_fasta(args.accessions, args.download_to, db=args.db)
        fasta_path = args.download_to
    if fasta_path is None or not fasta_path.exists():
        raise SystemExit("Provide --fasta or --accessions to fetch from NCBI.")

    args.meta_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    manifest_lines = ["image_path\taccession\tsequence_id\tseq_type\tlength\tncbi_db\ttemperature\tnoise_scale\tseed"]
    esm2_model_path = resolve_esm2_model_path(args.esm2_model_dir) if args.use_esm2 else None

    for idx, (header, seq) in enumerate(parse_fasta(fasta_path), start=1):
        seq_type = infer_seq_type(seq)
        accession = parse_accession(header)
        ent = entropy_profile(seq, args.patch)
        kmers = kmer_hist(seq, args.k, args.top_n_kmers)

        n = min(max(64, int(np.sqrt(len(seq))) * 4), args.max_grid_size)
        if args.use_esm2:
            embedding = esm2_embedding(seq, esm2_model_path, device=args.embedding_device)
            art = render_embedding_art(embedding, n, args.temperature, args.noise_scale, args.seed + idx)
        else:
            art = render_art(
                seq,
                ent,
                kmers,
                args.temperature,
                args.noise_scale,
                args.seed + idx,
                args.max_grid_size,
            )

        safe_id = "".join(ch if ch.isalnum() else "_" for ch in header.split()[0])[:80] or f"seq_{idx}"
        img_path = args.outdir / f"{idx:03d}_{safe_id}.png"
        annotation = f"accession={accession} | db={args.db} | type={seq_type} | len={len(seq)} | temp={args.temperature} | noise={args.noise_scale}"
        save_figure(art, img_path, f"{safe_id} | len={len(seq)} | temp={args.temperature}", annotation)

        meta = ArtMetadata(
            run_utc=datetime.now(timezone.utc).isoformat(),
            input_fasta=str(fasta_path),
            sequence_id=header,
            accession=accession,
            sequence_length=len(seq),
            sequence_type=seq_type,
            ncbi_db=args.db,
            k=args.k,
            patch=args.patch,
            temperature=args.temperature,
            noise_scale=args.noise_scale,
            top_n_kmers=args.top_n_kmers,
            seed=args.seed + idx,
            esm2_model_path=str(esm2_model_path) if esm2_model_path else None,
            top_kmers=kmers[:10],
            output_image=str(img_path),
        )
        meta_path = args.meta_dir / f"{idx:03d}_{safe_id}.json"
        meta_path.write_text(json.dumps(asdict(meta), ensure_ascii=False, indent=2), encoding="utf-8")
        manifest_lines.append(f"{img_path}\t{accession}\t{header}\t{seq_type}\t{len(seq)}\t{args.db}\t{args.temperature}\t{args.noise_scale}\t{args.seed + idx}")
        print(f"[OK] {img_path} | accession={accession}")

    args.manifest.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    print(f"[OK] manifest: {args.manifest}")


if __name__ == "__main__":
    main()
