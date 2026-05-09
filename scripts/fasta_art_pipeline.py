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


def render_art(seq: str, ent: np.ndarray, kmers: list[tuple[str, int]], temperature: float, noise_scale: float, seed: int) -> np.ndarray:
    random.seed(seed)
    np.random.seed(seed)

    n = max(64, int(np.sqrt(len(seq))) * 4)
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
    args = parser.parse_args()

    fasta_path = args.fasta
    if args.accessions:
        download_ncbi_fasta(args.accessions, args.download_to, db=args.db)
        fasta_path = args.download_to
    if fasta_path is None or not fasta_path.exists():
        raise SystemExit("Provide --fasta or --accessions to fetch from NCBI.")

    args.meta_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    manifest_lines = ["image_path\taccession\tsequence_id\tseq_type\tlength\tncbi_db\ttemperature\tnoise_scale\tseed"]

    for idx, (header, seq) in enumerate(parse_fasta(fasta_path), start=1):
        seq_type = infer_seq_type(seq)
        accession = parse_accession(header)
        ent = entropy_profile(seq, args.patch)
        kmers = kmer_hist(seq, args.k, args.top_n_kmers)

        art = render_art(seq, ent, kmers, args.temperature, args.noise_scale, args.seed + idx)

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
