"""
prep_gene_map_from_gtf.py

Parses a GENCODE GTF annotation file (gzip-compressed) and extracts a
three-column gene map containing Ensembl gene IDs (version suffix stripped),
HGNC gene symbols, and gene body length in base-pairs.

The resulting tab-separated table is written to a gzip-compressed TSV file
at ``data/reference/genes/gene_map.tsv.gz`` and is used downstream as a
lookup table for RNA-seq TPM normalisation (TPM requires gene length) and
for mapping between Ensembl identifiers and human-readable gene symbols.

Usage
-----
    python prep_gene_map_from_gtf.py

Input
-----
    data/reference/genes/gencode.v49.annotation.gtf.gz

Output
------
    data/reference/genes/gene_map.tsv.gz
    Columns: ensembl_id, gene_symbol, gene_length_bp
"""

import gzip
import re
from pathlib import Path
import pandas as pd

# ---------- CONFIG ----------
ROOT = Path(".")   # or absolute path if you want
GTF_PATH = ROOT / "data" / "reference" / "genes" / "gencode.v49.annotation.gtf.gz"
OUT_PATH = ROOT / "data" / "reference" / "genes" / "gene_map.tsv.gz"


def parse_gtf(gtf_path):
    """Parse a GENCODE GTF file and return a gene-level summary DataFrame.

    Only rows whose feature type (GTF column 3) is ``"gene"`` are retained;
    transcript, exon, and CDS records are skipped.  For each gene the
    function extracts:

    * ``ensembl_id``      – Ensembl gene ID with the version suffix removed
                           (e.g. ``ENSG00000123456.4`` → ``ENSG00000123456``).
    * ``gene_symbol``     – HGNC gene name from the ``gene_name`` attribute.
    * ``gene_length_bp``  – Genomic span of the gene body in base-pairs,
                           computed as ``end - start + 1`` (1-based, inclusive
                           GTF coordinates).

    If the same Ensembl ID appears more than once (e.g. genes on PAR regions),
    only the first occurrence is kept after deduplication.

    Parameters
    ----------
    gtf_path : str or Path
        Path to a gzip-compressed GENCODE GTF annotation file.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``["ensembl_id", "gene_symbol",
        "gene_length_bp"]``, one row per unique Ensembl gene ID.
    """
    rows = []
    with gzip.open(gtf_path, "rt") as f:
        for line in f:
            # Skip GTF comment / metadata header lines (begin with '#').
            if line.startswith("#"):
                continue

            fields = line.strip().split("\t")
            feature_type = fields[2]
            # Only process gene-level records; skip transcripts, exons, etc.
            if feature_type != "gene":
                continue

            # GTF coordinates are 1-based and inclusive on both ends.
            start = int(fields[3])
            end = int(fields[4])
            length = end - start + 1

            # The 9th field (index 8) holds semi-colon-delimited attributes.
            attrs = fields[8]
            gid = re.search('gene_id "([^"]+)"', attrs).group(1)
            gname = re.search('gene_name "([^"]+)"', attrs).group(1)

            # ENSG00000123456.4 → ENSG00000123456
            # Strip the version number so the ID can be matched against
            # expression matrices that may omit the version suffix.
            gid_nover = gid.split(".")[0]

            rows.append([gid_nover, gname, length])

    df = pd.DataFrame(rows, columns=["ensembl_id", "gene_symbol", "gene_length_bp"])
    # Keep first occurrence when an Ensembl ID appears on multiple contigs
    # (e.g. pseudoautosomal regions duplicated on chrX and chrY).
    df = df.drop_duplicates("ensembl_id")
    return df


def main():
    """Entry point: parse the GTF, print progress, and write the gene map TSV."""
    print(f"Reading GTF: {GTF_PATH}")
    df = parse_gtf(GTF_PATH)
    print(f"Parsed {df.shape[0]} genes")

    # Create output directory if it does not yet exist.
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, sep="\t", index=False, compression="gzip")

    print(f"Saved gene map to {OUT_PATH}")


if __name__ == "__main__":
    main()
