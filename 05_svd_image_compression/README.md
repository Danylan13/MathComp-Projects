# SVD Image Compression

## Goal

Compress a synthetic grayscale image using low-rank matrix approximation and compare reconstruction quality at different ranks.

## Why It Matters

This is a clean example of linear algebra turning into a practical engineering technique: fewer stored parameters with controlled reconstruction error.

## Concepts

- singular value decomposition
- low-rank approximation
- Frobenius norm
- compression-quality trade-off

## Run

```bash
python 05_svd_image_compression/src/svd_compression.py
```

## Output

The script prints:

- reconstruction error for several ranks
- retained energy ratio
- an interpretation of the compression trade-off
