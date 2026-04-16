from __future__ import annotations

import numpy as np


def synthetic_image(size: int = 64) -> np.ndarray:
    rng = np.random.default_rng(5)
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(x, y)

    radial = np.exp(-4 * (xx**2 + yy**2))
    waves = 0.35 * np.sin(8 * xx) * np.cos(6 * yy)
    edge = 0.25 * (xx + 1) / 2
    blob_1 = 0.35 * np.exp(-18 * ((xx + 0.4) ** 2 + (yy - 0.1) ** 2))
    blob_2 = 0.25 * np.exp(-24 * ((xx - 0.35) ** 2 + (yy + 0.45) ** 2))
    texture = 0.05 * rng.normal(size=(size, size))
    image = radial + waves + edge + blob_1 + blob_2 + texture
    image = (image - image.min()) / (image.max() - image.min())
    return image


def compress(image: np.ndarray, rank: int) -> tuple[np.ndarray, float]:
    u, singular_values, vt = np.linalg.svd(image, full_matrices=False)
    approximation = u[:, :rank] @ np.diag(singular_values[:rank]) @ vt[:rank, :]
    retained_energy = float(np.sum(singular_values[:rank] ** 2) / np.sum(singular_values**2))
    return approximation, retained_energy


def frobenius_error(original: np.ndarray, approximation: np.ndarray) -> float:
    return float(np.linalg.norm(original - approximation, ord="fro"))


def main() -> None:
    image = synthetic_image()
    ranks = [2, 5, 10, 20]

    print("SVD Image Compression")
    print("-" * 60)
    for rank in ranks:
        approximation, retained_energy = compress(image, rank)
        error = frobenius_error(image, approximation)
        print(
            f"rank={rank:>2} | reconstruction_error={error:.4f} "
            f"| retained_energy={retained_energy:.4%}"
        )


if __name__ == "__main__":
    main()
