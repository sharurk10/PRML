from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

square = Image.open("images/Square_cat.png")

print("Original image size:", square.size)

square = square.resize((100, 100))

print("Resized image size:", square.size)

square_gray = square.convert("L")

print("Image mode:", square_gray.mode)

A = np.array(square_gray, dtype=float)

print("Matrix shape:", A.shape)
print("Matrix data type:", A.dtype)
print("Minimum pixel value:", A.min())
print("Maximum pixel value:", A.max())

print("\nFirst 5 rows and first 5 columns of A:")
print(A[:5, :5])

os.makedirs(
    "output",
    exist_ok=True
)

print("\nPerforming EVD...")

C = A @ A.T

print("Symmetric matrix shape:", C.shape)

eigenvalues, eigenvectors = np.linalg.eigh(C)

print("Number of eigenvalues:", len(eigenvalues))
print("Eigenvector matrix shape:", eigenvectors.shape)

idx = np.argsort(eigenvalues)[::-1]

eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

eigenvalues = np.maximum(
    eigenvalues,
    0
)

print("\nEigenvalues sorted from largest to smallest.")

print("\nTop 20 EVD eigenvalues:")

for rank in range(
    min(20, len(eigenvalues))
):
    print(
        rank + 1,
        "Eigenvalue =",
        eigenvalues[rank]
    )

print("\nCreating EVD eigenvalue spectrum...")

plt.figure(figsize=(10, 6))

plt.plot(
    range(1, len(eigenvalues) + 1),
    eigenvalues,
    marker="o",
    markersize=3
)

plt.xlabel(
    "Eigenvalue Rank"
)

plt.ylabel(
    "Eigenvalue"
)

plt.title(
    "EVD Eigenvalue Spectrum"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "output/evd_spectrum.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "\nEVD spectrum saved as output/evd_spectrum.png"
)

requested_k_values = [
    5,
    10,
    20,
    50,
    60,
    70,
    100
]

print("\nRequested K values:")
print(requested_k_values)

print("\nCalculating EVD reconstructions...")

saved_reconstructions = {}
saved_differences = {}
saved_errors = {}

all_k = []
all_errors = []

for k in requested_k_values:

    Qk = eigenvectors[:, :k]

    reconstructed = (
        Qk
        @ Qk.T
        @ A
    )

    reconstructed = np.clip(
        reconstructed,
        0,
        255
    )

    difference = np.abs(
        A - reconstructed
    )

    error = np.linalg.norm(
        A - reconstructed,
        ord="fro"
    )

    saved_reconstructions[k] = (
        reconstructed.copy()
    )

    saved_differences[k] = (
        difference.copy()
    )

    saved_errors[k] = error

    all_k.append(k)
    all_errors.append(error)

    reconstructed_image = Image.fromarray(
        reconstructed.astype(np.uint8)
    )

    reconstructed_image.save(
        f"output/evd_reconstructed_k{k}.png"
    )

    difference_image = Image.fromarray(
        np.clip(
            difference,
            0,
            255
        ).astype(np.uint8)
    )

    difference_image.save(
        f"output/evd_difference_k{k}.png"
    )

    plt.figure(figsize=(6, 6))

    plt.imshow(
        reconstructed,
        cmap="gray",
        vmin=0,
        vmax=255
    )

    plt.axis("off")

    plt.title(
        f"EVD Reconstruction\nK = {k}"
    )

    plt.tight_layout()

    plt.savefig(
        f"output/evd_reconstruction_k{k}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    plt.figure(figsize=(6, 6))

    plt.imshow(
        difference,
        cmap="gray",
        vmin=0,
        vmax=255
    )

    plt.axis("off")

    plt.title(
        f"EVD Difference Image\nK = {k}"
    )

    plt.tight_layout()

    plt.savefig(
        f"output/evd_difference_plot_k{k}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"K = {k}, "
        f"Frobenius error = {error:.4f}"
    )

print(
    "\nAll EVD reconstructions calculated."
)

print(
    "\nCreating Frobenius error vs K plot..."
)

plt.figure(figsize=(10, 6))

plt.plot(
    all_k,
    all_errors,
    marker="o",
    markersize=3
)

plt.xlabel(
    "Number of retained eigenvectors"
)

plt.ylabel(
    "Frobenius reconstruction error"
)

plt.title(
    "EVD Reconstruction Error vs K"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "output/evd_error_vs_k.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "EVD error plot saved as "
    "output/evd_error_vs_k.png"
)

print("\nFinal EVD Results")

for k in requested_k_values:

    print(
        f"K = {k}, "
        f"Frobenius error = "
        f"{saved_errors[k]:.4f}"
    )

print("\nEVD analysis completed.")

print("\nOutput files saved in the output folder:")

print("evd_spectrum.png")
print("evd_error_vs_k.png")

for k in requested_k_values:

    print(
        f"evd_reconstructed_k{k}.png"
    )

    print(
        f"evd_difference_k{k}.png"
    )

print(
    "\nReconstructed image size:",
    reconstructed_image.size
)