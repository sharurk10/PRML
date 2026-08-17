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

print("\nPerforming SVD...")

U, S, VT = np.linalg.svd(A, full_matrices=False)

print("U matrix shape:", U.shape)
print("Singular values:", len(S))
print("VT matrix shape:", VT.shape)

print("\nFirst 20 singular values:")

for i, value in enumerate(S[:20], start=1):
    print(f"{i}: {value}")

os.makedirs("output", exist_ok=True)

print("\nCreating SVD singular value spectrum...")

plt.figure(figsize=(10, 6))

plt.plot(
    range(1, len(S) + 1),
    S,
    marker="o",
    markersize=3
)

plt.xlabel("Component number")
plt.ylabel("Singular value")
plt.title("SVD Singular Value Spectrum")
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "output/svd_spectrum.png",
    dpi=300
)

plt.close()

print("SVD spectrum saved as output/svd_spectrum.png")

print("\nPreparing SVD reconstruction...")

requested_k_values = [
    10,
    20,
    50,
    70,
    80,
    100
]

valid_k_values = [
    k for k in requested_k_values
    if k <= len(S)
]

print("\nRequested k values:")
print(requested_k_values)

print("\nAvailable k values:")
print(valid_k_values)

saved_reconstructions = {}
saved_differences = {}

all_k = []
all_errors = []

print("\nCalculating SVD reconstructions and errors...")

for k in range(1, len(S) + 1):

    U_k = U[:, :k]

    S_k = S[:k]

    VT_k = VT[:k, :]

    A_k = U_k @ np.diag(S_k) @ VT_k

    A_k = np.clip(
        A_k,
        0,
        255
    )

    difference = np.abs(
        A - A_k
    )

    error = np.linalg.norm(
        A - A_k,
        ord="fro"
    )

    all_k.append(k)
    all_errors.append(error)

    if k in valid_k_values:

        saved_reconstructions[k] = A_k.copy()

        saved_differences[k] = difference.copy()

print("All SVD errors calculated.")

print("\nSVD Reconstruction Results")

for k in valid_k_values:

    reconstructed = saved_reconstructions[k]

    difference = saved_differences[k]

    error = all_errors[k - 1]

    reconstructed_image = Image.fromarray(
        reconstructed.astype(np.uint8)
    )

    difference_image = Image.fromarray(
        np.clip(
            difference,
            0,
            255
        ).astype(np.uint8)
    )

    reconstructed_image.save(
        f"output/svd_reconstructed_k{k}.png"
    )

    difference_image.save(
        f"output/svd_difference_k{k}.png"
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
        f"SVD Reconstruction, k = {k}"
    )

    plt.tight_layout()

    plt.savefig(
        f"output/svd_reconstruction_k{k}.png",
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
        f"SVD Difference Image, k = {k}"
    )

    plt.tight_layout()

    plt.savefig(
        f"output/svd_difference_plot_k{k}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"k = {k}, "
        f"retained singular values = {k}, "
        f"Frobenius error = {error:.4f}"
    )

print("\nCreating Frobenius error vs k plot...")

plt.figure(figsize=(10, 6))

plt.plot(
    all_k,
    all_errors,
    marker="o",
    markersize=3
)

plt.xlabel(
    "Number of retained SVD components (k)"
)

plt.ylabel(
    "Frobenius reconstruction error"
)

plt.title(
    "SVD Reconstruction Error vs k"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "output/svd_error_vs_k.png",
    dpi=300
)

plt.close()

print(
    "SVD error plot saved as "
    "output/svd_error_vs_k.png"
)

print("\nFinal SVD Results")

for k in valid_k_values:

    print(
        f"k = {k}: "
        f"Frobenius error = "
        f"{all_errors[k - 1]:.4f}"
    )

print("\nSVD analysis completed.")

print("\nOutput files saved in the output folder:")

print("svd_spectrum.png")
print("svd_error_vs_k.png")

for k in valid_k_values:

    print(
        f"svd_reconstructed_k{k}.png"
    )

    print(
        f"svd_difference_k{k}.png"
    )

print("\nSVD output generation completed.")