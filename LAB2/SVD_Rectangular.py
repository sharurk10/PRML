from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

rectangle = Image.open("images/rectangle_cat.png")

print("Original image size:", rectangle.size)

max_size = 100

width, height = rectangle.size

scale = min(
    max_size / width,
    max_size / height,
    1
)

new_width = int(width * scale)
new_height = int(height * scale)

rectangle = rectangle.resize(
    (new_width, new_height)
)

print("Resized image size:", rectangle.size)

rectangle_gray = rectangle.convert("L")

print("Image mode:", rectangle_gray.mode)

A = np.array(
    rectangle_gray,
    dtype=float
)

print("Matrix shape:", A.shape)
print("Matrix data type:", A.dtype)
print("Minimum pixel value:", A.min())
print("Maximum pixel value:", A.max())

print("\nFirst 5 rows and first 5 columns of A:")
print(A[:5, :5])

print("\nPerforming SVD...")

U, singular_values, Vt = np.linalg.svd(
    A,
    full_matrices=False
)

print("\nSVD Results")
print("U shape:", U.shape)
print("Singular values shape:", singular_values.shape)
print("Vt shape:", Vt.shape)

print("\nNumber of singular values:", len(singular_values))

print("\nFirst 20 singular values:")

for i, value in enumerate(
    singular_values[:20],
    start=1
):
    print(f"{i}: {value}")

os.makedirs(
    "output",
    exist_ok=True
)

print("\nCreating SVD singular value spectrum...")

plt.figure(figsize=(8, 5))

plt.plot(
    range(
        1,
        len(singular_values) + 1
    ),
    singular_values,
    marker="o",
    markersize=3
)

plt.xlabel("Component number")
plt.ylabel("Singular value")
plt.title(
    "Rectangular Image SVD Singular Value Spectrum"
)
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "output/rectangle_svd_spectrum.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

k_values = [
    k for k in [10, 20, 50, 70, 80, 100]
    if k <= len(singular_values)
]

print("\nSelected k values:")
print(k_values)

svd_results = {}

print("\nRectangular SVD Reconstruction Results")

for k in k_values:

    U_k = U[:, :k]

    S_k = singular_values[:k]

    Vt_k = Vt[:k, :]

    A_k = (
        U_k * S_k
    ) @ Vt_k

    A_k = np.clip(
        A_k,
        0,
        255
    )

    difference = np.abs(
        A - A_k
    )

    frobenius_error = np.linalg.norm(
        A - A_k,
        ord="fro"
    )

    svd_results[k] = {
        "reconstructed": A_k,
        "difference": difference,
        "error": frobenius_error
    }

    reconstructed_image = Image.fromarray(
        A_k.astype(np.uint8)
    )

    difference_image = Image.fromarray(
        np.clip(
            difference,
            0,
            255
        ).astype(np.uint8)
    )

    reconstructed_image.save(
        f"output/rectangle_svd_reconstructed_k{k}.png"
    )

    difference_image.save(
        f"output/rectangle_svd_difference_k{k}.png"
    )

    print(
        f"k = {k}, "
        f"retained singular values = {k}, "
        f"Frobenius error = "
        f"{frobenius_error:.4f}"
    )

    plt.figure(figsize=(7, 5))

    plt.imshow(
        A_k,
        cmap="gray"
    )

    plt.axis("off")

    plt.title(
        f"Rectangular SVD Reconstruction, k = {k}"
    )

    plt.tight_layout()

    plt.savefig(
        f"output/rectangle_svd_reconstruction_k{k}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    plt.figure(figsize=(7, 5))

    plt.imshow(
        difference,
        cmap="gray"
    )

    plt.axis("off")

    plt.title(
        f"Rectangular SVD Difference Image, k = {k}"
    )

    plt.tight_layout()

    plt.savefig(
        f"output/rectangle_svd_difference_plot_k{k}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

print("\nCalculating Frobenius error for all possible k values...")

all_k = np.arange(
    1,
    len(singular_values) + 1
)

squared_values = (
    singular_values ** 2
)

remaining_squared_sum = np.cumsum(
    squared_values[::-1]
)[::-1]

all_errors = np.sqrt(
    np.concatenate(
        [
            remaining_squared_sum[1:],
            [0]
        ]
    )
)

print("\nCreating SVD error vs k plot...")

plt.figure(figsize=(10, 6))

plt.plot(
    all_k,
    all_errors,
    marker="o",
    markersize=2
)

plt.xlabel(
    "Number of retained SVD components (k)"
)

plt.ylabel(
    "Frobenius reconstruction error"
)

plt.title(
    "Rectangular Image SVD Reconstruction Error vs k"
)

plt.grid(True)
plt.tight_layout()

plt.savefig(
    "output/rectangle_svd_error_vs_k.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nFinal Rectangular SVD Results")

for k in k_values:

    print(
        f"k = {k}: "
        f"Frobenius error = "
        f"{svd_results[k]['error']:.4f}"
    )

print("\nRectangular SVD analysis completed.")

print("\nOutput files saved in the output folder:")

print("rectangle_svd_spectrum.png")
print("rectangle_svd_error_vs_k.png")

for k in k_values:

    print(
        f"rectangle_svd_reconstructed_k{k}.png"
    )

    print(
        f"rectangle_svd_difference_k{k}.png"
    )

print("\nAll selected rectangular SVD reconstructions:")
print(k_values)