# Pattern Recognition and Machine Learning (PRML) Lab

This repository contains lab assignments and implementations for the **Pattern Recognition and Machine Learning (PRML)** course.

---

## 📁 Repository Structure

```text
PRML/
├── LAB1/
│   ├── dotproduct.cpp
│   ├── matrix_mul.cpp
│   ├── matrix_transpose.cpp
│   ├── semmetric_matrix.cpp
│   ├── Upper and Lower Triangular Matrix.cpp
│   ├── Uniform and Gaussian Random Variable Generation.cpp
│   ├── histogram.py
│   ├── uniform_data.txt
│   ├── gaussian_data.txt
│   ├── uniform_histogram.png
│   └── gaussian_histogram.png
│
├── LAB2/
│   ├── images/
│   │   ├── Square_cat.png
│   │   └── rectangle_cat.png
│   ├── output/                   # Reconstructed images and error/spectrum plots
│   ├── Matrix.py                 # Eigenvalue Decomposition (EVD) on image data
│   ├── SVD_Square.py             # SVD on square image matrices
│   └── SVD_Rectangular.py        # SVD on rectangular image matrices
│
└── README.md
```

---

## 🔬 Lab Modules

### 🧪 LAB 1: Linear Algebra & Probability Fundamentals

Focuses on foundational linear algebra operations and random variable generation using C++ and Python.

- **Matrix & Vector Operations (C++)**:
  - `dotproduct.cpp`: Vector dot product calculation.
  - `matrix_mul.cpp`: Matrix multiplication.
  - `matrix_transpose.cpp`: Computing transpose of a matrix.
  - `semmetric_matrix.cpp`: Symmetric matrix verification and handling.
  - `Upper and Lower Triangular Matrix.cpp`: Extracting upper and lower triangular matrices.
- **Random Variable Generation & Visualization**:
  - `Uniform and Gaussian Random Variable Generation.cpp`: Generates uniform and normal (Gaussian) distributed data points using PRNG & Box-Muller transformation.
  - `histogram.py`: Plots frequency distributions and histograms for `uniform_data.txt` and `gaussian_data.txt`.

#### How to Run (LAB 1):
```bash
# Compile and run C++ programs (using g++)
g++ "LAB1/dotproduct.cpp" -o LAB1/dotproduct
./LAB1/dotproduct

g++ "LAB1/Uniform and Gaussian Random Variable Generation.cpp" -o LAB1/rv_gen
./LAB1/rv_gen

# Plot histograms
python LAB1/histogram.py
```

---

### 🧪 LAB 2: Matrix Decompositions & Image Reconstruction

Explores low-rank matrix approximation and image compression techniques using **Eigenvalue Decomposition (EVD)** and **Singular Value Decomposition (SVD)** in Python.

- `Matrix.py`: Performs Eigenvalue Decomposition (EVD) on square image representations ($C = A A^T$) and computes low-rank approximations at varying ranks ($k$).
- `SVD_Square.py`: Applies Singular Value Decomposition ($A = U \Sigma V^T$) to square images to analyze energy compaction, reconstruct images at different ranks, and plot reconstruction error vs. $k$.
- `SVD_Rectangular.py`: Applies SVD to rectangular images with varying aspect ratios.
- `output/`: Contains spectrum curves, difference heatmaps, error vs. $k$ plots, and reconstructed output images across various rank thresholds ($k = 5, 10, 20, 50, 60, 70, 80, 100$).

#### How to Run (LAB 2):
```bash
# Execute SVD on square images
python LAB2/SVD_Square.py

# Execute SVD on rectangular images
python LAB2/SVD_Rectangular.py

# Execute EVD image analysis
python LAB2/Matrix.py
```

---

## ⚙️ Prerequisites & Dependencies

- **C++ Compiler**: `g++` (GCC) / Clang / MSVC (C++11 or higher)
- **Python 3.8+** with the following packages:
  ```bash
  pip install numpy matplotlib pillow
  ```
