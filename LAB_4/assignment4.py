import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
N = 10000
mu = np.array([5., 5.])
levels = [1, 2, 3]

D1 = np.random.normal(mu[0], 1, N)
D2 = np.random.normal(mu[1], 1, N)
X = np.column_stack((D1, D2))

def cov_sqrt(S):
    l, V = np.linalg.eigh(S)
    return V @ np.diag(np.sqrt(l)) @ V.T

def generate(S):
    return mu + (X - mu) @ cov_sqrt(S).T

def analyze(Y, S, title):
    m = Y.mean(0)
    C = np.cov(Y, rowvar=False)
    l, V = np.linalg.eigh(C)

    print(f"\n{title}")
    print("Estimated Mean =", np.round(m, 4))
    print("Required Covariance =\n", S)
    print("Estimated Covariance =\n", np.round(C, 4))
    print("Theoretical Eigenvalues =", np.round(np.linalg.eigvalsh(S), 4))
    print("Estimated Eigenvalues =", np.round(l, 4))
    print("Eigenvectors =\n", np.round(V, 4))

    major = np.argmax(l)
    minor = np.argmin(l)
    print("Major Eigenvalue =", round(l[major], 4))
    print("Minor Eigenvalue =", round(l[minor], 4))
    print("Major Semi-axis =", round(np.sqrt(3*l[major]), 4))
    print("Minor Semi-axis =", round(np.sqrt(3*l[minor]), 4))
    print("Larger eigenvalue -> Major axis")
    print("Smaller eigenvalue -> Minor axis")

    x = np.linspace(m[0]-5, m[0]+5, 400)
    y = np.linspace(m[1]-5, m[1]+5, 400)
    xx, yy = np.meshgrid(x, y)
    P = np.dstack((xx, yy))
    D = P - m
    Q = np.einsum("...i,ij,...j->...", D, np.linalg.inv(C), D)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(Y[::20, 0], Y[::20, 1], s=5, alpha=.2)
    ax.contour(xx, yy, Q, levels=levels)

    for i in range(2):
        d = np.sqrt(3*l[i]) * V[:, i]
        ax.plot([m[0]-d[0], m[0]+d[0]],
                [m[1]-d[1], m[1]+d[1]], "--")

    ax.scatter(m[0], m[1], color="black", marker="x", s=70)
    ax.set_title(title)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.axis("equal")
    ax.grid(alpha=.25)
    plt.tight_layout()
    plt.show()

    return C, l, V

S1 = np.eye(2)
Y1 = generate(S1)
C1, L1, V1 = analyze(Y1, S1, "1. Isotropic Covariance")

S2 = np.diag([4., 1.])
Y2 = generate(S2)
C2, L2, V2 = analyze(Y2, S2, "2. Diagonal Covariance")

S3 = np.array([[4., 1.5], [1.5, 2.]])
Y3 = generate(S3)
C3, L3, V3 = analyze(Y3, S3, "3. Full Covariance")

def gaussian(P, m, S):
    D = P - m
    q = np.einsum("...i,ij,...j->...", D, np.linalg.inv(S), D)
    return np.exp(-q/2)/(2*np.pi*np.sqrt(np.linalg.det(S)))

def decision_plot(mu1, S1, mu2, S2, title):
    Y1 = np.random.multivariate_normal(mu1, S1, N)
    Y2 = np.random.multivariate_normal(mu2, S2, N)

    x = np.linspace(0, 13, 500)
    y = np.linspace(0, 13, 500)
    xx, yy = np.meshgrid(x, y)
    P = np.dstack((xx, yy))

    G1 = gaussian(P, mu1, S1)
    G2 = gaussian(P, mu2, S2)

    print(f"\n{title}")
    print("Class 1 Mean =", mu1)
    print("Class 1 Covariance =\n", S1)
    print("Class 2 Mean =", mu2)
    print("Class 2 Covariance =\n", S2)

    plt.figure(figsize=(7, 7))
    plt.scatter(Y1[::20, 0], Y1[::20, 1], s=5, alpha=.2)
    plt.scatter(Y2[::20, 0], Y2[::20, 1], s=5, alpha=.2)
    plt.contour(xx, yy, G1, levels=8)
    plt.contour(xx, yy, G2, levels=8)
    plt.contour(xx, yy, G1-G2, levels=[0], linewidths=2)
    plt.scatter(*mu1, marker="x", s=80)
    plt.scatter(*mu2, marker="x", s=80)
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title(title)
    plt.axis("equal")
    plt.grid(alpha=.25)
    plt.tight_layout()
    plt.show()

mu1 = np.array([5., 5.])
mu2 = np.array([8., 8.])
S4_1 = np.array([[4., 1.], [1., 2.]])
S4_2 = np.array([[2., -.8], [-.8, 3.]])
decision_plot(mu1, S4_1, mu2, S4_2,
              "4. Experiment 1: Different Means and Covariances")

mu1 = np.array([5., 5.])
mu2 = np.array([8., 8.])
S4_1 = np.eye(2)
S4_2 = np.eye(2)
decision_plot(mu1, S4_1, mu2, S4_2,
              "4. Experiment 2: Different Means, Same Covariance")

mu1 = np.array([6., 6.])
mu2 = np.array([6., 6.])
S4_1 = np.array([[4., 1.5], [1.5, 2.]])
S4_2 = np.array([[2., -.8], [-.8, 3.]])
decision_plot(mu1, S4_1, mu2, S4_2,
              "4. Experiment 3: Same Mean, Different Covariances")