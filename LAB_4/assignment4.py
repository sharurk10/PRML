import numpy as np
import matplotlib.pyplot as plt

# Fix the random seed so the same random values are generated every time
np.random.seed(42)

# Set the number of samples
N = 10000

# Define the mean vector for the 2-D Gaussian data
mu = np.array([5., 5.])

# Define the constant-density levels to be plotted
levels = [1, 2, 3]

# Generate the first independent Gaussian random variable
D1 = np.random.normal(mu[0], 1, N)

# Generate the second independent Gaussian random variable
D2 = np.random.normal(mu[1], 1, N)

# Combine the two Gaussian variables to form 2-D observations
X = np.column_stack((D1, D2))

# Function to calculate the square root of a covariance matrix
def cov_sqrt(S):
    # Calculate eigenvalues and eigenvectors of the covariance matrix
    l, V = np.linalg.eigh(S)

    # Construct the square root of the covariance matrix
    return V @ np.diag(np.sqrt(l)) @ V.T

# Function to generate data with the required covariance matrix
def generate(S):
    # Transform the original data using the square root of covariance
    return mu + (X - mu) @ cov_sqrt(S).T

# Function to analyze mean, covariance, eigenvalues and eigenvectors
def analyze(Y, S, title):
    # Estimate the mean of the generated data
    m = Y.mean(0)

    # Estimate the covariance matrix from the generated data
    C = np.cov(Y, rowvar=False)

    # Calculate eigenvalues and eigenvectors of the estimated covariance
    l, V = np.linalg.eigh(C)

    # Display the experiment title and estimated results
    print(f"\n{title}")
    print("Estimated Mean =", np.round(m, 4))
    print("Required Covariance =\n", S)
    print("Estimated Covariance =\n", np.round(C, 4))

    # Calculate and display theoretical eigenvalues
    print("Theoretical Eigenvalues =", np.round(np.linalg.eigvalsh(S), 4))

    # Display estimated eigenvalues
    print("Estimated Eigenvalues =", np.round(l, 4))

    # Display estimated eigenvectors
    print("Eigenvectors =\n", np.round(V, 4))

    # Identify the eigenvalue corresponding to the major axis
    major = np.argmax(l)

    # Identify the eigenvalue corresponding to the minor axis
    minor = np.argmin(l)

    # Display the major and minor eigenvalues
    print("Major Eigenvalue =", round(l[major], 4))
    print("Minor Eigenvalue =", round(l[minor], 4))

    # Calculate and display the major semi-axis length
    print("Major Semi-axis =", round(np.sqrt(3*l[major]), 4))

    # Calculate and display the minor semi-axis length
    print("Minor Semi-axis =", round(np.sqrt(3*l[minor]), 4))

    # State the relationship between the larger eigenvalue and major axis
    print("Larger eigenvalue -> Major axis")

    # State the relationship between the smaller eigenvalue and minor axis
    print("Smaller eigenvalue -> Minor axis")

    # Create x and y ranges around the estimated mean
    x = np.linspace(m[0]-5, m[0]+5, 400)
    y = np.linspace(m[1]-5, m[1]+5, 400)

    # Create a 2-D coordinate grid
    xx, yy = np.meshgrid(x, y)

    # Combine grid coordinates into 2-D points
    P = np.dstack((xx, yy))

    # Calculate displacement of each point from the mean
    D = P - m

    # Calculate the squared Mahalanobis distance
    Q = np.einsum("...i,ij,...j->...", D, np.linalg.inv(C), D)

    # Create the figure and axes for plotting
    fig, ax = plt.subplots(figsize=(7, 7))

    # Plot the generated data points
    ax.scatter(Y[::20, 0], Y[::20, 1], s=5, alpha=.2)

    # Plot the constant-density curves
    ax.contour(xx, yy, Q, levels=levels)

    # Plot the major and minor axes using eigenvectors
    for i in range(2):
        d = np.sqrt(3*l[i]) * V[:, i]
        ax.plot([m[0]-d[0], m[0]+d[0]],
                [m[1]-d[1], m[1]+d[1]], "--")

    # Mark the estimated mean of the data
    ax.scatter(m[0], m[1], color="black", marker="x", s=70)

    # Set the title of the plot
    ax.set_title(title)

    # Set the x-axis label
    ax.set_xlabel("$x_1$")

    # Set the y-axis label
    ax.set_ylabel("$x_2$")

    # Keep the same scale on both axes
    ax.axis("equal")

    # Display grid lines
    ax.grid(alpha=.25)

    # Adjust the spacing of the plot
    plt.tight_layout()

    # Display the plot
    plt.show()

    # Return covariance, eigenvalues and eigenvectors
    return C, l, V

# Experiment 1: Isotropic covariance matrix
S1 = np.eye(2)

# Generate data using the isotropic covariance matrix
Y1 = generate(S1)

# Analyze the isotropic covariance experiment
C1, L1, V1 = analyze(Y1, S1, "1. Isotropic Covariance")

# Experiment 2: Diagonal covariance matrix
S2 = np.diag([4., 1.])

# Generate data using the diagonal covariance matrix
Y2 = generate(S2)

# Analyze the diagonal covariance experiment
C2, L2, V2 = analyze(Y2, S2, "2. Diagonal Covariance")

# Experiment 3: Full covariance matrix
S3 = np.array([[4., 1.5], [1.5, 2.]])

# Generate data using the full covariance matrix
Y3 = generate(S3)

# Analyze the full covariance experiment
C3, L3, V3 = analyze(Y3, S3, "3. Full Covariance")

# Function to calculate the 2-D Gaussian probability density
def gaussian(P, m, S):
    # Calculate displacement of points from the mean
    D = P - m

    # Calculate squared Mahalanobis distance
    q = np.einsum("...i,ij,...j->...", D, np.linalg.inv(S), D)

    # Calculate the Gaussian probability density
    return np.exp(-q/2)/(2*np.pi*np.sqrt(np.linalg.det(S)))

# Function to generate two classes and plot their Gaussian densities
def decision_plot(mu1, S1, mu2, S2, title):
    # Generate samples for class 1
    Y1 = np.random.multivariate_normal(mu1, S1, N)

    # Generate samples for class 2
    Y2 = np.random.multivariate_normal(mu2, S2, N)

    # Create x and y ranges for the density plot
    x = np.linspace(0, 13, 500)
    y = np.linspace(0, 13, 500)

    # Create a 2-D coordinate grid
    xx, yy = np.meshgrid(x, y)

    # Combine grid coordinates into 2-D points
    P = np.dstack((xx, yy))

    # Calculate Gaussian density for class 1
    G1 = gaussian(P, mu1, S1)

    # Calculate Gaussian density for class 2
    G2 = gaussian(P, mu2, S2)

    # Display the parameters of both classes
    print(f"\n{title}")
    print("Class 1 Mean =", mu1)
    print("Class 1 Covariance =\n", S1)
    print("Class 2 Mean =", mu2)
    print("Class 2 Covariance =\n", S2)

    # Create the plot for the two classes
    plt.figure(figsize=(7, 7))

    # Plot the data points belonging to class 1
    plt.scatter(Y1[::20, 0], Y1[::20, 1], s=5, alpha=.2)

    # Plot the data points belonging to class 2
    plt.scatter(Y2[::20, 0], Y2[::20, 1], s=5, alpha=.2)

    # Plot constant-density curves for class 1
    plt.contour(xx, yy, G1, levels=8)

    # Plot constant-density curves for class 2
    plt.contour(xx, yy, G2, levels=8)

    # Plot the decision boundary where both densities are equal
    plt.contour(xx, yy, G1-G2, levels=[0], linewidths=2)

    # Mark the mean of class 1
    plt.scatter(*mu1, marker="x", s=80)

    # Mark the mean of class 2
    plt.scatter(*mu2, marker="x", s=80)

    # Set the x-axis label
    plt.xlabel("$x_1$")

    # Set the y-axis label
    plt.ylabel("$x_2$")

    # Set the plot title
    plt.title(title)

    # Keep the same scale on both axes
    plt.axis("equal")

    # Display grid lines
    plt.grid(alpha=.25)

    # Adjust the spacing of the plot
    plt.tight_layout()

    # Display the plot
    plt.show()

# Experiment 4.1: Different means and different covariance matrices
mu1 = np.array([5., 5.])
mu2 = np.array([8., 8.])
S4_1 = np.array([[4., 1.], [1., 2.]])
S4_2 = np.array([[2., -.8], [-.8, 3.]])

# Generate and plot the first classification experiment
decision_plot(mu1, S4_1, mu2, S4_2,
              "4. Experiment 1: Different Means and Covariances")

# Experiment 4.2: Different means and same covariance matrices
mu1 = np.array([5., 5.])
mu2 = np.array([8., 8.])
S4_1 = np.eye(2)
S4_2 = np.eye(2)

# Generate and plot the second classification experiment
decision_plot(mu1, S4_1, mu2, S4_2,
              "4. Experiment 2: Different Means, Same Covariance")

# Experiment 4.3: Same mean and different covariance matrices
mu1 = np.array([6., 6.])
mu2 = np.array([6., 6.])
S4_1 = np.array([[4., 1.5], [1.5, 2.]])
S4_2 = np.array([[2., -.8], [-.8, 3.]])

# Generate and plot the third classification experiment
decision_plot(mu1, S4_1, mu2, S4_2,
              "4. Experiment 3: Same Mean, Different Covariances")
