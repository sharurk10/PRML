import matplotlib.pyplot as plt
data = []

with open("noisy_19.txt", "r") as file:
    for line in file:
        values = line.split()

        x_value = float(values[0])
        y_value = float(values[1])

        data.append([x_value, y_value])

print("Number of data points:", len(data))

print("First 5 data points:")
for i in range(5):
    print(data[i])

# Separate x and y values

x = []
y = []

for point in data:
    x.append(point[0])
    y.append(point[1])

print("\nNumber of x values:", len(x))
print("Number of y values:", len(y))

print("First 5 x values:", x[:5])
print("First 5 y values:", y[:5])

# Find minimum and maximum values

x_min = min(x)
x_max = max(x)

y_min = min(y)
y_max = max(y)

print("\nMinimum x:", x_min)
print("Maximum x:", x_max)

print("Minimum y:", y_min)
print("Maximum y:", y_max)

# Normalize x using Min-Max normalization

x_normalized = []

for value in x:
    normalized_value = (value - x_min) / (x_max - x_min)
    x_normalized.append(normalized_value)

print("\nFirst 5 normalized x values:")
print(x_normalized[:5])

print("Minimum normalized x:", min(x_normalized))
print("Maximum normalized x:", max(x_normalized))

# Combine normalized x and y into pairs

normalized_data = []

for i in range(len(x_normalized)):
    normalized_data.append([x_normalized[i], y[i]])

print("\nNumber of normalized data points:", len(normalized_data))
print("First 5 normalized data points:")

for i in range(5):
    print(normalized_data[i])

import random

# Set a fixed seed so that the same split is obtained every time
random.seed(42)

# Shuffle the data
random.shuffle(normalized_data)


# Split the data into training, test and validation sets

train_data = normalized_data[:7000]
test_data = normalized_data[7000:8500]
validation_data = normalized_data[8500:]

print("\nTraining samples:", len(train_data))
print("Test samples:", len(test_data))
print("Validation samples:", len(validation_data))

# Separate x and y for each dataset

x_train = []
y_train = []

for point in train_data:
    x_train.append(point[0])
    y_train.append(point[1])


x_test = []
y_test = []

for point in test_data:
    x_test.append(point[0])
    y_test.append(point[1])


x_validation = []
y_validation = []

for point in validation_data:
    x_validation.append(point[0])
    y_validation.append(point[1])


print("\nTraining x values:", len(x_train))
print("Training y values:", len(y_train))

print("Test x values:", len(x_test))
print("Test y values:", len(y_test))

print("Validation x values:", len(x_validation))
print("Validation y values:", len(y_validation))

# Create polynomial features

def create_polynomial_features(x_values, degree):

    X = []

    for value in x_values:

        row = []

        for power in range(degree + 1):
            row.append(value ** power)

        X.append(row)

    return X

# Matrix multiplication

def matrix_multiply(A, B):

    rows_A = len(A)
    cols_A = len(A[0])

    rows_B = len(B)
    cols_B = len(B[0])

    if cols_A != rows_B:
        raise ValueError("Matrix dimensions are not compatible")

    result = []

    for i in range(rows_A):
        row = []

        for j in range(cols_B):
            total = 0.0

            for k in range(cols_A):
                total += A[i][k] * B[k][j]

            row.append(total)

        result.append(row)

    return result

# Matrix transpose

def transpose(A):

    rows = len(A)
    cols = len(A[0])

    result = []

    for j in range(cols):
        row = []

        for i in range(rows):
            row.append(A[i][j])

        result.append(row)

    return result

# Matrix inverse using Gauss-Jordan elimination

def matrix_inverse(A):

    n = len(A)

    # Create augmented matrix [A | I]
    augmented = []

    for i in range(n):
        row = []

        for j in range(n):
            row.append(float(A[i][j]))

        for j in range(n):
            if i == j:
                row.append(1.0)
            else:
                row.append(0.0)

        augmented.append(row)

    # Gauss-Jordan elimination
    for i in range(n):

        # Find a suitable pivot
        pivot = i

        for j in range(i + 1, n):
            if abs(augmented[j][i]) > abs(augmented[pivot][i]):
                pivot = j

        # Check whether matrix is singular
        if abs(augmented[pivot][i]) < 1e-12:
            raise ValueError("Matrix is singular and cannot be inverted")

        # Swap rows
        augmented[i], augmented[pivot] = augmented[pivot], augmented[i]

        # Make pivot equal to 1
        pivot_value = augmented[i][i]

        for j in range(2 * n):
            augmented[i][j] /= pivot_value

        # Make other entries in this column zero
        for k in range(n):

            if k != i:

                factor = augmented[k][i]

                for j in range(2 * n):
                    augmented[k][j] -= factor * augmented[i][j]

    # Extract inverse matrix
    inverse = []

    for i in range(n):
        row = []

        for j in range(n, 2 * n):
            row.append(augmented[i][j])

        inverse.append(row)

    return inverse

# Predict output using the polynomial coefficients

def predict(X, weights):

    predictions = []

    for row in X:

        value = 0.0

        for j in range(len(weights)):
            value += row[j] * weights[j][0]

        predictions.append(value)

    return predictions


# Calculate Mean Squared Error

def calculate_mse(actual, predicted):

    total_error = 0.0

    for i in range(len(actual)):
        error = actual[i] - predicted[i]
        total_error += error ** 2

    mse = total_error / len(actual)

    return mse

# Run polynomial regression for a given degree

def run_polynomial_regression(degree):

    # Create polynomial features for training data
    X_train = create_polynomial_features(x_train, degree)

    # Create polynomial features for test data
    X_test = create_polynomial_features(x_test, degree)

    # Convert training y values into a column matrix
    y_train_matrix = []

    for value in y_train:
        y_train_matrix.append([value])

    # Calculate X^T
    X_transpose = transpose(X_train)

    # Calculate X^T X
    X_transpose_X = matrix_multiply(X_transpose, X_train)

    # Calculate X^T y
    X_transpose_y = matrix_multiply(X_transpose, y_train_matrix)

    # Calculate (X^T X)^-1
    X_transpose_X_inverse = matrix_inverse(X_transpose_X)

    # Calculate weights
    weights = matrix_multiply(
        X_transpose_X_inverse,
        X_transpose_y
    )

    # Make predictions
    train_predictions = predict(X_train, weights)
    test_predictions = predict(X_test, weights)

    # Calculate MSE
    train_mse = calculate_mse(y_train, train_predictions)
    test_mse = calculate_mse(y_test, test_predictions)

    return weights, train_mse, test_mse

# Compare different polynomial degrees

print("\nPolynomial Degree Comparison")
print("-----------------------------------------")
print("Degree\tTraining MSE\tTest MSE")

results = []

for degree in range(1, 11):

    weights, train_mse, test_mse = run_polynomial_regression(degree)

    results.append([degree, train_mse, test_mse])

    print(
        degree,
        "\t",
        train_mse,
        "\t",
        test_mse
    )
#############################################################
# Select the best polynomial degree based on test MSE

# Select the degree with the lowest test MSE

best_result = results[0]

for result in results:
    if result[2] < best_result[2]:
        best_result = result

best_degree = best_result[0]
best_test_mse = best_result[2]

print("\nSelected polynomial degree:", best_degree)
print("Best test MSE:", best_test_mse)

# Create polynomial features for the selected degree

X_train_best = create_polynomial_features(
    x_train,
    best_degree
)

X_validation_best = create_polynomial_features(
    x_validation,
    best_degree
)

# Convert training y values into a column matrix

y_train_matrix = []

for value in y_train:
    y_train_matrix.append([value])


# Calculate X^T

X_transpose = transpose(X_train_best)

# Calculate X^T X

X_transpose_X = matrix_multiply(
    X_transpose,
    X_train_best
)

# Calculate X^T y

X_transpose_y = matrix_multiply(
    X_transpose,
    y_train_matrix
)

# Calculate inverse of X^T X

X_transpose_X_inverse = matrix_inverse(
    X_transpose_X
)

# Calculate the weights

best_weights = matrix_multiply(
    X_transpose_X_inverse,
    X_transpose_y
)

# Predict the validation data

validation_predictions = predict(
    X_validation_best,
    best_weights
)

# Calculate validation MSE

validation_mse = calculate_mse(
    y_validation,
    validation_predictions
)

print("\nValidation MSE:", validation_mse)

# Display the coefficients of the selected polynomial

print("\nCoefficients of the selected degree-8 polynomial:")

for i in range(len(best_weights)):
    print("w" + str(i) + " =", best_weights[i][0])


import math

validation_rmse = math.sqrt(validation_mse)

print("Validation RMSE:", validation_rmse)



# Plot the original data and the selected polynomial

x_plot = []

for i in range(500):
    value = i / 499
    x_plot.append(value)

X_plot = create_polynomial_features(x_plot, best_degree)

y_plot = predict(X_plot, best_weights)

# Plot training data and selected polynomial

plt.figure(figsize=(10, 6))

plt.scatter(
    x_train,
    y_train,
    s=5,
    label="Training data"
)

plt.plot(
    x_plot,
    y_plot,
    linewidth=2,
    label="Degree 8 polynomial"
)

plt.xlabel("Normalized x")
plt.ylabel("y")
plt.title("Team 19 - Degree 8 Polynomial Regression")
plt.legend()
plt.grid(True)

plt.show()

###########################
# Plot MSE versus polynomial degree

degrees = []
training_errors = []
test_errors = []

for result in results:
    degrees.append(result[0])
    training_errors.append(result[1])
    test_errors.append(result[2])

plt.figure(figsize=(10, 6))

plt.plot(
    degrees,
    training_errors,
    marker="o",
    label="Training MSE"
)

plt.plot(
    degrees,
    test_errors,
    marker="o",
    label="Test MSE"
)

plt.xlabel("Polynomial Degree")
plt.ylabel("Mean Squared Error")
plt.title("Team 19 - MSE vs Polynomial Degree")
plt.legend()
plt.grid(True)

plt.show()