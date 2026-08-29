import numpy as np
import matplotlib.pyplot as plt
# ================================================================
# STEP 1: DATASET LOCATION
# Function performed:
# Specify the location of the assigned dataset.
# ================================================================

DATA_FILE = "noisy_19.txt"

# ================================================================
# STEP 2: CALCULATE MEAN SQUARED ERROR
# Function performed:
# Calculate the average squared difference between
# actual and predicted values.
#
# MSE = (1/N) * SUM((actual - predicted)^2)
# ================================================================

def find_mse(actual, predicted):

    total_error = 0.0

    for i in range(len(actual)):

        error = actual[i] - predicted[i]

        total_error = total_error + error * error

    mse = total_error / len(actual)

    return mse
#step -3 load dataset
def load_dataset(filename):

    x = []
    y = []

    file = open(filename, "r")
    for line in file:

        line = line.strip()

        if line == "":
            continue

        parts = line.split()

        if len(parts) >= 2:

            x_value = float(parts[0])

            y_value = float(parts[1])

            x.append(x_value)

            y.append(y_value)

    file.close()

    return x, y
#step 4-:splitting dataset
def create_split(x, y):

    x_train = []
    y_train = []

    x_test = []
    y_test = []

    x_validation = []
    y_validation = []

    total = len(x)

    # ------------------------------------------------------------
    # Divide the dataset
    # ------------------------------------------------------------

    for i in range(total):

        # --------------------------------------------------------
        # Every 10th sample is used for validation
        # --------------------------------------------------------

        if i % 10 == 0:

            x_validation.append(x[i])

            y_validation.append(y[i])

        # --------------------------------------------------------
        # Every 10th sample starting from index 1 is used
        # for testing
        # --------------------------------------------------------

        elif i % 10 == 1:

            x_test.append(x[i])

            y_test.append(y[i])

        # --------------------------------------------------------
        # Remaining samples are used for training
        # --------------------------------------------------------

        else:

            x_train.append(x[i])

            y_train.append(y[i])

    return (x_train, y_train, x_test, y_test,
            x_validation, y_validation)
#step 5: creating matrix
def create_polynomial_matrix(x, degree):

    matrix = []

    # ------------------------------------------------------------
    # Create one polynomial row for every x value
    # ------------------------------------------------------------

    for value in x:

        row = []

        # --------------------------------------------------------
        # Create powers:
        #
        # x^0, x^1, x^2, ..., x^degree
        # --------------------------------------------------------

        for power in range(degree + 1):

            row.append(value ** power)

        matrix.append(row)

    # ------------------------------------------------------------
    # Convert the matrix into a NumPy array
    # so that matrix operations can be performed.
    # ------------------------------------------------------------

    return np.array(matrix, dtype=float)


# ================================================================
# STEP 6: TRANSPOSE MATRIX
# Function performed:
# Calculate the transpose of the polynomial design matrix.
#
# Example:
#
# X = [1  x1]
#     [1  x2]
#     [1  x3]
#
# X^T = [1  1  1]
#       [x1 x2 x3]
#
# The transpose is calculated using NumPy.
# ================================================================

def transpose(matrix):

    return np.transpose(matrix)
# ================================================================
# STEP 7: MATRIX MULTIPLICATION
# Function performed:
# Multiply two matrices.
#
# This function is used to calculate:
#
# X^T X
#
# and
#
# X^T y
#
# The multiplication is implemented manually.
# ================================================================

def matrix_multiply(A, B):

    rows_A = len(A)

    columns_A = len(A[0])

    rows_B = len(B)

    columns_B = len(B[0])

    # ------------------------------------------------------------
    # Check whether matrix multiplication is possible
    # ------------------------------------------------------------

    if columns_A != rows_B:

        print("Matrix multiplication error.")

        return None

    result = []

    # ------------------------------------------------------------
    # Perform matrix multiplication
    # ------------------------------------------------------------

    for i in range(rows_A):

        row = []

        for j in range(columns_B):

            total = 0.0

            for k in range(columns_A):

                total = (
                    total
                    +
                    A[i][k] * B[k][j]
                )

            row.append(total)

        result.append(row)

    return np.array(result, dtype=float)
# ================================================================
# STEP 8: FIT POLYNOMIAL REGRESSION MODEL
# Function performed:
# Calculate polynomial regression coefficients using
# the Least Squares Normal Equation.
#
# Normal Equation:
#
# w = (X^T X)^(-1) X^T y
#
# NumPy inverse is used to calculate:
#
# (X^T X)^(-1)

def fit_polynomial(x_train, y_train, degree):

    # ------------------------------------------------------------
    # Create polynomial design matrix using ORIGINAL X values
    # ------------------------------------------------------------

    X = create_polynomial_matrix(
        x_train,
        degree
    )

    # ------------------------------------------------------------
    # Convert y values into a column matrix
    # ------------------------------------------------------------

    Y = []

    for value in y_train:

        Y.append([value])

    Y = np.array(Y, dtype=float)
    XT = transpose(X)
    A = matrix_multiply(
        XT,
        X
    )
    b = matrix_multiply(
        XT,
        Y
    )

    # ------------------------------------------------------------
    # Calculate inverse of X^T X
    #
    # (X^T X)^(-1)
    #
    # NumPy inverse is used here.
    # ------------------------------------------------------------

    inverse_A = np.linalg.inv(A)
    weights = inverse_A @ b

    # ------------------------------------------------------------
    # Convert the column matrix into a one-dimensional array
    # ------------------------------------------------------------

    weights = weights.flatten()

    return weights
#Step 9 Predicting outputs
def predict_values(x, weights):

    predictions = []

    # ------------------------------------------------------------
    # Determine polynomial degree
    # ------------------------------------------------------------

    degree = len(weights) - 1

    # ------------------------------------------------------------
    # Predict y for each x value
    # ------------------------------------------------------------

    for value in x:

        prediction = 0.0

        # --------------------------------------------------------
        # Calculate:
        #
        # w0 + w1*x + w2*x^2 + ... + wn*x^n
        # --------------------------------------------------------

        for power in range(degree + 1):

            prediction = (
                prediction
                +
                weights[power]
                *
                (value ** power)
            )

        predictions.append(prediction)

    return predictions


# ================================================================
# STEP 10: CALCULATE RMSE
# Function performed:
# Calculate Root Mean Squared Error.
#
# RMSE = sqrt(MSE)
# ================================================================

def calculate_rmse(mse):

    return mse ** 0.5


# ================================================================
# STEP 11: CALCULATE R-SQUARED
# Function performed:
# Calculate R-squared.
#
# R^2 = 1 - SSE / SST
#
# Higher R^2 generally indicates a better fit.
# ================================================================

def calculate_r_squared(actual, predicted):

    # ------------------------------------------------------------
    # Calculate mean of actual y values
    #
    # NOTE:
    # This is NOT normalization.
    # It is only required for calculating R^2.
    # ------------------------------------------------------------

    total = 0.0

    for value in actual:

        total = total + value

    mean = total / len(actual)

    # ------------------------------------------------------------
    # Calculate SSE
    #
    # SSE = SUM((actual - predicted)^2)
    # ------------------------------------------------------------

    sse = 0.0

    for i in range(len(actual)):

        error = actual[i] - predicted[i]

        sse = (
            sse
            +
            error * error
        )

    # ------------------------------------------------------------
    # Calculate SST
    #
    # SST = SUM((actual - mean)^2)
    # ------------------------------------------------------------

    sst = 0.0

    for value in actual:

        difference = value - mean

        sst = (
            sst
            +
            difference * difference
        )

    # ------------------------------------------------------------
    # Avoid division by zero
    # ------------------------------------------------------------

    if sst == 0:

        return 0.0

    # ------------------------------------------------------------
    # Calculate R-squared
    # ------------------------------------------------------------

    r_squared = 1.0 - (sse / sst)

    return r_squared


# ================================================================
# STEP 12: PRINT POLYNOMIAL EQUATION
# Function performed:
# Display the learned polynomial equation using the
# calculated regression coefficients.
# ================================================================

def print_polynomial(weights):

    degree = len(weights) - 1

    print()

    print("Polynomial Equation:")

    equation = "y = "

    # ------------------------------------------------------------
    # Construct polynomial equation
    # ------------------------------------------------------------

    for i in range(degree + 1):

        coefficient = weights[i]

        # --------------------------------------------------------
        # Constant term
        # --------------------------------------------------------

        if i == 0:

            equation = equation + (
                str(round(coefficient, 6))
            )

        else:

            # ----------------------------------------------------
            # Positive coefficient
            # ----------------------------------------------------

            if coefficient >= 0:

                equation = equation + " + "

            # ----------------------------------------------------
            # Negative coefficient
            # ----------------------------------------------------

            else:

                equation = equation + " - "

                coefficient = -coefficient

            equation = equation + (
                str(round(coefficient, 6))
            )

            equation = equation + "*x"

            # ----------------------------------------------------
            # Add exponent for powers greater than 1
            # ----------------------------------------------------

            if i > 1:

                equation = (
                    equation
                    +
                    "^"
                    +
                    str(i)
                )

    print(equation)


# ================================================================
# STEP 13: PLOT THE FITTED POLYNOMIAL CURVE
# Function performed:
# Plot the original data points and the fitted polynomial curve.
#
# ORIGINAL X VALUES ARE USED DIRECTLY.
# NO NORMALIZATION OR STANDARDIZATION IS PERFORMED.
# ================================================================

def plot_fitted_curve(x, y, weights, degree):
    # Sort x values so the fitted curve is drawn smoothly.
    sorted_x = sorted(x)

    # Calculate fitted y values using the selected polynomial.
    fitted_y = predict_values(sorted_x, weights)

    # Create the graph.
    plt.figure(figsize=(10, 6))

    # Plot the original noisy observations.
    plt.scatter(x, y, s=8, alpha=0.4, label="Original Data")

    # Plot the fitted polynomial curve.
    plt.plot(sorted_x, fitted_y, linewidth=2, color='orange', label="Fitted Polynomial")

    # Add title and axis labels.
    plt.title("Polynomial Regression - Fitted Curve")
    plt.xlabel("x")
    plt.ylabel("y")

    # Display the selected polynomial degree.
    plt.text(0.02, 0.95, "Polynomial Degree = " + str(degree),
             transform=plt.gca().transAxes)

    # Add legend, grid and display the graph.
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ================================================================
# STEP 13: RUN REQUIRED ASSIGNMENT
# Function performed:
# Train polynomial regression models for degrees 1 to 15.
#
# The TEST SET is used to select the best polynomial degree.
#
# The VALIDATION SET is not used for model selection.
# ================================================================

def run_assignment(x, y):

    print()

    print("=" * 75)

    print("POLYNOMIAL REGRESSION")

    print("=" * 75)

    # ------------------------------------------------------------
    # Split dataset into training, testing and validation sets
    # ------------------------------------------------------------

    (
        x_train,y_train,x_test,y_test,x_validation,y_validation
    ) = create_split(x, y)

    # ------------------------------------------------------------
    # Display dataset sizes
    # ------------------------------------------------------------

    print()

    print(
        "Total samples      :",
        len(x)
    )

    print(
        "Training samples   :",
        len(x_train)
    )

    print(
        "Testing samples    :",
        len(x_test)
    )

    print(
        "Validation samples :",
        len(x_validation)
    )

    # ------------------------------------------------------------
    # Initialize variables for best model
    # ------------------------------------------------------------

    best_degree = None

    best_test_mse = float("inf")

    best_weights = None

    # ------------------------------------------------------------
    # Display degree comparison heading
    # ------------------------------------------------------------

    print()

    print("=" * 75)

    print("DEGREE COMPARISON")

    print("=" * 75)

    print()

    print(
        "{:>8} | {:>18} | {:>18}".format(
            "Degree",
            "Train MSE",
            "Test MSE"
        )
    )

    print("-" * 75)

    # ------------------------------------------------------------
    # Test polynomial degrees from 1 to 15
    # ------------------------------------------------------------

    for degree in range(1, 16):

        try:

            # ----------------------------------------------------
            # Train polynomial regression model
            # ----------------------------------------------------

            weights = fit_polynomial(
                x_train,
                y_train,
                degree
            )

            # ----------------------------------------------------
            # Predict training values
            # ----------------------------------------------------

            train_prediction = predict_values(
                x_train,
                weights
            )

            # ----------------------------------------------------
            # Predict testing values
            # ----------------------------------------------------

            test_prediction = predict_values(
                x_test,
                weights
            )

            # ----------------------------------------------------
            # Calculate training MSE
            # ----------------------------------------------------

            train_error = find_mse(
                y_train,
                train_prediction
            )

            # ----------------------------------------------------
            # Calculate testing MSE
            # ----------------------------------------------------

            test_error = find_mse(
                y_test,
                test_prediction
            )

            # ----------------------------------------------------
            # Display results for current degree
            # ----------------------------------------------------

            print(
                "{:>8} | {:>18.6f} | {:>18.6f}".format(
                    degree,
                    train_error,
                    test_error
                )
            )

            # ----------------------------------------------------
            # Select degree having the lowest Test MSE
            # ----------------------------------------------------

            if test_error < best_test_mse:

                best_test_mse = test_error

                best_degree = degree

                best_weights = weights

        except np.linalg.LinAlgError:

            # ----------------------------------------------------
            # Handle singular matrix
            # ----------------------------------------------------

            print(
                "{:>8} | Matrix inversion failed".format(
                    degree
                )
            )

    print("-" * 75)

    # ------------------------------------------------------------
    # Display the best model
    # ------------------------------------------------------------

    print()

    print("=" * 75)

    print("BEST MODEL")

    print("=" * 75)

    print()

    print(
        "Best polynomial degree:",
        best_degree
    )

    print()

    print(
        "Best test MSE:",
        best_test_mse
    )

    print()

    print(
        "Best test RMSE:",
        calculate_rmse(best_test_mse)
    )

    # ------------------------------------------------------------
    # Display polynomial equation
    # ------------------------------------------------------------

    print_polynomial(
        best_weights
    )

    # ------------------------------------------------------------
    # Display regression coefficients
    # ------------------------------------------------------------

    print()

    print("Regression Coefficients:")

    print("-" * 40)

    for i in range(len(best_weights)):

        print(
            "w{} = {:.10f}".format(
                i,
                best_weights[i]
            )
        )

    print("-" * 40)

    # ------------------------------------------------------------
    # Predict validation data using the selected model
    # ------------------------------------------------------------

    validation_prediction = predict_values(
        x_validation,
        best_weights
    )

    # ------------------------------------------------------------
    # Calculate validation MSE
    # ------------------------------------------------------------

    validation_mse = find_mse(
        y_validation,
        validation_prediction
    )

    # ------------------------------------------------------------
    # Calculate validation RMSE
    # ------------------------------------------------------------

    validation_rmse = calculate_rmse(
        validation_mse
    )

    # ------------------------------------------------------------
    # Calculate validation R-squared
    # ------------------------------------------------------------

    validation_r_squared = calculate_r_squared(
        y_validation,
        validation_prediction
    )

    # ------------------------------------------------------------
    # Display final validation results
    # ------------------------------------------------------------

    print()

    print("=" * 75)

    print("FINAL VALIDATION RESULTS")

    print("=" * 75)

    print()

    print(
        "Selected polynomial degree :",
        best_degree
    )

    print()

    print(
        "Validation MSE             :",
        validation_mse
    )

    print()

    print(
        "Validation RMSE            :",
        validation_rmse
    )

    print()

    print(
        "Validation R-squared       :",
        validation_r_squared
    )

    print()

    print("=" * 75)

    # ------------------------------------------------------------
    # Display sample validation predictions
    # ------------------------------------------------------------

    print()

    print("SAMPLE VALIDATION PREDICTIONS")

    print("-" * 75)

    print(
        "{:>15} {:>20} {:>20}".format(
            "x",
            "Actual y",
            "Predicted y"
        )
    )

    print("-" * 75)

    number_to_display = 10

    if len(x_validation) < number_to_display:

        number_to_display = len(x_validation)

    # ------------------------------------------------------------
    # Display first 10 validation predictions
    # ------------------------------------------------------------

    for i in range(number_to_display):

        print(
            "{:>15.6f} {:>20.6f} {:>20.6f}".format(
                x_validation[i],
                y_validation[i],
                validation_prediction[i]
            )
        )

    print("-" * 75)

    # ------------------------------------------------------------
    # Return final results
    # ------------------------------------------------------------

    return (
        best_degree,
        best_weights,
        validation_mse,
        validation_rmse,
        validation_r_squared
    )
#Additional
def additional_experiment(x, y):

    print()

    print("=" * 75)

    print("ADDITIONAL DEGREE EXPERIMENT")

    print("=" * 75)

    # ------------------------------------------------------------
    # Split dataset
    # ------------------------------------------------------------

    (
        x_train,
        y_train,
        x_test,
        y_test,
        x_validation,
        y_validation
    ) = create_split(x, y)

    # ------------------------------------------------------------
    # Polynomial degrees used for additional experiment
    #
    # High degrees are avoided because raw x values are used
    # without normalization.
    # ------------------------------------------------------------

    degrees = [
        1,
        2,
        3,
        5,
        8,
        10,
        15
    ]

    print()

    print(
        "{:>8} | {:>18} | {:>18} | {:>18}".format(
            "Degree",
            "Train MSE",
            "Test MSE",
            "Validation MSE"
        )
    )

    print("-" * 80)

    # ------------------------------------------------------------
    # Train each selected polynomial degree
    # ------------------------------------------------------------

    for degree in degrees:

        try:

            # ----------------------------------------------------
            # Train model
            # ----------------------------------------------------

            weights = fit_polynomial(
                x_train,
                y_train,
                degree
            )

            # ----------------------------------------------------
            # Training prediction
            # ----------------------------------------------------

            train_prediction = predict_values(
                x_train,
                weights
            )

            # ----------------------------------------------------
            # Testing prediction
            # ----------------------------------------------------

            test_prediction = predict_values(
                x_test,
                weights
            )

            # ----------------------------------------------------
            # Validation prediction
            # ----------------------------------------------------

            validation_prediction = predict_values(
                x_validation,
                weights
            )

            # ----------------------------------------------------
            # Calculate training MSE
            # ----------------------------------------------------

            train_mse = find_mse(
                y_train,
                train_prediction
            )

            # ----------------------------------------------------
            # Calculate testing MSE
            # ----------------------------------------------------

            test_mse = find_mse(
                y_test,
                test_prediction
            )

            # ----------------------------------------------------
            # Calculate validation MSE
            # ----------------------------------------------------

            validation_mse = find_mse(
                y_validation,
                validation_prediction
            )

            # ----------------------------------------------------
            # Display results
            # ----------------------------------------------------

            print(
                "{:>8} | {:>18.6f} | {:>18.6f} | {:>18.6f}".format(
                    degree,
                    train_mse,
                    test_mse,
                    validation_mse
                )
            )

        except np.linalg.LinAlgError:

            print(
                "{:>8} | Matrix inversion failed".format(
                    degree
                )
            )

    print("-" * 80)


# ================================================================
# STEP 15: MAIN PROGRAM
# Function performed:
# Load the dataset and execute the complete experiment.
# ================================================================

if __name__ == "__main__":

    # ------------------------------------------------------------
    # Load the dataset
    # ------------------------------------------------------------

    print()

    print("Reading dataset...")

    x_values, y_values = load_dataset(
        DATA_FILE
    )

    # ------------------------------------------------------------
    # Check whether dataset was loaded
    # ------------------------------------------------------------

    if len(x_values) == 0:

        print("Error: Dataset could not be loaded.")

    else:

        # --------------------------------------------------------
        # Display number of samples
        # --------------------------------------------------------

        print("Dataset loaded:", len(x_values), "samples")

        # --------------------------------------------------------
        # Run the required assignment experiment
        # --------------------------------------------------------

        results = run_assignment(
            x_values,
            y_values
        )

        # --------------------------------------------------------
        # Run additional experiment
        # --------------------------------------------------------

        additional_experiment(x_values, y_values)

        # --------------------------------------------------------
        # Display the fitted polynomial curve
        # --------------------------------------------------------

        plot_fitted_curve(x_values, y_values, results[1], results[0])