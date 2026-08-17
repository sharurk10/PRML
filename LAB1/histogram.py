import matplotlib.pyplot as plt
with open("uniform_data.txt", "r") as file:
 uniform_data = [
 float(line.strip())
 for line in file
 ]
with open("gaussian_data.txt", "r") as file:
 gaussian_data = [
 float(line.strip())
 for line in file
 ]
plt.figure(figsize=(10, 6))
plt.hist(
 uniform_data,
 bins=10,
 edgecolor="black"
)
plt.title(
 "Histogram of Uniform Random Variables"
)
plt.xlabel(
 "Value"
)
plt.ylabel(
 "Frequency"
)
plt.grid(
 True,
 alpha=0.3
)
plt.tight_layout()
plt.savefig(
 "uniform_histogram.png",
 dpi=300
)
plt.show()
plt.figure(figsize=(10, 6))
plt.hist(
 gaussian_data,
 bins=10,
 edgecolor="black"
)
plt.title(
 "Histogram of Gaussian Random Variables"
)
plt.xlabel(
 "Value"
)
plt.ylabel(
 "Frequency"
)
plt.grid(
 True,
 alpha=0.3
)
plt.tight_layout()
plt.savefig(
 "gaussian_histogram.png",
 dpi=300
)
plt.show()