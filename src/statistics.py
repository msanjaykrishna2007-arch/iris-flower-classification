import pandas as pd

# Load the cleaned Iris dataset
data = pd.read_csv("data/iris_cleaned.csv")

# Define measurement columns
measurements = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width"
]

# Check dataset information
print("Dataset shape:", data.shape)
print("Species:", data["species"].unique())

# Calculate mean measurements by species
mean_by_species = data.groupby("species")[measurements].mean()

print("\nMean measurements by species:")
print(mean_by_species.round(2))

# Calculate median measurements by species
median_by_species = data.groupby("species")[measurements].median()

print("\nMedian measurements by species:")
print(median_by_species.round(2))

# Calculate standard deviation by species
std_by_species = data.groupby("species")[measurements].std()

print("\nStandard deviation by species:")
print(std_by_species.round(2))

# Calculate minimum measurements by species
min_by_species = data.groupby("species")[measurements].min()

print("\nMinimum measurements by species:")
print(min_by_species.round(2))

# Calculate maximum measurements by species
max_by_species = data.groupby("species")[measurements].max()

print("\nMaximum measurements by species:")
print(max_by_species.round(2))

# Combine mean and median into one summary
statistical_summary = pd.concat(
    {
        "Mean": mean_by_species,
        "Median": median_by_species
    },
    axis=1
)

# Round values for readability
rounded_summary = statistical_summary.round(2)

print("\nStatistical Summary (Mean and Median):")
print(rounded_summary)

# Save the statistical summary
rounded_summary.to_csv("data/statistical_summary.csv")

print("\nStatistical summary saved to data/statistical_summary.csv")

# Statistical observations
print("\nStatistical Observations:")
print("1. Iris-setosa has the smallest average petal length and petal width.")
print("2. Iris-virginica has the largest average petal length and petal width.")
print("3. Iris-virginica has the largest average sepal length.")
print("4. Iris-setosa has the smallest average sepal length.")
print("5. Petal measurements show clear differences between the three species.")