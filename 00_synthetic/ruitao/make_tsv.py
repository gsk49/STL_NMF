import pandas as pd
import numpy as np

# Read the CSV file into a DataFrame
file = pd.read_csv("00_synthetic/ruitao/square_2x2/S.csv", header=0, index_col=0)

# Get the row indexes and column names
indexes = file.index
columns = file.columns

file = file

# Prepare the header with column names
header = '\t'.join([''] + list(columns))

# Convert the DataFrame to a NumPy array and combine with the row labels (indexes)
data_with_labels = np.column_stack((indexes, file.to_numpy()))

# Save the matrix to a TSV file with labels
np.savetxt("00_synthetic/ruitao/square_2x2/S.tsv", data_with_labels, delimiter="\t", header=header, fmt="%s", comments='')

print("File saved as 'meta.tsv'.")
