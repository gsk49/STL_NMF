import pandas as pd
import numpy as np

# Read the CSV file into a DataFrame
file = np.array(pd.read_csv("00_synthetic/ruitao/square_2x2/meta.csv"), dtype=str)

# # Get the row indexes and column names
# indexes = file.index
# columns = file.columns

# file = file

# # Prepare the header with column names
# header = '\t'.join([''] + list(columns))

# # Convert the DataFrame to a NumPy array and combine with the row labels (indexes)
# data_with_labels = np.column_stack((indexes, file.to_numpy()))
samples = np.ones((file.shape[0], 1), dtype=str)
file = np.hstack((file, samples))

# Save the matrix to a TSV file with labels
np.savetxt("00_synthetic/ruitao/square_2x2/meta2.tsv", file, delimiter="\t", fmt="%s", comments='')

print("File saved as 'meta.tsv'.")
