"""
Pandas Notes
------------
Pandas is a library for data manipulation and analysis.

Key Concepts:
- DataFrame: 2D table of data with columns and rows.
- Series: 1D labeled array (like a single column).
- pd.DataFrame(dict): Create a DataFrame from a dictionary.
- pd.read_csv('file.csv'): Read a CSV file into a DataFrame.
- df['col']: Select a column (returns a Series).
- df[df['col'] > x]: Filter rows where column 'col' > x.
- df.mean(): Mean of each column.
- df.describe(): Summary statistics for each column.
- df.groupby('col'): Group rows by column value.

Function/Method Explanations:
- pd.DataFrame(data): Creates a DataFrame from a dictionary or array.
- df['A']: Selects column 'A' as a Series.
- df['B'].mean(): Computes mean of column 'B'.
- df.describe(): Returns count, mean, std, min, max, quartiles for each column.
- df[df['A'] > 1]: Filters rows where column 'A' > 1.
"""
import pandas as pd

def pandas_examples():
    """ """
    data = {'A': [1, 2, 3], 'B': [4, 5, 6]}  # Dictionary for DataFrame
    df = pd.DataFrame(data)  # Create DataFrame from dictionary
    print("DataFrame:\n", df)
    print("Column A:", df['A'])  # Select column 'A' (Series)
    print("Mean of B:", df['B'].mean())  # Mean of column 'B'
    print("Describe:\n", df.describe())  # Summary statistics
    # Filtering rows where column 'A' > 1
    print("Rows where A > 1:\n", df[df['A'] > 1])

if __name__ == "__main__":
    pandas_examples()
