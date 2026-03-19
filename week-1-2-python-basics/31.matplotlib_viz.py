"""
What is Matplotlib?
-------------------
Matplotlib is the most widely used Python library for creating static, animated, and interactive visualizations.
It allows you to create a wide variety of plots (line, scatter, bar, histogram, etc.) with fine control over every element.

Why is it important?
--------------------
- Essential for data analysis and scientific computing in Python.
- Lets you explore, understand, and communicate data visually.
- Used in research, engineering, finance, and machine learning for plotting trends, distributions, and results.
- Forms the foundation for higher-level libraries like Seaborn and Pandas plotting.

Seaborn is built on top of matplotlib and provides a high-level interface for attractive statistical graphics.


Common plot types:
- Line plot: Shows trends over time or ordered categories.
- Scatter plot: Shows relationship between two variables.
- Bar plot: Compares quantities across categories.
- Histogram: Shows distribution of a variable.

Key functions and methods:
- plt.figure(): Creates a new figure window for plotting. Useful for making multiple plots in one script.
- plt.plot(x, y): Plots a line graph of y vs x.
- plt.scatter(x, y): Plots a scatter graph of y vs x.
- plt.bar(x, y): Plots a bar chart.
- plt.hist(data, bins): Plots a histogram of data with specified number of bins.
- plt.xlabel(), plt.ylabel(): Set the label for the x and y axes.
- plt.title(): Sets the title of the plot.
- plt.legend(): Shows the legend for labeled plot elements.
- plt.show(): Displays the plot window. Required to actually see the plot in most environments.

Seaborn is built on top of matplotlib and provides a high-level interface for attractive statistical graphics.
- sns.pairplot(df, hue='col'): Plots pairwise relationships in a dataset, colored by 'col'.

Mini Project: Data Dashboard
- Load a dataset (e.g., iris)
- Clean missing values (df.dropna())
- Plot distributions and correlations (sns.pairplot)
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def basic_plots():
    """ """
    x = [1, 2, 3, 4, 5]
    y = [2, 3, 5, 7, 11]
    plt.figure()  # Creates a new figure window for plotting
    plt.plot(x, y, label='Line')  # Line plot
    plt.scatter(x, y, label='Scatter')  # Scatter plot
    plt.bar(x, y, label='Bar')  # Bar plot
    plt.xlabel('X axis')  # Label for x-axis
    plt.ylabel('Y axis')  # Label for y-axis
    plt.title('Basic Plots')  # Title of the plot
    plt.legend()  # Show legend for all labeled plots
    plt.show()  # Display the plot

#plotting a histogram
def histogram_plot():
    """ """
    data = [1, 2, 2, 3, 3, 3, 4, 4, 5]
    plt.figure()  # New figure for histogram
    plt.hist(data, bins=5)  # Histogram with 5 bins
    plt.title('Histogram')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()

#data dashboard with seaborn
def data_dashboard():
    """ """
    df = sns.load_dataset('iris')
    # Clean missing values (if any)
    df = df.dropna()
    # Pairplot for correlations
    sns.pairplot(df, hue='species')
    plt.suptitle('Iris Data Dashboard', y=1.02)
    plt.show()

if __name__ == "__main__":
    basic_plots()
    histogram_plot()
    data_dashboard()
