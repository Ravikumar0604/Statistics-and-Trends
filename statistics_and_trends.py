#!/usr/bin/env python
# coding: utf-8


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns


def plot_relational_plot(df):
    """
    Generates, displays, and saves a line plot showing the total sales over time.

    Parameters:
        df (pd.DataFrame): The dataset containing 'Date' and 'Total'.
    """
    fig, ax = plt.subplots()
    if 'Date' in df.columns and 'Total' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df_time_series = df.groupby(df['Date'].dt.month)['Total'].sum()
        df_time_series.plot(ax=ax, linestyle='-', marker='o', alpha=0.7)
        plt.title('Relational Plot: Total Sales Over Time')
        plt.xlabel('Month')
        plt.ylabel('Total Sales')
        plt.grid(True)
    plt.savefig('relational_plot.png')
    plt.show()


def plot_categorical_plot(df):
    """
    Generates, displays, and saves a bar plot of the most common product lines.

    
    """
    fig, ax = plt.subplots()
    if 'Product line' in df.columns:
        df['Product line'].value_counts().plot(kind='bar', colormap="coolwarm")
        plt.title('Categorical Plot: Distribution of Product Lines')
        plt.xlabel('Product Line')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('categorical_plot.png')
    plt.show()


def plot_statistical_plot(df):
    """
    Generates, displays, and saves a correlation heatmap for numerical variables.

    
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt='.2f', linewidths=0.5)
    plt.title('Statistical Plot: Correlation Heatmap')
    plt.savefig('statistical_plot.png')
    plt.show()


def statistical_analysis(df, col: str):
    """
    Computes and returns the mean, standard deviation, skewness, and excess kurtosis for a given column.

    
    """
    mean = df[col].mean()
    stddev = df[col].std()
    skew = df[col].skew()
    excess_kurtosis = df[col].kurtosis()
    return mean, stddev, skew, excess_kurtosis


def preprocessing(df):
    """
    Preprocesses the dataset by handling missing values, converting data types, and cleaning categorical variables.
    Additionally, it provides quick insights using describe, head/tail, and correlation.

    
    """
    df = df.copy()

    # Convert date columns to datetime format
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Convert numeric columns to float
    for col in ['Unit price', 'Total', 'Tax 5%', 'COGS', 'Gross income']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fill missing values in numerical columns with the median value
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())

    # Fill missing values in categorical columns with the mode
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Display quick insights
    print("Dataset Summary:\n", df.describe(include='all'))
    print("\nFirst five rows:\n", df.head())
    print("\nLast five rows:\n", df.tail())
    print("\nCorrelation Matrix:\n", df.select_dtypes(include=[np.number]).corr())

    return df


def writing(moments, col):
    """
    Prints a summary of statistical analysis results for a given column.

  
    """
    print(f'For the attribute {col}:')
    print(f'Mean = {moments[0]:.2f}, Standard Deviation = {moments[1]:.2f}, Skewness = {moments[2]:.2f}, and Excess Kurtosis = {moments[3]:.2f}.')

    skewness_desc = ("right-skewed" if moments[2] > 0 else "left-skewed" if moments[2] < 0 else "not skewed")
    kurtosis_desc = ("leptokurtic" if moments[3] > 0 else "platykurtic" if moments[3] < 0 else "mesokurtic")

    print(f'The data was {skewness_desc} and {kurtosis_desc}.')


def main():
    """
    Main function that loads data, applies preprocessing, generates plots, performs statistical analysis, and prints results.
    """
    df = pd.read_csv('data.csv')
    df = preprocessing(df)
    col = 'Total'  # Chosen column for statistical analysis
    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)
    moments = statistical_analysis(df, col)
    writing(moments, col)


if __name__ == '__main__':
    main()
