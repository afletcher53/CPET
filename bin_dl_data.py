import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import random
import pandas as pd
from tqdm import tqdm

from Classes.ProjectStrings import ProjectStrings

import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("./logs/main.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

def bin_time_series(df, n_bins=50):
    """
    Bin a time series DataFrame into a fixed number of bins and normalize all numeric columns.
    Works via dividing the df (Breaths) into n_bins and then taking the mean of each bin.
    The last bin will have the remainder of the breaths
    The data within the bin column is averaged and then normalized between 0 and 1

    Parameters:
    df (pd.DataFrame): Input DataFrame with time series data
    n_bins (int): Number of bins to create

    Returns:
    pd.DataFrame: Binned DataFrame with n_bins rows and all numeric columns normalized
    """

    bin_size = len(df) // n_bins
    if bin_size == 0:
        raise ValueError("Number of bins is too large for the size of the DataFrame")
    bins = [i * bin_size for i in range(n_bins + 1)]
    bins[-1] = len(df)
    labels = list(range(1, n_bins + 1))

    df['bin'] = pd.cut(df.index, bins=bins, labels=labels, include_lowest=True)


    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    agg_dict = {col: 'mean' for col in numeric_columns}
    binned_df = df.groupby('bin').agg(agg_dict).reset_index(drop=True)

    for col in numeric_columns:
        min_val = binned_df[col].min()
        max_val = binned_df[col].max()
        if min_val != max_val: 
            binned_df[col] = (binned_df[col] - min_val) / (max_val - min_val)
        else:
            binned_df[col] = 1 

    return binned_df

def bin_time_series_adaptive(df, n_bins=50, fill_empty=False):
    """
    Bin a time series DataFrame into a variable number of bins using adaptive binning,
    normalize all numeric columns, and stretch the data when actual_n_bins < n_bins.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame with time series data
    n_bins (int): Target number of bins to create
    fill_empty (bool): If True, fill empty bins with NaN values
    
    Returns:
    pd.DataFrame: Binned DataFrame with n_bins rows and all numeric columns normalized
    """
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    actual_n_bins = min(n_bins, len(df))
    points_per_bin = max(1, len(df) // actual_n_bins)
    
    # Create initial bins
    bins = [i * points_per_bin for i in range(actual_n_bins)]
    bins.append(len(df))
    df['bin'] = pd.cut(df.index, bins=bins, labels=range(1, actual_n_bins + 1), include_lowest=True)
    
    # Aggregate data
    agg_dict = {col: 'mean' for col in numeric_columns}
    binned_df = df.groupby('bin', observed=False).agg(agg_dict).reset_index()
    
    # Stretch data if actual_n_bins < n_bins
    if actual_n_bins < n_bins:
        stretch_factor = n_bins / actual_n_bins
        new_index = np.arange(1, n_bins + 1)
        old_index = np.linspace(1, n_bins, num=actual_n_bins)
        
        stretched_df = pd.DataFrame(index=new_index)
        for col in numeric_columns:
            stretched_df[col] = np.interp(new_index, old_index, binned_df[col])
        
        binned_df = stretched_df.reset_index(drop=True)
    elif fill_empty:
        # Fill empty bins if necessary
        all_bins = pd.DataFrame({'bin': range(1, n_bins + 1)})
        binned_df = pd.merge(all_bins, binned_df, on='bin', how='left')
    
    # Normalize numeric columns
    for col in numeric_columns:
        min_val = binned_df[col].min()
        max_val = binned_df[col].max()
        if pd.notna(min_val) and pd.notna(max_val) and min_val != max_val:
            binned_df[col] = (binned_df[col] - min_val) / (max_val - min_val)
        elif pd.notna(min_val) and pd.notna(max_val):
            binned_df[col] = 1
    
    # Drop unnecessary columns
    binned_df = binned_df.drop(columns=['bin', 'Phase'], errors='ignore')
    
    return binned_df


def get_files(folder, extension):
    """Get all files in a folder with a given extension."""
    import os

    files = []
    for file in os.listdir(folder):
        if file.endswith(extension):
            files.append(os.path.join(folder, file))
    if not files:
        raise FileNotFoundError(
            f"No files with extension {extension} found in folder {folder}"
        )

    return files


def plotting_functions():
    def normalize_dataframe(df):
        """Normalize all numeric columns in the dataframe to be between 0 and 1."""
        for column in df.select_dtypes(include=['float64', 'int64']).columns:
            min_val = df[column].min()
            max_val = df[column].max()
            if min_val != max_val:
                df[column] = (df[column] - min_val) / (max_val - min_val)
            else:
                df[column] = 1
        return df

    ps = ProjectStrings()
    files = get_files(ps.york_binned_normalised, ".csv")
    file = files[random.randint(0, len(files) - 1)]
    df_binned = pd.read_csv(file)

    if df_binned.select_dtypes(include=['float64', 'int64']).max().max() > 1 or \
       df_binned.select_dtypes(include=['float64', 'int64']).min().min() < 0:
        print("Binned data is not normalized. Normalizing now...")
        df_binned = normalize_dataframe(df_binned)

    unbinned_file = os.path.join(ps.york_dl, os.path.basename(file))
    df_unbinned = pd.read_csv(unbinned_file)

    plt.figure(figsize=(20, 10))

    plt.subplot(1, 2, 1)
    sns.heatmap(df_unbinned, annot=False, cmap="viridis")
    plt.title("Unbinned Data")

    plt.subplot(1, 2, 2)
    sns.heatmap(df_binned, annot=False, cmap="viridis", vmin=0, vmax=1)
    plt.title("Binned Data (Normalized)")

    plt.tight_layout()
    plt.show()

    print("Unbinned data range:")
    print(df_unbinned.select_dtypes(include=['float64', 'int64']).min().min(), "to",
          df_unbinned.select_dtypes(include=['float64', 'int64']).max().max())

    print("\nBinned data range:")
    print(df_binned.select_dtypes(include=['float64', 'int64']).min().min(), "to",
          df_binned.select_dtypes(include=['float64', 'int64']).max().max())

    def calculate_correlations(df):
        return df.corr()

    correlation_matrices = []

    def has_constant_columns(df):
        return any(df.nunique() <= 1)

    logger.info(f"Found {len(files)} files to calculate correlations")

    for file in tqdm(files, desc="Calculating correlations", unit="file"):

        df = pd.read_csv(file)
        if has_constant_columns(df):
            print(f"Skipping file {file} due to constant columns")
            continue 
        if df.select_dtypes(include=['float64', 'int64']).max().max() > 1 or \
                df.select_dtypes(include=['float64', 'int64']).min().min() < 0:
            df = normalize_dataframe(df)

        corr_matrix = calculate_correlations(df)
        correlation_matrices.append(corr_matrix)

    if correlation_matrices:
        combined_correlation_matrix = np.mean(correlation_matrices, axis=0)
        combined_correlation_df = pd.DataFrame(combined_correlation_matrix,
                                               columns=df.columns,
                                               index=df.columns)


        plt.figure(figsize=(10, 8))
        sns.heatmap(combined_correlation_df, annot=True, cmap="viridis")
        plt.title("Mean Correlation Matrix Across All Files")
        plt.show()
    else:
        print("No valid correlation matrices were found. Please check the data.")


def main():
    ps = ProjectStrings()
    logger.info("Binning and normalizing York DL data")
    files = get_files(ps.york_dl, ".csv")
    files = [file for file in files if "_single_" not in file]
    logger.info(f"Found {len(files)} files to process")
    for file in tqdm(files, desc="Processing files", unit="file"):
        df = pd.read_csv(file)
        file_name = os.path.basename(file)
        df = bin_time_series_adaptive(df, n_bins=100, fill_empty=True)
        df.to_csv(os.path.join(ps.york_binned_normalised, file_name), index=False)

if __name__ == "__main__":
    plotting_functions()
    # main()
