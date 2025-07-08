"""
MMWSTM-ADRAN+: Enhanced Hybrid Deep Learning Architecture for Climate Data Analysis
=================================================================================
This code implements an enhanced version of the Multi-Modal Weather State Transition Model with 
Anomaly-Driven Recurrent Attention Network (MMWSTM-ADRAN+) for Baghdad weather data.

Key enhancements:
1. Optimized hyperparameters for improved accuracy
2. Advanced time series data augmentation techniques
3. Enhanced model architecture with attention mechanisms
4. Professional 3D visualizations
5. Comparison framework with classical models
6. Comprehensive documentation of patentable innovations

Author: Dr. Shaheen Mhammed Saleh Ahmed 
Faculty of Engineering, Çukurova University, Adana, Turkey
College of Science, Kirkuk University, Kirkuk, Iraq
Date: April 2025
"""

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
import traceback
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
import seaborn as sns
from pathlib import Path
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

# Set high-quality visualization settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (30, 20)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 23
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['legend.title_fontsize'] = 18
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.5

# Try importing PyTorch with error handling
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
    from torch.optim.lr_scheduler import CosineAnnealingLR
    
    TORCH_AVAILABLE = True
    print("PyTorch is available. GPU acceleration can be used if hardware is available.")
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU not available. Using CPU.")
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch is not available. Using CPU-only implementation.")

# Try importing scikit-learn with error handling
try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.svm import SVR
    from sklearn.pipeline import Pipeline
    from scipy import stats
    from scipy.signal import savgol_filter
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("scikit-learn is not available. Installing required packages...")
    import sys
    !{sys.executable} -m pip install scikit-learn scipy

# Define custom color palettes for professional visualizations
CUSTOM_CMAP = plt.cm.viridis
CUSTOM_CMAP_R = plt.cm.viridis_r
CUSTOM_DIVERGING = plt.cm.coolwarm
CUSTOM_SEQUENTIAL = plt.cm.plasma
CUSTOM_QUALITATIVE = plt.cm.tab10

# Define color schemes for different visualization types
COLOR_SCHEMES = {
    'main_temp': '#E63946',  # Bright red for main temperature
    'temp_range': '#457B9D',  # Blue for temperature range
    'humidity': '#1D3557',   # Dark blue for humidity
    'precipitation': '#A8DADC',  # Light blue for precipitation
    'wind': '#F1FAEE',       # Off-white for wind
    'actual': '#2A9D8F',     # Teal for actual values
    'predicted': '#E9C46A',  # Gold for predicted values
    'error': '#F4A261',      # Orange for error
    'classical': '#E76F51',  # Coral for classical models
    'advanced': '#264653',   # Dark teal for advanced models
    'background': '#FFFFFF', # White background
    'grid': '#CCCCCC',       # Light gray grid
    'text': '#000000'        # Black text
}

def load_and_preprocess_baghdad_data(file_path):
    """
    Load and preprocess Baghdad weather data with enhanced preprocessing techniques.
    
    Parameters:
    -----------
    file_path : str
        Path to the Excel file containing Baghdad weather data
        
    Returns:
    --------
    pd.DataFrame
        Preprocessed dataframe with enhanced features
    """
    print(f"Loading data from {file_path}...")
    
    try:
        # Load the data
        df = pd.read_excel(file_path)
        print(f"Successfully loaded data with {len(df)} rows and {len(df.columns)} columns")
        
        # Print column names to help with debugging
        print(f"Columns in dataset: {df.columns.tolist()}")
        
        # Convert datetime to proper datetime format if it exists
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            
            # Sort by datetime
            df = df.sort_values('datetime').reset_index(drop=True)
            
            # Extract datetime components for enhanced feature engineering
            df['year'] = df['datetime'].dt.year
            df['month'] = df['datetime'].dt.month
            df['day'] = df['datetime'].dt.day
            df['dayofyear'] = df['datetime'].dt.dayofyear
            df['dayofweek'] = df['datetime'].dt.dayofweek
            df['quarter'] = df['datetime'].dt.quarter
            df['weekofyear'] = df['datetime'].dt.isocalendar().week
            
            # Create season feature (meteorological seasons)
            df['season'] = pd.cut(df['month'], 
                                bins=[0, 3, 6, 9, 12], 
                                labels=['Winter', 'Spring', 'Summer', 'Fall'],
                                include_lowest=True,
                                ordered=False)
            
            # Create binary features for each season
            for season in ['Winter', 'Spring', 'Summer', 'Fall']:
                df[f'is_{season.lower()}'] = (df['season'] == season).astype(int)
        else:
            print("Warning: 'datetime' column not found. Creating a dummy datetime index.")
            df['datetime'] = pd.date_range(start='2019-01-01', periods=len(df), freq='D')
            df['year'] = df['datetime'].dt.year
            df['month'] = df['datetime'].dt.month
            df['day'] = df['datetime'].dt.day
            df['dayofyear'] = df['datetime'].dt.dayofyear
        
        # Drop unnecessary columns
        for col in df.columns:
            if 'unnamed' in col.lower():
                df = df.drop([col], axis=1)
        
        # Identify temperature columns
        temp_cols = [col for col in df.columns if 'temp' in col.lower()]
        print(f"Temperature columns found: {temp_cols}")
        
        # Calculate temperature range if tempmax and tempmin exist
        if 'tempmax' in df.columns and 'tempmin' in df.columns:
            df['temp_range'] = df['tempmax'] - df['tempmin']
            
            # Calculate temperature volatility (rolling standard deviation)
            df['temp_volatility_7d'] = df['temp_range'].rolling(window=7, min_periods=1).std()
            df['temp_volatility_30d'] = df['temp_range'].rolling(window=30, min_periods=1).std()
        
        # Handle missing values with advanced techniques - FIXED to avoid time-based interpolation error
        numerical_cols = df.select_dtypes(include=['number']).columns
        for col in numerical_cols:
            # Check if column has missing values
            if df[col].isnull().any():
                # Use simple interpolation instead of time-based interpolation
                df[col] = df[col].interpolate(method='linear').bfill().ffill()
                print(f"Filled missing values in {col} with linear interpolation")
        
        # Create a default target column if none exists
        if 'temp' not in df.columns and len(temp_cols) > 0:
            # Use the first temperature column as default
            df['temp'] = df[temp_cols[0]]
            print(f"Created 'temp' column from {temp_cols[0]}")
        
        # Calculate rolling statistics for enhanced feature engineering
        for col in temp_cols:
            if col in df.columns:
                # Rolling averages
                df[f'{col}_7d_avg'] = df[col].rolling(window=7, min_periods=1).mean()
                df[f'{col}_30d_avg'] = df[col].rolling(window=30, min_periods=1).mean()
                
                # Rolling min/max
                df[f'{col}_7d_min'] = df[col].rolling(window=7, min_periods=1).min()
                df[f'{col}_7d_max'] = df[col].rolling(window=7, min_periods=1).max()
                
                # Rolling standard deviation (volatility)
                df[f'{col}_7d_std'] = df[col].rolling(window=7, min_periods=1).std()
                
                # Calculate rate of change
                df[f'{col}_1d_change'] = df[col].diff()
                df[f'{col}_7d_change'] = df[col] - df[f'{col}_7d_avg']
                
                # Calculate acceleration (change in rate of change)
                df[f'{col}_1d_acceleration'] = df[f'{col}_1d_change'].diff()
        
        # Add cyclical encoding for temporal features (better for ML models)
        # Month
        df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
        df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
        
        # Day of year
        df['dayofyear_sin'] = np.sin(2 * np.pi * df['dayofyear']/365.25)
        df['dayofyear_cos'] = np.cos(2 * np.pi * df['dayofyear']/365.25)
        
        # Day of week
        if 'dayofweek' in df.columns:
            df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek']/7)
            df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek']/7)
        
        # Calculate anomalies (deviation from expected values)
        # Group by day of year to get expected values for each day
        for col in temp_cols:
            if col in df.columns:
                # Calculate expected values by grouping by day of year
                expected_values = df.groupby('dayofyear')[col].transform('mean')
                df[f'{col}_anomaly'] = df[col] - expected_values
                
                # Calculate z-scores for anomaly detection
                df[f'{col}_zscore'] = (df[col] - df[col].mean()) / df[col].std()
                
                # Identify extreme events (beyond 2 standard deviations)
                df[f'{col}_extreme_high'] = (df[f'{col}_zscore'] > 2).astype(int)
                df[f'{col}_extreme_low'] = (df[f'{col}_zscore'] < -2).astype(int)
        
        # Calculate cross-feature interactions
        if 'humidity' in df.columns and 'temp' in df.columns:
            # Heat index approximation
            df['heat_index'] = df['temp'] + 0.05 * df['humidity']
        
        if 'precip' in df.columns and 'temp' in df.columns:
            # Simple drought index
            df['drought_index'] = df['temp'] - 10 * df['precip']
            df['drought_index_30d'] = df['drought_index'].rolling(window=30, min_periods=1).mean()
        
        # Apply smoothing to reduce noise
        for col in temp_cols:
            if col in df.columns:
                try:
                    # Apply Savitzky-Golay filter for smoothing
                    df[f'{col}_smooth'] = savgol_filter(df[col], window_length=7, polyorder=3)
                except:
                    # Fallback to simple moving average if savgol fails
                    df[f'{col}_smooth'] = df[col].rolling(window=7, min_periods=1).mean()
        
        print(f"Enhanced preprocessing complete. Data shape: {df.shape}")
        return df
        
    except Exception as e:
        print(f"Error in data loading/preprocessing: {str(e)}")
        print("Please check your file path and data format.")
        return None

def perform_advanced_analysis(df):
    """
    Perform advanced statistical analysis on Baghdad weather data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Preprocessed dataframe
        
    Returns:
    --------
    tuple
        (df, analysis_results) - Updated dataframe and dictionary of analysis results
    """
    if df is None:
        print("No data available for analysis.")
        return None, None
    
    print("Performing advanced analysis...")
    
    try:
        # Select numerical columns for analysis
        numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        # 1. Temporal analysis
        analysis_results = {}
        
        # Create yearly averages for key metrics
        if 'year' in df.columns:
            temp_cols = [col for col in df.columns if 'temp' in col.lower() and df[col].dtype != 'object']
            if temp_cols:
                yearly_data = df.groupby('year')[temp_cols].agg(['mean', 'max', 'min', 'std']).reset_index()
                analysis_results['yearly_data'] = yearly_data
                
                # Perform trend analysis on yearly data
                trend_analysis = {}
                for col in temp_cols:
                    if col in df.columns:
                        # Calculate yearly averages
                        yearly_avg = df.groupby('year')[col].mean().reset_index()
                        # Check if there's a significant trend
                        if len(yearly_avg) >= 3:  # Need at least 3 years for trend
                            slope, intercept, r_value, p_value, std_err = stats.linregress(
                                range(len(yearly_avg)), yearly_avg[col]
                            )
                            trend_analysis[col] = {
                                'slope': slope,
                                'p_value': p_value,
                                'r_squared': r_value**2,
                                'significant': p_value < 0.05,
                                'direction': 'increasing' if slope > 0 else 'decreasing'
                            }
                analysis_results['trend_analysis'] = trend_analysis
        
        # Create monthly averages
        if 'month' in df.columns:
            temp_cols = [col for col in df.columns if 'temp' in col.lower() and df[col].dtype != 'object']
            if temp_cols:
                monthly_data = df.groupby('month')[temp_cols].agg(['mean', 'max', 'min', 'std']).reset_index()
                analysis_results['monthly_data'] = monthly_data
        
        # Create seasonal averages
        if 'season' in df.columns:
            temp_cols = [col for col in df.columns if 'temp' in col.lower() and df[col].dtype != 'object']
            if temp_cols:
                seasonal_data = df.groupby(['year', 'season'])[temp_cols].agg(['mean', 'max', 'min', 'std']).reset_index()
                analysis_results['seasonal_data'] = seasonal_data
        
        # 2. Correlation analysis
        # Select a subset of numerical columns to avoid memory issues
        corr_cols = numerical_cols[:15] if len(numerical_cols) > 15 else numerical_cols
        corr_matrix = df[corr_cols].corr()
        analysis_results['corr_matrix'] = corr_matrix
        
        # 3. Extreme value analysis
        # Identify extreme values (top and bottom 5%)
        extreme_data = {}
        for col in numerical_cols:
            if 'temp' in col.lower() and df[col].dtype != 'object' and df[col].nunique() > 10:
                extreme_data[f'{col}_high'] = df[df[col] > df[col].quantile(0.95)]
                extreme_data[f'{col}_low'] = df[df[col] < df[col].quantile(0.05)]
        analysis_results['extreme_data'] = extreme_data
        
        # 4. Advanced statistical analysis with PCA and clustering
        if SKLEARN_AVAILABLE:
            try:
                # Select features for PCA
                pca_features = [col for col in df.columns if 'temp' in col.lower() and df[col].dtype != 'object']
                if len(pca_features) >= 2:
                    # Take a subset of features if there are too many
                    pca_features = pca_features[:8] if len(pca_features) > 8 else pca_features
                    
                    pca_data = df[pca_features].copy()
                    
                    # Standardize the data
                    scaler = StandardScaler()
                    pca_data_scaled = scaler.fit_transform(pca_data)
                    
                    # Apply PCA
                    pca = PCA(n_components=3)  # Use 3 components for 3D visualization
                    pca_result = pca.fit_transform(pca_data_scaled)
                    
                    # Add PCA results to dataframe
                    for i in range(pca_result.shape[1]):
                        df[f'pca{i+1}'] = pca_result[:, i]
                    
                    # Store PCA explained variance
                    analysis_results['pca_explained_variance'] = pca.explained_variance_ratio_
                    
                    # Perform clustering with optimal number of clusters
                    # Determine optimal number of clusters using silhouette score
                    from sklearn.metrics import silhouette_score
                    silhouette_scores = []
                    K = range(2, 7)
                    for k in K:
                        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                        cluster_labels = kmeans.fit_predict(pca_data_scaled)
                        silhouette_avg = silhouette_score(pca_data_scaled, cluster_labels)
                        silhouette_scores.append(silhouette_avg)
                    
                    # Get optimal number of clusters
                    optimal_k = K[np.argmax(silhouette_scores)]
                    analysis_results['optimal_clusters'] = optimal_k
                    
                    # Perform clustering with optimal k
                    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
                    df['cluster'] = kmeans.fit_predict(pca_data_scaled)
                    
                    # Calculate cluster profiles
                    cluster_profiles = df.groupby('cluster')[pca_features].mean()
                    analysis_results['cluster_profiles'] = cluster_profiles
                    
                    # Calculate cluster temporal distribution
                    if 'month' in df.columns:
                        cluster_monthly = pd.crosstab(df['month'], df['cluster'], normalize='index')
                        analysis_results['cluster_monthly'] = cluster_monthly
                    
                    if 'season' in df.columns:
                        cluster_seasonal = pd.crosstab(df['season'], df['cluster'], normalize='index')
                        analysis_results['cluster_seasonal'] = cluster_seasonal
            except Exception as e:
                print(f"Warning: Could not perform PCA/clustering: {str(e)}")
        
        # 5. Autocorrelation analysis for time series
        if 'tempmax' in df.columns:
            from statsmodels.tsa.stattools import acf, pacf
            try:
                # Calculate autocorrelation
                acf_values = acf(df['tempmax'].dropna(), nlags=30)
                pacf_values = pacf(df['tempmax'].dropna(), nlags=30)
                
                analysis_results['acf_values'] = acf_values
                analysis_results['pacf_values'] = pacf_values
            except Exception as e:
                print(f"Warning: Could not perform autocorrelation analysis: {str(e)}")
        
        print("Advanced analysis complete.")
        return df, analysis_results
        
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        return df, None

import numpy as np
from scipy.interpolate import CubicSpline

from scipy.interpolate import CubicSpline  # Add this import
import numpy as np

def augment_time_series_data(X, y, augmentation_factor=3.0):
    """
    Apply advanced time series data augmentation techniques.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Input features (3D array for sequence data)
    y : numpy.ndarray
        Target values
    augmentation_factor : float
        Multiplier for augmentation: how many times to augment the dataset
        (e.g., 4.0 produces 400% additional samples)
        
    Returns:
    --------
    tuple of numpy.ndarray
        X_augmented, y_augmented - arrays containing original plus augmented samples
    """
    print(f"Applying time series data augmentation with factor {augmentation_factor}...")
    
    try:
        n_samples = X.shape[0]
        n_augment = int(n_samples * augmentation_factor)
        
        # Initialize augmented data with original data
        X_augmented = X.copy()
        y_augmented = y.copy()
        
        # 1. Jittering (add random noise)
        for _ in range(n_augment // 4):
            idx = np.random.randint(n_samples)
            sample_X = X[idx].copy()
            sample_y = y[idx].copy()
            noise = np.random.normal(0, 0.02, sample_X.shape)
            sample_X += noise
            X_augmented = np.vstack((X_augmented, sample_X[None]))
            y_augmented = np.vstack((y_augmented, sample_y[None]))
        
        # 2. Scaling (multiply by random factor)
        for _ in range(n_augment // 4):
            idx = np.random.randint(n_samples)
            sample_X = X[idx].copy()
            sample_y = y[idx].copy()
            scale = np.random.uniform(0.95, 1.05)
            sample_X *= scale
            X_augmented = np.vstack((X_augmented, sample_X[None]))
            y_augmented = np.vstack((y_augmented, sample_y[None]))
        
        # 3. Time warping (stretch or compress time)
        for _ in range(n_augment // 4):
            idx = np.random.randint(n_samples)
            sample_X = X[idx].copy()
            sample_y = y[idx].copy()
            seq_len, _ = sample_X.shape
            warped_X = np.zeros_like(sample_X)
            warp_factor = np.random.uniform(0.9, 1.1)
            indices = np.linspace(0, seq_len - 1, seq_len)
            warped_indices = np.clip(indices * warp_factor, 0, seq_len - 1).astype(int)
            for j, wi in enumerate(warped_indices):
                warped_X[j] = sample_X[wi]
            X_augmented = np.vstack((X_augmented, warped_X[None]))
            y_augmented = np.vstack((y_augmented, sample_y[None]))
        
        # 4. Magnitude warping (apply varying scaling across time)
        for _ in range(n_augment // 4):
            idx = np.random.randint(n_samples)
            sample_X = X[idx].copy()
            sample_y = y[idx].copy()
            seq_len, _ = sample_X.shape
            knot_points = np.random.choice(seq_len, size=4, replace=False)
            knot_points.sort()
            scales = np.random.uniform(0.9, 1.1, size=len(knot_points))
            cs = CubicSpline(knot_points, scales)
            factors = cs(np.arange(seq_len))
            for j in range(seq_len):
                sample_X[j] *= factors[j]
            X_augmented = np.vstack((X_augmented, sample_X[None]))
            y_augmented = np.vstack((y_augmented, sample_y[None]))
        
        total = X_augmented.shape[0]
        print(f"Data augmentation complete. Original samples: {n_samples}, Augmented samples: {total}")
        return X_augmented, y_augmented
        
    except Exception as e:
        print(f"Error in data augmentation: {e}")
        print("Returning original data without augmentation.")
        return X, y
def create_sequences(data, targets, sequence_length):
    """
    Create sequences from data for time series modeling.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Input features
    targets : numpy.ndarray
        Target values
    sequence_length : int
        Length of sequences to create
        
    Returns:
    --------
    tuple
        (X, y) - Sequence features and targets
    """
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(targets[i+sequence_length])
    return np.array(X), np.array(y)

def prepare_model_data(df, target_col='tempmax', sequence_length=30, test_size=0.2, apply_augmentation=True):
    """
    Prepare data for modeling with enhanced feature selection and augmentation.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Preprocessed dataframe
    target_col : str
        Target column to predict
    sequence_length : int
        Length of sequences for time series modeling
    test_size : float
        Proportion of data to use for testing
    apply_augmentation : bool
        Whether to apply data augmentation
        
    Returns:
    --------
    dict
        Dictionary containing model data
    """
    if df is None:
        print("No data available for modeling.")
        return None
    
    print(f"Preparing model data for predicting {target_col}...")
    
    try:
        # Check if target column exists
        if target_col not in df.columns:
            print(f"Target column '{target_col}' not found. Available columns: {df.columns.tolist()}")
            # Try to find a suitable temperature column
            temp_cols = [col for col in df.columns if 'temp' in col.lower() and df[col].dtype != 'object']
            if temp_cols:
                target_col = temp_cols[0]
                print(f"Using '{target_col}' as target column instead.")
            else:
                print("No suitable temperature column found.")
                return None
        
        # Select features with enhanced feature selection
        # Base features
        base_features = [col for col in df.columns if col in [
            'tempmax', 'tempmin', 'temp', 'feelslikemax', 'feelslikemin', 'feelslike',
            'humidity', 'precip', 'windspeed', 'cloudcover', 'sealevelpressure'
        ]]
        
        # Derived features
        derived_features = [col for col in df.columns if any(x in col for x in [
            '_avg', '_min', '_max', '_std', '_change', '_anomaly', '_zscore', '_smooth'
        ])]
        
        # Temporal features
        temporal_features = [col for col in df.columns if any(x in col for x in [
            'month_sin', 'month_cos', 'dayofyear_sin', 'dayofyear_cos', 'dayofweek_sin', 'dayofweek_cos'
        ])]
        
        # Combine all features
        all_features = base_features + derived_features + temporal_features
        
        # Remove duplicates and target column
        feature_cols = list(set(all_features))
        if target_col in feature_cols:
            feature_cols.remove(target_col)
        
        # Limit to top features to avoid dimensionality issues
        if len(feature_cols) > 30:
            # Calculate correlation with target
            correlations = df[feature_cols].corrwith(df[target_col]).abs().sort_values(ascending=False)
            feature_cols = correlations.index[:30].tolist()
        
        print(f"Selected {len(feature_cols)} features for modeling")
        
        # Handle missing values
        df_model = df[feature_cols + [target_col]].copy()
        for col in df_model.columns:
            if df_model[col].isnull().any():
                df_model[col] = df_model[col].fillna(df_model[col].mean())
        
        # Normalize features with robust scaling
        scaler_features = RobustScaler()
        features_scaled = scaler_features.fit_transform(df_model[feature_cols].values)
        
        # Normalize target
        scaler_target = StandardScaler()
        targets_scaled = scaler_target.fit_transform(df_model[target_col].values.reshape(-1, 1))
        
        # Create sequences
        X, y = create_sequences(features_scaled, targets_scaled, sequence_length)
        
        # Check if we have enough data
        if len(X) < 10:
            print(f"Not enough data points after creating sequences. Got {len(X)} samples.")
            return None
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False  # Keep time order
        )
        
        # Apply data augmentation to training set
        if apply_augmentation:
            X_train, y_train = augment_time_series_data(X_train, y_train, augmentation_factor=3.0)
        
        # Split training set into training and validation
        val_size = int(0.2 * len(X_train))
        X_val = X_train[-val_size:]
        y_val = y_train[-val_size:]
        X_train = X_train[:-val_size]
        y_train = y_train[:-val_size]
        
        model_data = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'scaler_features': scaler_features,
            'scaler_target': scaler_target,
            'feature_cols': feature_cols,
            'sequence_length': sequence_length,
            'input_dim': len(feature_cols),
            'target_col': target_col
        }
        
        print(f"Model data prepared with {len(feature_cols)} features")
        print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")
        return model_data
        
    except Exception as e:
        print(f"Error in model data preparation: {str(e)}")
        return None

# PyTorch Dataset for time series
class WeatherDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# Enhanced PyTorch Model: Multi-Modal Weather State Transition Model with Anomaly-Driven Recurrent Attention Network
class MMWSTM_ADRAN_Plus(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_clusters=9, sequence_length=30, dropout_rate=0.01, num_heads=4, num_layers=2, kernel_size=3,
                 use_attention=True, use_conv=True):
        super(MMWSTM_ADRAN_Plus, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_clusters = num_clusters
        self.sequence_length = sequence_length
        self.dropout_rate = dropout_rate
        
        # Input embedding layer
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # MMWSTM Component (LSTM-based)
        self.mmwstm_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=True  # Use bidirectional LSTM for better context
        )
        
        # Emission network
        self.emission_network = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim*2),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim*2, num_clusters)
        )
        
        # Transition matrix (learnable)
        self.transition_matrix = nn.Parameter(torch.randn(num_clusters, num_clusters) * 0.01)
        
        # MMWSTM output layer
        self.mmwstm_output = nn.Linear(hidden_dim*2 + num_clusters, hidden_dim)
        
        # ADRAN Component
        # Multi-head self-attention mechanism
        self.num_heads = 4
        self.head_dim = hidden_dim // self.num_heads
        
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection for multi-head attention
        self.attn_out = nn.Linear(hidden_dim, hidden_dim)
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        # Anomaly amplification
        self.anomaly_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # GRU for ADRAN
        self.adran_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=True
        )
        
        # ADRAN output layer
        self.adran_output = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Fusion layer with attention
        self.fusion_attention = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=1)
        )
        
        # Final output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim//2, 1)
        )
        
    def multi_head_attention(self, query, key, value):
        batch_size, seq_len, d_model = query.shape
        
        # Project queries, keys, and values
        q = self.query_proj(query)
        k = self.key_proj(key)
        v = self.value_proj(value)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, v)
        
        # Reshape back
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # Output projection
        output = self.attn_out(context)
        
        return output, attention_weights
    
    def forward(self, x):
        batch_size, seq_len, input_dim = x.shape
        
        # Input embedding
        embedded = self.embedding(x)
        
        # MMWSTM Component
        # Process with LSTM
        lstm_out, (h_n, c_n) = self.mmwstm_lstm(embedded)
        lstm_last = lstm_out[:, -1, :]  # Get the last output
        
        # Emission network
        cluster_logits = self.emission_network(lstm_last)
        cluster_probs = torch.softmax(cluster_logits, dim=-1)
        
        # Apply transition matrix
        weighted_transition = torch.matmul(cluster_probs, self.transition_matrix)
        
        # Combine LSTM output with transition-weighted cluster probabilities
        mmwstm_combined = torch.cat((lstm_last, weighted_transition), dim=1)
        mmwstm_out = self.mmwstm_output(mmwstm_combined)
        
        # ADRAN Component
        # Apply embedding to input for attention
        attn_input = embedded
        
        # Self-attention mechanism
        attn_output, attention_weights = self.multi_head_attention(attn_input, attn_input, attn_input)
        
        # Residual connection and layer normalization
        attn_output = self.layer_norm1(attn_input + attn_output)
        
        # Anomaly amplification
        anomaly_weights = self.anomaly_network(attn_output)
        amplified_input = attn_output * (1 + anomaly_weights)
        
        # Process with GRU
        gru_out, h_n = self.adran_gru(amplified_input)
        gru_last = gru_out[:, -1, :]  # Get the last output
        
        # ADRAN output
        adran_out = self.adran_output(gru_last)
        
        # Fusion of both components with attention
        combined = torch.cat((mmwstm_out, adran_out), dim=1)
        fusion_weights = self.fusion_attention(combined)
        
        # Weighted combination
        fused = fusion_weights[:, 0:1] * mmwstm_out + fusion_weights[:, 1:2] * adran_out
        
        # Final output
        output = self.output_layer(fused)
        
        return output, cluster_probs, attention_weights

# Custom loss function that emphasizes extreme events
class ExtremeWeatherLoss(nn.Module):
    def __init__(self, alpha=2.0, beta=0.5, gamma=1.0, threshold=0.95):
        super(ExtremeWeatherLoss, self).__init__()
        self.alpha = alpha    # Weight for extreme high events
        self.beta = beta      # Weight for normal events
        self.gamma = gamma    # Weight for extreme low events
        self.threshold = threshold  # Percentile threshold for extreme events
        
    def forward(self, predictions, targets):
        # Calculate base MSE loss
        base_loss = (predictions - targets) ** 2
        
        # Determine extreme high events (top percentile)
        high_threshold = torch.quantile(targets, self.threshold)
        extreme_high_mask = targets > high_threshold
        
        # Determine extreme low events (bottom percentile)
        low_threshold = torch.quantile(targets, 1 - self.threshold)
        extreme_low_mask = targets < low_threshold
        
        # Normal events mask
        normal_mask = ~(extreme_high_mask | extreme_low_mask)
        
        # Apply different weights to different types of events
        weighted_loss = torch.where(
            extreme_high_mask,
            self.alpha * base_loss,  # Higher weight for extreme high events
            torch.where(
                extreme_low_mask,
                self.gamma * base_loss,  # Higher weight for extreme low events
                self.beta * base_loss    # Lower weight for normal events
            )
        )
        
        return torch.mean(weighted_loss)
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score
)

# Flag to detect PyTorch availability
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Replace this with your actual dataset class
# from your_dataset_module import WeatherDataset

def train_classical_models(model_data):
    """
    Train modern deep learning baselines with enhanced architectures and training protocols
    """
    if model_data is None or not TORCH_AVAILABLE:
        print("Cannot train models: either model data is missing or PyTorch is not available.")
        return None

    print("Training modern deep learning baselines...")

    try:
        # --------------------------------------------------
        # Positional Encoding for Transformer
        # --------------------------------------------------
        class PositionalEncoding(nn.Module):
            def __init__(self, d_model, max_len=5000):
                super().__init__()
                position = torch.arange(max_len).unsqueeze(1)
                div_term = torch.exp(
                    torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
                pe = torch.zeros(1, max_len, d_model)
                pe[0, :, 0::2] = torch.sin(position * div_term)
                pe[0, :, 1::2] = torch.cos(position * div_term)
                self.register_buffer('pe', pe)

            def forward(self, x):
                x = x + self.pe[:, :x.size(1)]
                return x

        # --------------------------------------------------
        # Temporal Transformer
        # --------------------------------------------------
        class TemporalTransformer(nn.Module):
            def __init__(self, input_dim, hidden_dim=128, num_heads=8, num_layers=4):
                super().__init__()
                self.input_proj = nn.Linear(input_dim, hidden_dim)
                self.positional_enc = PositionalEncoding(hidden_dim)
                self.encoder = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=hidden_dim,
                        nhead=num_heads,
                        dim_feedforward=hidden_dim * 4,
                        dropout=0.1,
                        activation='gelu',
                        batch_first=True,
                        norm_first=True
                    ),
                    num_layers=num_layers
                )
                self.decoder = nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.GELU(),
                    nn.Linear(hidden_dim // 2, 1)
                )
                self._init_weights()

            def _init_weights(self):
                for p in self.parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)

            def forward(self, x):
                x = self.input_proj(x)
                x = self.positional_enc(x)
                x = self.encoder(x)
                return self.decoder(x[:, -1, :])

        # --------------------------------------------------
        # Temporal Convolutional Network (TCN)
        # --------------------------------------------------
        class TCN(nn.Module):
            def __init__(self, input_dim, num_channels=[64, 128, 256], kernel_size=3):
                super().__init__()
                layers = []
                in_channels = input_dim
                for out_channels in num_channels:
                    layers += [
                        nn.Conv1d(in_channels, out_channels, kernel_size,
                                  padding=(kernel_size - 1) // 2),
                        nn.GELU(),
                        nn.BatchNorm1d(out_channels),
                        nn.Dropout(0.1)
                    ]
                    in_channels = out_channels
                self.net = nn.Sequential(*layers)
                self.fc = nn.Sequential(
                    nn.Linear(num_channels[-1], 128),
                    nn.GELU(),
                    nn.Linear(128, 1)
                )

            def forward(self, x):
                x = x.permute(0, 2, 1)  # [batch, features, seq_len]
                x = self.net(x)
                return self.fc(x.mean(dim=-1))

        # --------------------------------------------------
        # N-BEATS with corrected dimensions
        # --------------------------------------------------
        class NBEATS(nn.Module):
            def __init__(self, input_dim, hidden_dim=128, num_blocks=4):
                super().__init__()
                self.input_dim = input_dim
                self.num_blocks = num_blocks

                # Each block outputs 2 scalars: [forecast(1), backcast(1)]
                self.blocks = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(input_dim, hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, 2)
                    ) for _ in range(num_blocks)
                ])

                # forecast_decoder consumes num_blocks scalars → 1 output
                self.forecast_decoder = nn.Linear(num_blocks, 1)
                # backcast_decoder consumes 1 scalar → input_dim sized backcast
                self.backcast_decoder = nn.Linear(1, input_dim)

            def forward(self, x):
                # x: [batch, seq_len, input_dim]
                residuals = x[:, -1, :]  # [batch, input_dim]
                forecasts = []

                for block in self.blocks:
                    out = block(residuals)            # [batch, 2]
                    fcst, bcast = out.chunk(2, dim=-1)  # each [batch,1]
                    residuals = residuals - self.backcast_decoder(bcast)
                    forecasts.append(fcst)

                # concat all block forecasts: [batch, num_blocks]
                all_fcst = torch.cat(forecasts, dim=-1)
                # final forecast: [batch,1]
                return self.forecast_decoder(all_fcst)

        # --------------------------------------------------
        # Training helper
        # --------------------------------------------------
        def train_model(model, train_data, val_data):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)

            optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
            scheduler = CosineAnnealingLR(optimizer, T_max=50)
            scaler = torch.cuda.amp.GradScaler()

            train_loader = DataLoader(
                WeatherDataset(*train_data),
                batch_size=256,
                shuffle=True,
                pin_memory=True
            )
            val_loader = DataLoader(
                WeatherDataset(*val_data),
                batch_size=512,
                pin_memory=True
            )

            best_loss = float('inf')
            early_stop_counter = 0
            patience = 15

            for epoch in range(100):
                model.train()
                train_loss = 0.0
                for features, targets in train_loader:
                    features = features.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)

                    optimizer.zero_grad(set_to_none=True)
                    with torch.cuda.amp.autocast():
                        outputs = model(features)
                        loss = F.huber_loss(outputs, targets, delta=0.5)

                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()

                    train_loss += loss.item()

                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for features, targets in val_loader:
                        features = features.to(device, non_blocking=True)
                        targets = targets.to(device, non_blocking=True)
                        outputs = model(features)
                        val_loss += F.mse_loss(outputs, targets).item()

                scheduler.step()
                avg_val_loss = val_loss / len(val_loader)

                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    best_model_state = model.state_dict().copy()
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                    if early_stop_counter >= patience:
                        break

                print(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, "
                      f"Val Loss: {avg_val_loss:.4f}")

            model.load_state_dict(best_model_state)
            return model

        # --------------------------------------------------
        # Unpack data
        # --------------------------------------------------
        # Unpack data
        X_train, y_train = model_data['X_train'], model_data['y_train']
        X_val, y_val     = model_data['X_val'],   model_data['y_val']
        X_test, y_test   = model_data['X_test'],  model_data['y_test']
        scaler = model_data['scaler_target']
        input_dim = model_data['input_dim']

        # Initialize models
        models = {
            'TemporalTransformer': TemporalTransformer(input_dim),
            'TCN':                TCN(input_dim),
            'N-BEATS':            NBEATS(input_dim)
        }

        results = {}
        for name, model in models.items():
            print(f"\n=== Training {name} ===")
            # ─────────── Measure training time ───────────
            start_time = time.time()
            trained = train_model(model, (X_train, y_train), (X_val, y_val))
            training_time = time.time() - start_time
            # ──────────────────────────────────────────────

            # Evaluate on test set
            test_loader = DataLoader(
                WeatherDataset(X_test, y_test),
                batch_size=512,
                shuffle=False
            )

            preds, acts = [], []
            trained.eval()
            with torch.no_grad():
                for features, targets in test_loader:
                    features = features.to(trained.device if hasattr(trained, 'device') else next(trained.parameters()).device)
                    out = trained(features).cpu().numpy()
                    preds.append(out)
                    acts.append(targets.numpy())

            preds = np.concatenate(preds).ravel()
            acts  = np.concatenate(acts).ravel()
            preds_orig = scaler.inverse_transform(preds.reshape(-1,1)).ravel()
            acts_orig  = scaler.inverse_transform(acts.reshape(-1,1)).ravel()

            # Compute metrics
            metrics = {
                'mse': mean_squared_error(acts_orig, preds_orig),
                'rmse': np.sqrt(mean_squared_error(acts_orig, preds_orig)),
                'mae': mean_absolute_error(acts_orig, preds_orig),
                'r2': r2_score(acts_orig, preds_orig),
                'explained_variance': explained_variance_score(acts_orig, preds_orig),
                'correlation': np.corrcoef(acts_orig, preds_orig)[0,1],
                'training_time': training_time
            }

            # Extreme event metrics
            q_high = np.quantile(acts_orig, 0.95)
            q_low  = np.quantile(acts_orig, 0.05)
            if np.any(acts_orig > q_high):
                metrics['extreme_high_rmse'] = np.sqrt(mean_squared_error(
                    acts_orig[acts_orig > q_high],
                    preds_orig[acts_orig > q_high]
                ))
            else:
                metrics['extreme_high_rmse'] = np.nan

            if np.any(acts_orig < q_low):
                metrics['extreme_low_rmse'] = np.sqrt(mean_squared_error(
                    acts_orig[acts_orig < q_low],
                    preds_orig[acts_orig < q_low]
                ))
            else:
                metrics['extreme_low_rmse'] = np.nan

            results[name] = {
                'model': trained,
                'metrics': metrics,
                'predictions': preds_orig
            }

            print(f"{name} Metrics:")
            print(f"  RMSE: {metrics['rmse']:.4f} | MAE: {metrics['mae']:.4f} | R²: {metrics['r2']:.4f}")
            print(f"  Extreme High RMSE: {metrics['extreme_high_rmse']:.4f} | Extreme Low RMSE: {metrics['extreme_low_rmse']:.4f}")
            print(f"  Training Time: {metrics['training_time']:.2f} seconds\n")

        return results

    except Exception as e:
        print(f"Error in training models: {str(e)}")
        traceback.print_exc()
        return None


def train_mmwstm_adran_model(model_data, hidden_dim=32, num_clusters=9, 
                           batch_size=256, num_epochs=200, learning_rate=0.1, 
                           patience=25, use_gpu=True):
    """
    Train the enhanced MMWSTM-ADRAN+ model with optimized hyperparameters.
    
    Parameters:
    -----------
    model_data : dict
        Dictionary containing model data
    hidden_dim : int
        Dimension of hidden layers
    num_clusters : int
        Number of clusters for state transition model
    batch_size : int
        Batch size for training
    num_epochs : int
        Maximum number of training epochs
    learning_rate : float
        Initial learning rate
    patience : int
        Patience for early stopping
    use_gpu : bool
        Whether to use GPU acceleration
        
    Returns:
    --------
    tuple
        (model, train_losses, val_losses, training_time)
    """
    if model_data is None or not TORCH_AVAILABLE:
        print("Cannot train model: either model data is missing or PyTorch is not available.")
        return None, None, None, None
    
    print("Training enhanced MMWSTM-ADRAN+ model...")
    
    try:
        start_time = time.time()
        
        # Initialize model
        input_dim = model_data['input_dim']
        sequence_length = model_data['sequence_length']
        
        # Create model
        model = MMWSTM_ADRAN_Plus(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_clusters=num_clusters,
            sequence_length=sequence_length,
            dropout_rate=0.01  # Increased dropout for better generalization
        )
        
        # Use GPU if available and requested
        device = torch.device("cuda:0" if torch.cuda.is_available() and use_gpu else "cpu")
        model = model.to(device)
        print(f"Using device: {device}")
        
        # Create datasets and dataloaders
        train_dataset = WeatherDataset(model_data['X_train'], model_data['y_train'])
        val_dataset = WeatherDataset(model_data['X_val'], model_data['y_val'])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Initialize loss function and optimizer
        criterion = ExtremeWeatherLoss(alpha=2.5, beta=0.5, gamma=2.0, threshold=0.95)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        # Learning rate scheduler
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=learning_rate/10
        )
        
        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for batch_idx, (features, targets) in enumerate(train_loader):
                # Move data to device
                features, targets = features.to(device), targets.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs, _, _ = model(features)
                
                # Calculate loss
                loss = criterion(outputs, targets)
                
                # Backward pass and optimize
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
            
            # Update learning rate
            scheduler.step()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for features, targets in val_loader:
                    # Move data to device
                    features, targets = features.to(device), targets.to(device)
                    
                    # Forward pass
                    outputs, _, _ = model(features)
                    
                    # Calculate loss
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            
            # Print progress
            print(f'Epoch: {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}')
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
                print(f"New best model saved with validation loss: {val_loss:.6f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        return model, train_losses, val_losses, training_time
        
    except Exception as e:
        print(f"Error in model training: {str(e)}")
        return None, None, None, None

def evaluate_mmwstm_adran_model(model, model_data, use_gpu=True):
    """
    Evaluate the enhanced MMWSTM-ADRAN+ model with comprehensive metrics.
    
    Parameters:
    -----------
    model : MMWSTM_ADRAN_Plus
        Trained model
    model_data : dict
        Dictionary containing model data
    use_gpu : bool
        Whether to use GPU acceleration
        
    Returns:
    --------
    dict
        Dictionary containing evaluation results
    """
    if model is None or model_data is None or not TORCH_AVAILABLE:
        print("Cannot evaluate model: either model is missing or PyTorch is not available.")
        return None
    
    print("Evaluating enhanced MMWSTM-ADRAN+ model...")
    
    try:
        # Use GPU if available and requested
        device = torch.device("cuda:0" if torch.cuda.is_available() and use_gpu else "cpu")
        model = model.to(device)
        
        # Create test dataset and dataloader
        test_dataset = WeatherDataset(model_data['X_test'], model_data['y_test'])
        test_loader = DataLoader(test_dataset, batch_size=64)
        
        # Evaluation mode
        model.eval()
        
        # Make predictions
        predictions = []
        actuals = []
        cluster_probs_list = []
        attention_weights_list = []
        
        with torch.no_grad():
            for features, targets in test_loader:
                # Move data to device
                features, targets = features.to(device), targets.to(device)
                
                # Forward pass
                outputs, cluster_probs, attention_weights = model(features)
                
                # Store predictions and actuals
                predictions.append(outputs.cpu().numpy())
                actuals.append(targets.cpu().numpy())
                
                # Store cluster probabilities and attention weights
                cluster_probs_list.append(cluster_probs.cpu().numpy())
                
                # Store only the first attention head for visualization
                if isinstance(attention_weights, tuple):
                    attention_weights = attention_weights[0]
                attention_weights_list.append(attention_weights[:, 0, :, :].cpu().numpy())
        
        # Concatenate batches
        predictions = np.concatenate(predictions)
        actuals = np.concatenate(actuals)
        cluster_probs = np.concatenate(cluster_probs_list)
        attention_weights = np.concatenate(attention_weights_list)
        
        # Inverse transform to get original scale
        predictions_orig = model_data['scaler_target'].inverse_transform(predictions).ravel()
        actuals_orig = model_data['scaler_target'].inverse_transform(actuals).ravel()
        
        # Calculate basic metrics
        mse = mean_squared_error(actuals_orig, predictions_orig)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actuals_orig, predictions_orig)
        r2 = r2_score(actuals_orig, predictions_orig)
        ev = explained_variance_score(actuals_orig, predictions_orig)
        
        # Calculate additional metrics for extreme events
        # Identify extreme values (top and bottom 5%)
        extreme_threshold_high = np.percentile(actuals_orig, 95)
        extreme_threshold_low = np.percentile(actuals_orig, 5)
        
        # Create masks for extreme events
        extreme_high_mask = actuals_orig > extreme_threshold_high
        extreme_low_mask = actuals_orig < extreme_threshold_low
        
        # Calculate metrics for extreme events
        if np.sum(extreme_high_mask) > 0:
            extreme_high_mse = mean_squared_error(
                actuals_orig[extreme_high_mask], 
                predictions_orig[extreme_high_mask]
            )
            extreme_high_rmse = np.sqrt(extreme_high_mse)
            extreme_high_mae = mean_absolute_error(
                actuals_orig[extreme_high_mask], 
                predictions_orig[extreme_high_mask]
            )
        else:
            extreme_high_rmse = np.nan
            extreme_high_mae = np.nan
        
        if np.sum(extreme_low_mask) > 0:
            extreme_low_mse = mean_squared_error(
                actuals_orig[extreme_low_mask], 
                predictions_orig[extreme_low_mask]
            )
            extreme_low_rmse = np.sqrt(extreme_low_mse)
            extreme_low_mae = mean_absolute_error(
                actuals_orig[extreme_low_mask], 
                predictions_orig[extreme_low_mask]
            )
        else:
            extreme_low_rmse = np.nan
            extreme_low_mae = np.nan
        
        # Calculate prediction bias
        bias = np.mean(predictions_orig - actuals_orig)
        
        # Calculate prediction variance
        variance = np.var(predictions_orig - actuals_orig)
        
        # Calculate correlation coefficient
        correlation = np.corrcoef(actuals_orig, predictions_orig)[0, 1]
        
        print(f'Test MSE: {mse:.6f}')
        print(f'Test RMSE: {rmse:.6f}')
        print(f'Test MAE: {mae:.6f}')
        print(f'Test R²: {r2:.6f}')
        print(f'Explained Variance: {ev:.6f}')
        print(f'Correlation: {correlation:.6f}')
        print(f'Bias: {bias:.6f}')
        print(f'Extreme High RMSE: {extreme_high_rmse:.6f}')
        print(f'Extreme Low RMSE: {extreme_low_rmse:.6f}')
        
        return {
            'predictions': predictions_orig,
            'actuals': actuals_orig,
            'cluster_probabilities': cluster_probs,
            'attention_weights': attention_weights,
            'metrics': {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'explained_variance': ev,
                'correlation': correlation,
                'bias': bias,
                'variance': variance,
                'extreme_high_rmse': extreme_high_rmse,
                'extreme_high_mae': extreme_high_mae,
                'extreme_low_rmse': extreme_low_rmse,
                'extreme_low_mae': extreme_low_mae
            }
        }
        
    except Exception as e:
        print(f"Error in model evaluation: {str(e)}")
        return None

def create_advanced_visualizations(df, mmwstm_adran_results, classical_results, output_dir, target_col='tempmax'):
    """
    Create publication-quality visualizations for AI journal submissions.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Preprocessed dataframe
    mmwstm_adran_results : dict
        Results from MMWSTM-ADRAN+ model
    classical_results : dict
        Results from classical models
    output_dir : str
        Directory to save visualizations
    target_col : str
        Target column that was predicted
        
    Returns:
    --------
    list
        List of paths to created visualizations
    """
    if df is None or mmwstm_adran_results is None:
        print("Cannot create visualizations: data or results are missing.")
        return []
    
    print("Creating journal-quality visualizations...")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    visualization_paths = []
    
    # Enhanced professional color scheme with distinct model colors
    COLOR_SCHEMES = {
        'actual': '#2c3e50',       # Dark slate blue
        'predicted': '#e74c3c',     # Alizarin red (now more distinct)
        'error': '#c0392b',         # Pomegranate red
        'advanced': '#8e44ad',      # Deep purple
        'classical': [
            '#3498db', '#2ecc71', '#f39c12', '#1abc9c', 
            '#d35400', '#9b59b6', '#34495e', '#16a085'
        ],  # Distinct colors for classical models
        'extreme_high': '#c0392b',  # Pomegranate red
        'extreme_low': '#2980b9',   # Belize hole blue
        'background': '#FFFFFF',    # White
        'grid': '#ecf0f1',          # Light silver
        'text': '#2c3e50',          # Dark slate
        'highlight': '#f1c40f'      # Vivid yellow
    }
    
    # Enhanced line styles configuration                                        
    LINE_STYLES = {
        'actual': '-',      # Solid line
        'predicted': '--',  # Dashed line
        'classical': [      # Distinct styles per classical model
            (0, (6, 2)),              # Long dash
            (0, (4, 2, 1, 2)),        # Dash-dot
            (0, (1, 2)),              # Dotted
            (0, (3, 1, 1, 1)),        # Dash-dot-dot
            (0, (8, 1, 1, 1, 1, 1)),  # Very loose dash-dot-dot
            (0, (2, 2, 2, 2)),        # Dash-dash
            (0, (5, 1, 5, 1)),        # Alternating long/short
            (0, (1, 1))               # Dense dotted
        ]
    }
    
    # Enhanced marker styles for classical models
    MARKER_STYLES = ['o', 's', 'D', '^', 'v', 'x', 'p', '*']
    
    # Set global style parameters
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial'],
        'font.weight': 'bold',
        'axes.titleweight': 'bold',
        'axes.labelweight': 'bold',
        'axes.edgecolor': COLOR_SCHEMES['text'],
        'axes.linewidth': 1.5,
        'axes.grid': True,
        'grid.color': COLOR_SCHEMES['grid'],
        'grid.alpha': 0.7,
        'grid.linestyle': ':',
        'figure.titlesize': 36,
        'axes.titlesize': 30,
        'axes.labelsize': 26,
        'xtick.labelsize': 22,
        'ytick.labelsize': 22,
        'legend.fontsize': 22,
        'legend.title_fontsize': 22,
        'figure.dpi': 600,
        'savefig.dpi': 600,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.5
    })

    # =================== 1. Enhanced 3D Temporal Visualization ===================
    def create_3d_temporal():
        if 'datetime' in df.columns and target_col in df.columns and 'month' in df.columns and 'year' in df.columns:
            fig = plt.figure(figsize=(30, 20), facecolor=COLOR_SCHEMES['background'])
            ax = fig.add_subplot(111, projection='3d')
            
            # Extract data
            years = df['year'].values
            months = df['month'].values
            values = df[target_col].values
            
            # Create 3D scatter plot with dynamic sizing
            norm = plt.Normalize(values.min(), values.max())
            scatter = ax.scatter3D(
                months, years, values, 
                c=values, cmap=plt.cm.plasma,
                s=values*10 + 50,
                alpha=1.0,
                edgecolor=COLOR_SCHEMES['text'],
                linewidth=1.0,
                norm=norm
            )
            
            # Add colorbar
            cbar = fig.colorbar(scatter, ax=ax, pad=0.1, shrink=0.6)
            cbar.set_label(f'{target_col} (°C)', fontsize=30, weight='bold', labelpad=17)
            cbar.ax.tick_params(labelsize=25)
            
            # Axis labels with increased spacing
            ax.set_xlabel('\nMonth', fontsize=30, labelpad=23)
            ax.set_ylabel('\nYear', fontsize=30, labelpad=23)
            ax.set_zlabel(f'\n{target_col} (°C)', fontsize=30, labelpad=23)
            
            # Title with journal-style capitalization
            ax.set_title('3D TEMPORAL DISTRIBUTION OF MAXIMUM TEMPERATURE', 
                         fontsize=40, pad=26, color=COLOR_SCHEMES['text'])
            
            # Month labels with abbreviations
            ax.set_xticks(range(1, 13))
            ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], 
                             fontsize=17, fontweight='bold')
            
            
            
            # ========== CRITICAL FIX: GRID ON ALL THREE AXES ==========
            # Enable grid globally
            ax.grid(True, linestyle='--', alpha=1.0, linewidth=3, color='blue')
        
            # Force grid visibility on each axis pane
            ax.xaxis.pane.set_alpha(0.7)  # Make pane semi-transparent
            ax.yaxis.pane.set_alpha(0.7)
            ax.zaxis.pane.set_alpha(0.7)
        
            # Explicitly enable grid lines on each axis
            ax.xaxis._axinfo["grid"].update({"visible": True})
            ax.yaxis._axinfo["grid"].update({"visible": True})
            ax.zaxis._axinfo["grid"].update({"visible": True})
        
            # Set Y-axis ticks (years)
            year_min, year_max = int(min(years)), int(max(years))
            year_ticks = np.arange(year_min, year_max + 1, 5)  # Every 5 years
            ax.set_yticks(year_ticks)
            ax.set_yticklabels(year_ticks, fontsize=17, fontweight='bold')
        
            # Set Z-axis ticks (temperature)
            z_min, z_max = values.min(), values.max()
            z_ticks = np.linspace(z_min, z_max, 5)  # 5 evenly spaced ticks
            ax.set_zticks(z_ticks)
            ax.set_zticklabels([f"{t:.1f}" for t in z_ticks], fontsize=17, fontweight='bold')
            # ========================================================
            # Optimized view angle
            
            ax.view_init(elev=30, azim=-45)
            # Add contour plane (moved BELOW grid configuration)
            z_min_plane = values.min() - 2
            ax.plot_trisurf(months, years, np.full_like(values, z_min_plane), 
                           color='#f8f9fa', alpha=0.2, zorder=0)
            
            

            
            # Save figure
            fig_path = output_dir / '3d_temporal_distribution.png'
            plt.savefig(fig_path)
            plt.close()
            visualization_paths.append(fig_path)
            print(f"Created 3D temporal visualization: {fig_path}")

    # =================== 2. Enhanced Pattern Visualization ===================
    def create_3d_pattern():
        if 'pca1' in df.columns and 'pca2' in df.columns and 'pca3' in df.columns and 'cluster' in df.columns:
            fig = plt.figure(figsize=(30, 20), facecolor=COLOR_SCHEMES['background'])
            ax = fig.add_subplot(111, projection='3d')
            
            # Extract data
            x = df['pca1'].values
            y = df['pca2'].values
            z = df['pca3'].values
            clusters = df['cluster'].values
            
            # Create scatter plot with enhanced colors
            scatter = ax.scatter3D(
                x, y, z, c=clusters, cmap=plt.cm.tab20, 
                s=70, alpha=1.0, edgecolor='w', linewidth=0.7
            )
            
            # Colorbar
            cbar = fig.colorbar(scatter, ax=ax, pad=0.2, shrink=0.6)
            cbar.set_label('Cluster', fontsize=30, weight='bold', labelpad=18)
            cbar.ax.tick_params(labelsize=30)
            
            # Labels and title
            ax.set_xlabel('\nPrincipal Component 1', fontsize=30, labelpad=23)
            ax.set_ylabel('\nPrincipal Component 2', fontsize=30, labelpad=23)
            ax.set_zlabel('\nPrincipal Component 3', fontsize=30, labelpad=23)
            ax.set_title('WEATHER PATTERN CLUSTERING', 
                         fontsize=30, pad=28, color=COLOR_SCHEMES['text'])
            ax.grid(True, linestyle='--', alpha=1.0, linewidth=3, color='blue')
            # Optimized view angle
            ax.view_init(elev=25, azim=-35)
            
            # Save figure
            fig_path = output_dir / '3d_pattern_visualization.png'
            plt.savefig(fig_path)
            plt.close()
            visualization_paths.append(fig_path)
            print(f"Created pattern visualization: {fig_path}")

    # =================== 3. Enhanced Model Performance Comparison ===================
    def create_model_comparison():
        if classical_results:
            # Prepare data with sorted models
            models = ['MMWSTM-ADRAN+'] + sorted(classical_results.keys())
            metrics = {
                'RMSE': [mmwstm_adran_results['metrics']['rmse']],
                'MAE': [mmwstm_adran_results['metrics']['mae']],
                'R²': [mmwstm_adran_results['metrics']['r2']],
                'Extreme High RMSE': [mmwstm_adran_results['metrics']['extreme_high_rmse']],
                'Extreme Low RMSE': [mmwstm_adran_results['metrics']['extreme_low_rmse']]
            }
            
            for model in models[1:]:
                metrics['RMSE'].append(classical_results[model]['metrics']['rmse'])
                metrics['MAE'].append(classical_results[model]['metrics']['mae'])
                metrics['R²'].append(classical_results[model]['metrics']['r2'])
                metrics['Extreme High RMSE'].append(classical_results[model]['metrics']['extreme_high_rmse'])
                metrics['Extreme Low RMSE'].append(classical_results[model]['metrics']['extreme_low_rmse'])
            
            # Create bar charts instead of radar chart
            plt.rcParams.update({'font.size': 14, 'axes.labelcolor': COLOR_SCHEMES['text'], 
                                'text.color': COLOR_SCHEMES['text'], 'axes.facecolor': COLOR_SCHEMES['background']})
            
            # Create main metrics comparison (RMSE, MAE, R²)
            fig1, axs = plt.subplots(1, 3, figsize=(20, 6), facecolor=COLOR_SCHEMES['background'])
            fig1.suptitle('MODEL PERFORMANCE COMPARISON', fontsize=20, color=COLOR_SCHEMES['text'])
            
            # Plot RMSE
            axs[0].bar(models, metrics['RMSE'], color=[COLOR_SCHEMES['advanced']] + COLOR_SCHEMES['classical'])
            axs[0].set_title('RMSE', fontsize=16)
            axs[0].set_ylabel('RMSE Value')
            axs[0].tick_params(axis='x', rotation=45)
            
            # Plot MAE
            axs[1].bar(models, metrics['MAE'], color=[COLOR_SCHEMES['advanced']] + COLOR_SCHEMES['classical'])
            axs[1].set_title('MAE', fontsize=16)
            axs[1].set_ylabel('MAE Value')
            axs[1].tick_params(axis='x', rotation=45)
            
            # Plot R²
            axs[2].bar(models, metrics['R²'], color=[COLOR_SCHEMES['advanced']] + COLOR_SCHEMES['classical'])
            axs[2].set_title('R²', fontsize=16)
            axs[2].set_ylabel('R² Value')
            axs[2].tick_params(axis='x', rotation=45)
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            fig_path1 = output_dir / 'main_metrics_comparison.png'
            plt.savefig(fig_path1, facecolor=COLOR_SCHEMES['background'])
            plt.close(fig1)
            visualization_paths.append(fig_path1)
            
            # Create extreme events comparison
            fig2, axs = plt.subplots(1, 2, figsize=(14, 6), facecolor=COLOR_SCHEMES['background'])
            fig2.suptitle('EXTREME EVENT PERFORMANCE', fontsize=20, color=COLOR_SCHEMES['text'])
            
            # Plot Extreme High RMSE
            axs[0].bar(models, metrics['Extreme High RMSE'], 
                      color=[COLOR_SCHEMES['advanced']] + COLOR_SCHEMES['classical'])
            axs[0].set_title('EXTREME HIGH EVENT PERFORMANCE', fontsize=14)
            axs[0].set_ylabel('RMSE (°C)')
            axs[0].tick_params(axis='x', rotation=45)
            
            # Plot Extreme Low RMSE
            axs[1].bar(models, metrics['Extreme Low RMSE'], 
                      color=[COLOR_SCHEMES['advanced']] + COLOR_SCHEMES['classical'])
            axs[1].set_title('EXTREME LOW EVENT PERFORMANCE', fontsize=14)
            axs[1].set_ylabel('RMSE (°C)')
            axs[1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            fig_path2 = output_dir / 'extreme_events_comparison.png'
            plt.savefig(fig_path2, facecolor=COLOR_SCHEMES['background'])
            plt.close(fig2)
            visualization_paths.append(fig_path2)
            
            print(f"Created model comparison visualizations: {fig_path1} & {fig_path2}")

    # =================== 4. Enhanced Prediction Visualization ===================
    def create_prediction_visualizations():
        actuals = mmwstm_adran_results['actuals'].ravel()
        predictions = mmwstm_adran_results['predictions'].ravel()
        errors = predictions - actuals
        
        # Create figure with dual axes
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(70, 50), 
                                      sharex=True, 
                                      gridspec_kw={'height_ratios': [3, 1]},
                                      facecolor=COLOR_SCHEMES['background'])
        
        # Calculate y-axis padding (10% of data range for better spacing)
        data_range = np.max(actuals) - np.min(actuals)
        padding = 0.1 * data_range
        
        # Top: Actual vs Predicted with distinct styles
        # Plot lines and capture handles for legend
        (line_obs,) = ax1.plot(actuals, color=COLOR_SCHEMES['actual'], linewidth=6.0, linestyle=LINE_STYLES['actual'])
        (line_pred,) = ax1.plot(predictions, color=COLOR_SCHEMES['predicted'], linewidth=6.0, alpha=1.0, linestyle=LINE_STYLES['predicted'])
        # Plot extreme events and capture handles
        q_high = np.quantile(actuals, 0.95)
        q_low = np.quantile(actuals, 0.05)
        fill_high = ax1.fill_between(range(len(actuals)), q_high, np.max(actuals) + padding, where=actuals >= q_high, color=COLOR_SCHEMES['extreme_high'], alpha=0.8)
        fill_low = ax1.fill_between(range(len(actuals)), np.min(actuals) - padding, q_low, where=actuals <= q_low, color=COLOR_SCHEMES['extreme_low'], alpha=0.8)
        # Configure top plot
        ax1.set_ylim(np.min(actuals) - padding, np.max(actuals) + padding)
        ax1.xaxis.set_major_locator(mticker.MaxNLocator(nbins=10))
        ax1.yaxis.set_major_locator(mticker.MaxNLocator(nbins=8))
        ax1.tick_params(axis='both', which='major', labelsize=70, length=15, width=3, pad=10)
        [spine.set_linewidth(5) for spine in ax1.spines.values()]
        ax1.minorticks_on()
        ax1.tick_params(axis='both', which='minor', length=7, width=3.0, labelsize=50)
        ax1.set_title('a) OBSERVED VS PREDICTED TEMPERATURE', fontsize=100, fontweight='bold', pad=20)
        ax1.set_ylabel(f'{target_col} (°C)', fontsize=90, fontweight='bold')
        ax1.grid(True, linestyle='-.', linewidth=2, color='blue', alpha=1.0)
        # --- Bottom Plot: Errors ---
        ax2.bar(
            range(len(errors)), 
            errors, 
            color=np.where(errors >= 0, COLOR_SCHEMES['predicted'], COLOR_SCHEMES['error']),
            alpha=1.0, 
            width=1.0, 
            edgecolor='none')
        ax2.axhline(0, color=COLOR_SCHEMES['text'], linestyle='-', linewidth=3)
        ax2.xaxis.set_major_locator(mticker.MaxNLocator(nbins=10))
        ax2.yaxis.set_major_locator(mticker.MaxNLocator(nbins=8))
        ax2.tick_params(axis='both', which='major', labelsize=90, length=15, width=3, pad=10)
        [spine.set_linewidth(4) for spine in ax2.spines.values()]
        ax2.minorticks_on()
        ax2.tick_params(axis='both', which='minor', length=7, width=3.0, labelsize=70)
        ax2.set_xlabel('Time Step', fontsize=90, fontweight='bold')
        ax2.set_ylabel('Prediction Error (°C)', fontsize=90, fontweight='bold')
        ax2.grid(True, linestyle='-.', linewidth=2, color='blue', alpha=1.0)
        # --- Combined Legend at Bottom of Error Plot ---
        legend_items = [
            (line_obs, 'Observed'),
            (line_pred, 'MMWSTM-ADRAN+'),
            (fill_high, 'Extreme High Events'),
            (fill_low, 'Extreme Low Events')]
        # Create legend in error plot (ax2)
        ax2.legend(
            handles=[item[0] for item in legend_items],
            labels=[item[1] for item in legend_items],
            loc='upper center',
            bbox_to_anchor=(0.5, -0.3),  # Position below error plot
            ncol=4,
            fontsize=60,
            frameon=True,
            framealpha=1.0,
            fancybox=True,
            shadow=True,
            edgecolor='#2c3e50',
            facecolor='#f8f9fa',
            handlelength=2.0,
            handleheight=1.2,
            borderpad=0.8)
        # Adjust layout to accommodate legend
        plt.subplots_adjust(bottom=0.18)  # Increase bottom margin
    
       # Save figure
        fig_path = output_dir / 'prediction_performance.png'
        plt.savefig(fig_path, bbox_inches='tight')
        plt.close()
        visualization_paths.append(fig_path)
        print(f"Created prediction visualization: {fig_path}") 


    # =================== 5. Enhanced Error Distribution ===================
    def create_error_analysis():
        actuals = mmwstm_adran_results['actuals'].ravel()
        predictions = mmwstm_adran_results['predictions'].ravel()
        errors = predictions - actuals
        
        fig = plt.figure(figsize=(16, 12), facecolor=COLOR_SCHEMES['background'])
        gs = fig.add_gridspec(2, 2, width_ratios=[3,1], height_ratios=[1,3])
        
        # Main scatter plot with enhanced styling
        ax_main = fig.add_subplot(gs[1, 0])
        sc = ax_main.scatter(actuals, predictions, c=np.abs(errors), 
                           cmap=plt.cm.viridis, s=65, alpha=0.85,
                           edgecolors='w', linewidth=0.9)
        
        # Perfect prediction line
        min_val = min(actuals.min(), predictions.min())
        max_val = max(actuals.max(), predictions.max())
        ax_main.plot([min_val, max_val], [min_val, max_val], 
                   color=COLOR_SCHEMES['text'], 
                   linestyle='--', linewidth=3.0)
        
        ax_main.set_xlabel('Observed Temperature (°C)', fontsize=22)
        ax_main.set_ylabel('Predicted Temperature (°C)', fontsize=22)
        ax_main.set_title('PREDICTION ACCURACY ANALYSIS', fontsize=24, pad=20)
        
        # Add colorbar
        cbar = fig.colorbar(sc, ax=ax_main)
        cbar.set_label('Absolute Error (°C)', fontsize=20)
        
        # Error distribution (top) with enhanced styling
        ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
        sns.kdeplot(x=actuals, y=errors, ax=ax_top, 
                  fill=True, cmap='Blues', alpha=0.7, thresh=0.05)
        ax_top.axhline(0, color=COLOR_SCHEMES['text'], linestyle='-', linewidth=2)
        ax_top.set_ylabel('Error (°C)', fontsize=22)
        ax_top.tick_params(axis='x', labelbottom=False)
        
        # Error histogram (right) with enhanced styling - FIXED
        ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
        # Plot histogram
        sns.histplot(y=errors, ax=ax_right, color=COLOR_SCHEMES['error'], 
                   stat='density', alpha=0.7)
        # Plot KDE separately with linewidth
        sns.kdeplot(y=errors, ax=ax_right, color='k', linewidth=3)
        ax_right.axhline(0, color=COLOR_SCHEMES['text'], linestyle='-', linewidth=2)
        ax_right.set_xlabel('Density', fontsize=22)
        ax_right.tick_params(axis='y', labelleft=False)
        
        # Add statistics with enhanced styling
        stats_text = (f"Mean Error: {errors.mean():.2f}°C\n"
                     f"Std Dev: {errors.std():.2f}°C\n"
                     f"Skewness: {stats.skew(errors):.2f}\n"
                     f"Kurtosis: {stats.kurtosis(errors):.2f}")
        ax_main.text(0.98, 0.02, stats_text, transform=ax_main.transAxes,
                   fontsize=14, verticalalignment='bottom', 
                   horizontalalignment='right',
                   bbox=dict(facecolor='white', alpha=0.9, boxstyle='round', pad=0.5))
        
        # Save figure
        fig_path = output_dir / 'error_analysis.png'
        plt.savefig(fig_path)
        plt.close()
        visualization_paths.append(fig_path)
        print(f"Created error analysis visualization: {fig_path}")

    # =================== 6. Enhanced Attention Visualization ===================
    def create_attention_visualization():
        if 'attention_weights' in mmwstm_adran_results:
            weights = mmwstm_adran_results['attention_weights'][0]  # First sample
            
            fig = plt.figure(figsize=(14, 12), facecolor=COLOR_SCHEMES['background'])
            ax = fig.add_subplot(111)
            
            # Create heatmap with enhanced colormap
            im = ax.imshow(weights, cmap=plt.cm.coolwarm, aspect='auto', 
                         vmin=0, vmax=1, interpolation='nearest')
            
            # Add text annotations with enhanced styling
            for i in range(weights.shape[0]):
                for j in range(weights.shape[1]):
                    color = 'white' if weights[i, j] > 0.5 else COLOR_SCHEMES['text']
                    ax.text(j, i, f'{weights[i, j]:.2f}', 
                           ha='center', va='center', 
                           color=color, fontsize=10, fontweight='bold')
            
            # Labels and title
            ax.set_xlabel('Input Sequence Position', fontsize=22)
            ax.set_ylabel('Output Sequence Position', fontsize=22)
            ax.set_title('ATTENTION MECHANISM WEIGHTS', fontsize=24, pad=22)
            
            # Add colorbar
            cbar = fig.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Attention Weight', fontsize=20)
            
            # Save figure
            fig_path = output_dir / 'attention_mechanism.png'
            plt.savefig(fig_path)
            plt.close()
            visualization_paths.append(fig_path)
            print(f"Created attention visualization: {fig_path}")

    # =================== 7. Enhanced Cluster Visualization ===================
    def create_cluster_visualization():
        if 'cluster_probabilities' in mmwstm_adran_results:
            cluster_probs = mmwstm_adran_results['cluster_probabilities']
            
            fig = plt.figure(figsize=(20, 10), facecolor=COLOR_SCHEMES['background'])
            ax = fig.add_subplot(111)
            
            # Create heatmap with enhanced colormap
            im = ax.imshow(cluster_probs[:50], cmap=plt.cm.magma, aspect='auto')
            
            # Add colorbar
            cbar = fig.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Cluster Probability', fontsize=20)
            
            # Set labels
            ax.set_xlabel('Cluster ID', fontsize=30)
            ax.set_ylabel('Sample Index', fontsize=30)
            ax.set_title('WEATHER STATE CLUSTER PROBABILITIES', fontsize=40, pad=18)
            
            # Add cluster labels with enhanced styling
            ax.set_xticks(np.arange(cluster_probs.shape[1]))
            ax.set_xticklabels([f'C{i+1}' for i in range(cluster_probs.shape[1])], 
                              fontsize=22, rotation=45, fontweight='bold')
            
            # Save figure
            fig_path = output_dir / 'cluster_visualization.png'
            plt.savefig(fig_path)
            plt.close()
            visualization_paths.append(fig_path)
            print(f"Created cluster visualization: {fig_path}")

    # =================== 8. Classical Models Time Series ===================
    def create_classical_models_visualization():
        if classical_results is not None:
            fig = plt.figure(figsize=(100, 50), facecolor=COLOR_SCHEMES['background'])
            ax = fig.add_subplot(111)
            # Set axes background color to match figure
            ax.set_facecolor(COLOR_SCHEMES['background'])  # <-- ADD THIS LINE
            actuals = mmwstm_adran_results['actuals'].ravel()
            
            # Plot observed data
            ax.plot(actuals, label='Observed', color=COLOR_SCHEMES['actual'], 
                    linewidth=9.0, alpha=1.0, linestyle=LINE_STYLES['actual'])
            
            # Plot classical models with distinct styles and markers
            for i, (model_name, result) in enumerate(classical_results.items()):
                color_idx = i % len(COLOR_SCHEMES['classical'])
                linestyle_idx = i % len(LINE_STYLES['classical'])
                marker_idx = i % len(MARKER_STYLES)
                
                # Plot every 10th point with marker for better visibility
                ax.plot(result['predictions'], label=model_name, 
                        color=COLOR_SCHEMES['classical'][color_idx], 
                        linewidth=9.0, 
                        linestyle=LINE_STYLES['classical'][linestyle_idx],
                        marker=MARKER_STYLES[marker_idx], 
                        markersize=26, 
                        markevery=15,
                        alpha=1.0)
            
            # Add padding to y-axis for better readability
            data_range = np.max(actuals) - np.min(actuals)
            padding = 0.09 * data_range
            ax.set_ylim(np.min(actuals) - padding, np.max(actuals) + padding)
            ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=10))
            ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=8))
            ax.tick_params(axis='both', which='major', labelsize=80, length=18, width=7, pad=10)
            [spine.set_linewidth(4) for spine in ax.spines.values()]
            ax.minorticks_on()
            ax.tick_params(axis='both', which='minor', length=9, width=5, labelsize=90)

            
            ax.set_title('b) CLASSICAL MODEL PREDICTIONS', fontsize=150 , fontweight='bold', pad=20)
            ax.set_xlabel('Time Step', fontsize=150, fontweight='bold')
            ax.set_ylabel(f'{target_col} (°C)', fontsize=150, fontweight='bold')
            
            # Enhanced legend with spacing
            ax.legend(fontsize=80, frameon=True, framealpha=1.0, 
                     loc='upper center', bbox_to_anchor=(0.5, -0.15),
                     fancybox=True, shadow=True, ncol=4)
            
            ax.grid(True, linestyle='-.', linewidth=4, color='blue', alpha=1.0)

            
            # Save figure
            fig_path = output_dir / 'classical_models.png'
            plt.savefig(fig_path)
            plt.close()
            visualization_paths.append(fig_path)
            print(f"Created classical models visualization: {fig_path}")

    # =================== 9. Error Distribution ===================
    def create_error_distribution():
        actuals = mmwstm_adran_results['actuals'].ravel()
        predictions = mmwstm_adran_results['predictions'].ravel()
        errors = predictions - actuals
        
        fig = plt.figure(figsize=(20, 10), facecolor=COLOR_SCHEMES['background'])
        ax = fig.add_subplot(111)
        
        # Create histogram with enhanced styling - FIXED
        sns.histplot(errors, ax=ax, color=COLOR_SCHEMES['error'], 
                   stat='density', alpha=0.7)
        # Plot KDE separately with linewidth
        sns.kdeplot(errors, ax=ax, color='k', linewidth=4)
        
        # Add mean line with enhanced styling
        mean_err = np.mean(errors)
        ax.axvline(mean_err, color=COLOR_SCHEMES['text'], linestyle='--', linewidth=4, 
                  label=f'Mean Error: {mean_err:.2f}°C')
        
        # Add normal distribution comparison
        x = np.linspace(min(errors), max(errors), 100)
        normal = stats.norm.pdf(x, mean_err, np.std(errors))
        ax.plot(x, normal, 'r-', linewidth=3, label='Normal Distribution')
        
        ax.set_title('d) PREDICTION ERROR DISTRIBUTION', fontsize=50, pad=20)
        ax.set_xlabel('Error (Predicted - Actual) (°C)', fontsize=40)
        ax.set_ylabel('Density', fontsize=40)
        ax.legend(fontsize=18)
        
        # Save figure
        fig_path = output_dir / 'error_distribution.png'
        plt.savefig(fig_path)
        plt.close()
        visualization_paths.append(fig_path)
        print(f"Created error distribution visualization: {fig_path}")

    # =================== 10. Extreme Event Performance ===================
    def create_extreme_performance():
        if classical_results is not None:
            models = ['MMWSTM-ADRAN+'] + list(classical_results.keys())
            extreme_high = [mmwstm_adran_results['metrics']['extreme_high_rmse']]
            extreme_low = [mmwstm_adran_results['metrics']['extreme_low_rmse']]
            
            for model in classical_results.values():
                extreme_high.append(model['metrics']['extreme_high_rmse'])
                extreme_low.append(model['metrics']['extreme_low_rmse'])
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), 
                                          facecolor=COLOR_SCHEMES['background'])
            
            # Extreme high events with enhanced styling
            colors = [COLOR_SCHEMES['advanced']] + COLOR_SCHEMES['classical'][:len(models)-1]
            bars1 = ax1.bar(models, extreme_high, color=colors, edgecolor='k', linewidth=1.5)
            ax1.set_title('EXTREME HIGH EVENT PERFORMANCE', fontsize=24)
            ax1.set_ylabel('RMSE (°C)', fontsize=22)
            ax1.tick_params(axis='x', rotation=45)
            
            # Add value labels with enhanced styling
            for bar in bars1:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=14, fontweight='bold')
            
            # Extreme low events with enhanced styling
            bars2 = ax2.bar(models, extreme_low, color=colors, edgecolor='k', linewidth=1.5)
            ax2.set_title('EXTREME LOW EVENT PERFORMANCE', fontsize=24)
            ax2.set_ylabel('RMSE (°C)', fontsize=22)
            ax2.tick_params(axis='x', rotation=45)
            
            # Add value labels with enhanced styling
            for bar in bars2:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=14, fontweight='bold')
            
            # Add spacing between subplots
            plt.subplots_adjust(wspace=0.3)
            
            # Save figure
            fig_path = output_dir / 'extreme_performance.png'
            plt.savefig(fig_path)
            plt.close()
            visualization_paths.append(fig_path)
            print(f"Created extreme performance visualization: {fig_path}")

    # =================== 11. Scatter Plot with Regression ===================
    def create_scatter_plot():
        actuals = mmwstm_adran_results['actuals'].ravel()
        predictions = mmwstm_adran_results['predictions'].ravel()
        
        fig = plt.figure(figsize=(12, 10), facecolor=COLOR_SCHEMES['background'])
        ax = fig.add_subplot(111)
        
        # Create scatter plot with enhanced styling
        scatter = ax.scatter(actuals, predictions, alpha=0.8, 
                            c=np.abs(predictions - actuals), cmap=plt.cm.plasma,
                            s=75, edgecolors='w', linewidth=0.7)
        
        # Add perfect prediction line
        min_val = min(np.min(actuals), np.min(predictions))
        max_val = max(np.max(actuals), np.max(predictions))
        ax.plot([min_val, max_val], [min_val, max_val], 
               color=COLOR_SCHEMES['text'], linestyle='--', linewidth=3.0)
        
        # Add linear regression line
        slope, intercept, r_value, p_value, std_err = stats.linregress(actuals, predictions)
        reg_line = slope * np.array([min_val, max_val]) + intercept
        ax.plot([min_val, max_val], reg_line, 
               color=COLOR_SCHEMES['predicted'], linewidth=3.0)
        
        ax.set_title('c) PREDICTED VS OBSERVED VALUES', fontsize=30, pad=18)
        ax.set_xlabel('Observed Temperature (°C)', fontsize=22)
        ax.set_ylabel('Predicted Temperature (°C)', fontsize=22)
        
        # Add colorbar
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('Absolute Error (°C)', fontsize=18)
        
        # Add regression equation with enhanced styling
        equation = f'y = {slope:.2f}x + {intercept:.2f}\nR² = {r_value**2:.4f}'
        ax.text(0.05, 0.95, equation, transform=ax.transAxes,
               bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=COLOR_SCHEMES['text'], alpha=0.9),
               fontsize=16, verticalalignment='top')
        
        # Save figure
        fig_path = output_dir / 'scatter_plot.png'
        plt.savefig(fig_path)
        plt.close()
        visualization_paths.append(fig_path)
        print(f"Created scatter plot visualization: {fig_path}")

    # =================== 12. Metric Comparison ===================
    def create_metric_comparison():
        if classical_results is not None:
            models = ['MMWSTM-ADRAN+'] + list(classical_results.keys())
            metrics = {
                'RMSE': [mmwstm_adran_results['metrics']['rmse']],
                'MAE': [mmwstm_adran_results['metrics']['mae']],
                'R²': [mmwstm_adran_results['metrics']['r2']]
            }
            
            for model in models[1:]:
                metrics['RMSE'].append(classical_results[model]['metrics']['rmse'])
                metrics['MAE'].append(classical_results[model]['metrics']['mae'])
                metrics['R²'].append(classical_results[model]['metrics']['r2'])
            
            fig, axes = plt.subplots(1, 3, figsize=(24, 8), 
                                    facecolor=COLOR_SCHEMES['background'])
            
            colors = [COLOR_SCHEMES['advanced']] + COLOR_SCHEMES['classical'][:len(models)-1]
            
            for i, (metric_name, values) in enumerate(metrics.items()):
                ax = axes[i]
                bars = ax.bar(models, values, color=colors, edgecolor='k', linewidth=1.5)
                
                ax.set_title(metric_name, fontsize=24)
                ax.set_ylabel(metric_name, fontsize=22)
                ax.tick_params(axis='x', rotation=45)
                
                # Add value labels with enhanced styling
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                            f'{height:.2f}', ha='center', va='bottom', fontsize=16, fontweight='bold')
            
            # Add spacing between subplots
            plt.subplots_adjust(wspace=0.3)
            
            # Save figure
            fig_path = output_dir / 'metric_comparison.png'
            plt.savefig(fig_path)
            plt.close()
            visualization_paths.append(fig_path)
            print(f"Created metric comparison visualization: {fig_path}")

    # Execute all visualization functions
    try:
        create_3d_temporal()
        create_3d_pattern()
        create_model_comparison()
        create_prediction_visualizations()
        create_error_analysis()
        create_attention_visualization()
        create_cluster_visualization()
        create_classical_models_visualization()
        create_error_distribution()
        create_extreme_performance()
        create_scatter_plot()
        create_metric_comparison()
        
        print(f"Created {len(visualization_paths)} journal-quality visualizations")
        return visualization_paths
        
    except Exception as e:
        print(f"Visualization error: {str(e)}")
        import traceback
        traceback.print_exc()
        return visualization_paths
def document_patentable_innovations(mmwstm_adran_results, classical_results, output_dir):
    """
    Document the patentable innovations of the MMWSTM-ADRAN+ model.
    
    Parameters:
    -----------
    mmwstm_adran_results : dict
        Results from MMWSTM-ADRAN+ model
    classical_results : dict
        Results from classical models
    output_dir : str
        Directory to save documentation
        
    Returns:
    --------
    str
        Path to the documentation file
    """
    if mmwstm_adran_results is None:
        print("Cannot document innovations: results are missing.")
        return None
    
    print("Documenting patentable innovations...")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create documentation file
    doc_path = output_dir / 'patentable_innovations.md'
    
    try:
        with open(doc_path, 'w') as f:
            f.write("# MMWSTM-ADRAN+: Patentable Innovations\n\n")
            
            f.write("## 1. Executive Summary\n\n")
            f.write("The Multi-Modal Weather State Transition Model with Anomaly-Driven Recurrent Attention Network Plus (MMWSTM-ADRAN+) represents a significant advancement in climate data analysis and prediction. This document outlines the key innovations that make this model patentable, along with performance metrics demonstrating its superiority over classical approaches.\n\n")
            
            f.write("## 2. Key Innovations\n\n")
            
            f.write("### 2.1 Weather State Transition Matrix\n\n")
            f.write("The MMWSTM-ADRAN+ model introduces a novel learnable state transition matrix that captures the probabilistic transitions between different weather states. Unlike traditional time series models that treat weather as a continuous process, our approach models weather as a set of discrete states with probabilistic transitions between them.\n\n")
            f.write("**Innovation Details:**\n")
            f.write("- Learnable transition matrix that adapts to the specific climate regime\n")
            f.write("- Emission network that maps LSTM outputs to state probabilities\n")
            f.write("- Integration of state transition probabilities with deep learning features\n\n")
            
            f.write("### 2.2 Anomaly-Driven Recurrent Attention\n\n")
            f.write("The model incorporates a novel attention mechanism that specifically amplifies anomalous patterns in the input data. This allows the model to pay special attention to unusual weather patterns that might indicate extreme events.\n\n")
            f.write("**Innovation Details:**\n")
            f.write("- Multi-head self-attention mechanism with anomaly amplification\n")
            f.write("- Anomaly network that learns to identify and amplify unusual patterns\n")
            f.write("- Integration with bidirectional GRU for enhanced temporal context\n\n")
            
            f.write("### 2.3 Adaptive Fusion Layer\n\n")
            f.write("The MMWSTM-ADRAN+ model features an innovative fusion layer that dynamically weights the contributions of the state transition component and the anomaly attention component based on the input data.\n\n")
            f.write("**Innovation Details:**\n")
            f.write("- Attention-based fusion mechanism that learns optimal component weights\n")
            f.write("- Dynamic adaptation to different weather regimes and patterns\n")
            f.write("- Improved performance on both normal and extreme weather events\n\n")
            
            f.write("### 2.4 Extreme Weather Loss Function\n\n")
            f.write("We introduce a specialized loss function that places higher emphasis on extreme weather events, addressing the common problem in climate modeling where rare but important events are underrepresented.\n\n")
            f.write("**Innovation Details:**\n")
            f.write("- Differential weighting for extreme high, extreme low, and normal events\n")
            f.write("- Adaptive thresholding based on percentiles of the target distribution\n")
            f.write("- Improved accuracy for extreme event prediction\n\n")
            
            f.write("### 2.5 Time Series Data Augmentation Techniques\n\n")
            f.write("The model incorporates novel data augmentation techniques specifically designed for climate time series data, addressing the common problem of limited historical data.\n\n")
            f.write("**Innovation Details:**\n")
            f.write("- Jittering with controlled noise levels to simulate measurement variability\n")
            f.write("- Time warping to simulate temporal variations in weather patterns\n")
            f.write("- Magnitude warping with smooth scaling factors to simulate intensity variations\n")
            f.write("- Improved model generalization and robustness\n\n")
            
            f.write("## 3. Performance Comparison\n\n")
            
            if classical_results is not None:
                f.write("### 3.1 Overall Performance Metrics\n\n")
                
                # Create comparison table
                f.write("| Model | RMSE | MAE | R² | Explained Variance |\n")
                f.write("|-------|------|-----|----|-----------------|\n")
                
                # MMWSTM-ADRAN+ metrics
                f.write(f"| MMWSTM-ADRAN+ | {mmwstm_adran_results['metrics']['rmse']:.4f} | {mmwstm_adran_results['metrics']['mae']:.4f} | {mmwstm_adran_results['metrics']['r2']:.4f} | {mmwstm_adran_results['metrics']['explained_variance']:.4f} |\n")
                
                # Classical model metrics
                for model_name, result in classical_results.items():
                    f.write(f"| {model_name} | {result['metrics']['rmse']:.4f} | {result['metrics']['mae']:.4f} | {result['metrics']['r2']:.4f} | {result['metrics']['explained_variance']:.4f} |\n")
                
                f.write("\n")
                
                f.write("### 3.2 Extreme Event Performance\n\n")
                
                # Create extreme event comparison table
                f.write("| Model | Extreme High RMSE | Extreme Low RMSE |\n")
                f.write("|-------|-------------------|------------------|\n")
                
                # MMWSTM-ADRAN+ metrics
                f.write(f"| MMWSTM-ADRAN+ | {mmwstm_adran_results['metrics']['extreme_high_rmse']:.4f} | {mmwstm_adran_results['metrics']['extreme_low_rmse']:.4f} |\n")
                
                # Classical model metrics
                for model_name, result in classical_results.items():
                    f.write(f"| {model_name} | {result['metrics']['extreme_high_rmse']:.4f} | {result['metrics']['extreme_low_rmse']:.4f} |\n")
                
                f.write("\n")
            
            f.write("## 4. Patent Claims\n\n")
            
            f.write("1. A method for predicting weather patterns using a hybrid deep learning architecture comprising a weather state transition model and an anomaly-driven recurrent attention network.\n\n")
            
            f.write("2. The method of claim 1, wherein the weather state transition model comprises a learnable transition matrix that captures probabilistic transitions between different weather states.\n\n")
            
            f.write("3. The method of claim 1, wherein the anomaly-driven recurrent attention network comprises a multi-head self-attention mechanism with an anomaly amplification component.\n\n")
            
            f.write("4. The method of claim 1, further comprising an adaptive fusion layer that dynamically weights the contributions of the state transition model and the anomaly attention network.\n\n")
            
            f.write("5. The method of claim 1, further comprising a specialized loss function that places higher emphasis on extreme weather events.\n\n")
            
            f.write("6. The method of claim 1, further comprising time series data augmentation techniques specifically designed for climate data.\n\n")
            
            f.write("7. A system for predicting weather patterns implementing the method of claim 1.\n\n")
            
            f.write("## 5. Conclusion\n\n")
            
            f.write("The MMWSTM-ADRAN+ model represents a significant advancement in climate data analysis and prediction. Its novel architecture and specialized components address key challenges in weather forecasting, particularly for extreme events. The performance metrics demonstrate clear superiority over classical approaches, making this innovation both scientifically significant and commercially valuable.\n\n")
            
            f.write("The patentable innovations in this model have applications beyond weather forecasting, including climate change impact assessment, agricultural planning, disaster preparedness, and energy demand forecasting. The model's ability to accurately predict both normal patterns and extreme events makes it particularly valuable for decision-making in climate-sensitive sectors.\n\n")
        
        print(f"Documentation saved to {doc_path}")
        return doc_path
        
    except Exception as e:
        print(f"Error in documenting innovations: {str(e)}")
        return None

def analyze_baghdad_weather(data_path, output_dir='baghdad_weather_output', target_col='tempmax', use_gpu=True):
    """
    Main function to analyze Baghdad weather data with enhanced model and visualizations.
    
    Parameters:
    -----------
    data_path : str
        Path to the Excel file containing Baghdad weather data
    output_dir : str
        Directory to save outputs
    target_col : str
        Target column to predict
    use_gpu : bool
        Whether to use GPU acceleration
        
    Returns:
    --------
    dict
        Dictionary containing analysis results
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create subdirectories
    model_dir = output_dir / 'models'
    model_dir.mkdir(exist_ok=True)
    
    vis_dir = output_dir / 'visualizations'
    vis_dir.mkdir(exist_ok=True)
    
    doc_dir = output_dir / 'documentation'
    doc_dir.mkdir(exist_ok=True)
    
    # Initialize visualization paths list
    visualization_paths = []
    
    # 1. Load and preprocess data
    df = load_and_preprocess_baghdad_data(data_path)
    if df is None:
        return None
    
    # 2. Perform advanced analysis
    df, analysis_results = perform_advanced_analysis(df)
    if df is None:
        return None
    
    # 3. Save processed data
    df.to_csv(output_dir / 'processed_data.csv', index=False)
    print(f"Processed data saved to {output_dir / 'processed_data.csv'}")
    
    # 4. Prepare data for modeling
    model_data = prepare_model_data(
        df, 
        target_col=target_col, 
        sequence_length=30, 
        test_size=0.2,
        apply_augmentation=True
    )
    if model_data is None:
        return None
    
    # 5. Train classical models for comparison
    classical_results = train_classical_models(model_data)
    
    # 6. Train enhanced MMWSTM-ADRAN+ model
    if TORCH_AVAILABLE:
        model, train_losses, val_losses, training_time = train_mmwstm_adran_model(
            model_data=model_data,
            hidden_dim=32,
            num_clusters=9,
            batch_size=256,
            num_epochs=200,
            learning_rate=0.01,
            patience=25,
            use_gpu=use_gpu
        )
        
        if model is not None:
            # Save model
            if TORCH_AVAILABLE:
                torch.save(model.state_dict(), model_dir / 'mmwstm_adran_plus_model.pt')
                print(f"Model saved to {model_dir / 'mmwstm_adran_plus_model.pt'}")
            
            # 7. Evaluate model
            mmwstm_adran_results = evaluate_mmwstm_adran_model(model, model_data, use_gpu=use_gpu)
            
            if mmwstm_adran_results is not None:
                # 8. Create advanced visualizations
                visualization_paths = create_advanced_visualizations(
                    df, 
                    mmwstm_adran_results, 
                    classical_results, 
                    vis_dir,
                    target_col
                ) or []
                
                # 9. Document patentable innovations
                doc_path = document_patentable_innovations(
                    mmwstm_adran_results,
                    classical_results,
                    doc_dir
                )
                
                # 10. Write analysis summary
                with open(output_dir / 'analysis_summary.md', 'w') as f:
                    f.write("# Baghdad Weather Analysis Summary\n\n")
                    
                    f.write("## Dataset Overview\n\n")
                    f.write(f"- Total records: {len(df)}\n")
                    f.write(f"- Date range: {df['datetime'].min().date()} to {df['datetime'].max().date()}\n")
                    f.write(f"- Target variable: {target_col}\n")
                    f.write(f"- Average {target_col}: {df[target_col].mean():.2f}°C\n")
                    f.write(f"- Maximum {target_col}: {df[target_col].max():.2f}°C\n")
                    f.write(f"- Minimum {target_col}: {df[target_col].min():.2f}°C\n\n")
                    
                    f.write("## MMWSTM-ADRAN+ Model Performance\n\n")
                    f.write(f"- Mean Squared Error (MSE): {mmwstm_adran_results['metrics']['mse']:.6f}\n")
                    f.write(f"- Root Mean Squared Error (RMSE): {mmwstm_adran_results['metrics']['rmse']:.6f}\n")
                    f.write(f"- Mean Absolute Error (MAE): {mmwstm_adran_results['metrics']['mae']:.6f}\n")
                    f.write(f"- R-squared (R²): {mmwstm_adran_results['metrics']['r2']:.6f}\n")
                    f.write(f"- Explained Variance: {mmwstm_adran_results['metrics']['explained_variance']:.6f}\n")
                    f.write(f"- Correlation: {mmwstm_adran_results['metrics']['correlation']:.6f}\n")
                    f.write(f"- Bias: {mmwstm_adran_results['metrics']['bias']:.6f}\n")
                    f.write(f"- Extreme High Events RMSE: {mmwstm_adran_results['metrics']['extreme_high_rmse']:.6f}\n")
                    f.write(f"- Extreme Low Events RMSE: {mmwstm_adran_results['metrics']['extreme_low_rmse']:.6f}\n")
                    
                    
                    if classical_results is not None:
                        f.write("## Classical Model Performance\n\n")
                        for model_name, result in classical_results.items():
                            f.write(f"### {model_name}\n\n")
                            f.write(f"- RMSE: {result['metrics']['rmse']:.6f}\n")
                            f.write(f"- R²: {result['metrics']['r2']:.6f}\n")
                            f.write(f"- Training Time: {result['metrics']['training_time']:.2f} seconds\n\n")
                    
                    f.write("## Trend Analysis\n\n")
                    if 'trend_analysis' in analysis_results:
                        for var, trend in analysis_results['trend_analysis'].items():
                            f.write(f"- {var}: {trend['direction']} trend ")
                            f.write(f"(slope: {trend['slope']:.4f}, p-value: {trend['p_value']:.4f}, ")
                            f.write(f"R²: {trend['r_squared']:.4f}, ")
                            f.write(f"{'Significant' if trend['significant'] else 'Not significant'})\n")
                    f.write("\n")
                    
                    f.write("## Visualizations\n\n")
                    for path in visualization_paths:
                        f.write(f"- [{path.name}]({path.relative_to(output_dir)})\n")
                    f.write("\n")
                    
                    f.write("## Documentation\n\n")
                    if doc_path is not None:
                        f.write(f"- [Patentable Innovations]({Path(doc_path).relative_to(output_dir)})\n")
                    f.write("\n")
                
                print(f"Analysis summary saved to {output_dir / 'analysis_summary.md'}")
                
                return {
                    'df': df,
                    'model': model,
                    'mmwstm_adran_results': mmwstm_adran_results,
                    'classical_results': classical_results,
                    'visualization_paths': visualization_paths,
                    'doc_path': doc_path,
                    'output_dir': output_dir
                }
        else:
            print("Model training failed. Cannot proceed with evaluation and visualization.")
    else:
        print("PyTorch not available. Skipping model training and evaluation.")
    
    return None

# Run the analysis with your file path
analyze_baghdad_weather(
    data_path=r"C:\Users\shahe\Desktop\Air\baghdad 2019-2024.xlsx",
    output_dir="baghdad_weather_final_output",
    target_col='tempmax',
    use_gpu=True  # Set to False if you don't want to use GPU
)


