import numpy as np
import pandas as pd
import os
import random
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import mplfinance as mpf
from datetime import datetime, timedelta

def generate_synthetic_price_data(days=100, pattern_type=None, volatility=0.02, trend_strength=0.001, seed=None):
    """
    Generate synthetic price data with optional patterns embedded.
    
    Args:
        days (int): Number of days of data to generate
        pattern_type (str): Type of pattern to embed ('head_and_shoulders', 'cup_and_handle', etc.)
        volatility (float): Daily price volatility
        trend_strength (float): Strength of the trend
        seed (int, optional): Random seed for reproducibility. If None, uses a random seed.
        
    Returns:
        pd.DataFrame: DataFrame with OHLCV data
    """
    # Start with a base price
    base_price = 100.0
    
    # Set random seed if provided, otherwise use system time
    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed(int(os.path.getmtime(__file__)) % 100000 + random.randint(0, 100000))
        
    # Generate random price movements
    daily_returns = np.random.normal(0, volatility, days)
    
    # Add a slight upward trend
    daily_returns += trend_strength
    
    # Calculate cumulative returns
    cumulative_returns = np.cumprod(1 + daily_returns)
    
    # Generate close prices
    close_prices = base_price * cumulative_returns
    
    # Generate other price data
    high_prices = close_prices * (1 + np.random.uniform(0, 0.015, days))
    low_prices = close_prices * (1 - np.random.uniform(0, 0.015, days))
    open_prices = low_prices + np.random.uniform(0, 1, days) * (high_prices - low_prices)
    
    # Generate volume data
    volume = np.random.randint(100000, 1000000, days)
    
    # Generate dates
    end_date = datetime.now()
    dates = [end_date - timedelta(days=i) for i in range(days, 0, -1)]
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': volume
    })
    
    # Apply pattern modifications if specified
    if pattern_type:
        df = embed_pattern(df, pattern_type)
    
    # Set Date as index
    df.set_index('Date', inplace=True)
    
    return df

def embed_pattern(df, pattern_type):
    """
    Modify the price data to embed a specific chart pattern.
    
    Args:
        df (pd.DataFrame): Original price data
        pattern_type (str): Type of pattern to embed
        
    Returns:
        pd.DataFrame: Modified DataFrame with the pattern embedded
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    if pattern_type == 'head_and_shoulders':
        # Head and Shoulders pattern (bearish reversal)
        # Typically has 3 peaks with the middle one (head) being the highest
        start_idx = len(df) // 3
        end_idx = start_idx + len(df) // 3
        
        # Create left shoulder
        peak1_idx = start_idx + len(df) // 20
        for i in range(start_idx, peak1_idx):
            progress = (i - start_idx) / (peak1_idx - start_idx)
            df.iloc[i, df.columns.get_loc('Close')] *= (1 + 0.05 * np.sin(progress * np.pi))
            df.iloc[i, df.columns.get_loc('High')] = max(df.iloc[i, df.columns.get_loc('High')], 
                                                       df.iloc[i, df.columns.get_loc('Close')] * 1.01)
        
        # Create head
        peak2_idx = (start_idx + end_idx) // 2
        for i in range(peak1_idx, peak2_idx + (peak2_idx - peak1_idx)):
            progress = (i - peak1_idx) / (peak2_idx - peak1_idx)
            if progress <= 1:
                df.iloc[i, df.columns.get_loc('Close')] *= (1 + 0.1 * np.sin(progress * np.pi))
            else:
                df.iloc[i, df.columns.get_loc('Close')] *= (1 + 0.1 * np.sin((2 - progress) * np.pi))
            df.iloc[i, df.columns.get_loc('High')] = max(df.iloc[i, df.columns.get_loc('High')], 
                                                       df.iloc[i, df.columns.get_loc('Close')] * 1.01)
        
        # Create right shoulder
        peak3_idx = end_idx - len(df) // 20
        for i in range(peak2_idx + (peak2_idx - peak1_idx), end_idx):
            progress = (i - (peak2_idx + (peak2_idx - peak1_idx))) / (end_idx - (peak2_idx + (peak2_idx - peak1_idx)))
            df.iloc[i, df.columns.get_loc('Close')] *= (1 + 0.05 * np.sin(progress * np.pi))
            df.iloc[i, df.columns.get_loc('High')] = max(df.iloc[i, df.columns.get_loc('High')], 
                                                       df.iloc[i, df.columns.get_loc('Close')] * 1.01)
        
        # Adjust other columns
        for i in range(start_idx, end_idx):
            df.iloc[i, df.columns.get_loc('Low')] = min(df.iloc[i, df.columns.get_loc('Low')], 
                                                      df.iloc[i, df.columns.get_loc('Close')] * 0.99)
            df.iloc[i, df.columns.get_loc('Open')] = df.iloc[i, df.columns.get_loc('Close')] * (0.99 + 0.02 * np.random.random())
    
    elif pattern_type == 'inverse_head_and_shoulders':
        # Inverse Head and Shoulders pattern (bullish reversal)
        # Same as head and shoulders but inverted
        start_idx = len(df) // 3
        end_idx = start_idx + len(df) // 3
        
        # Create left shoulder
        peak1_idx = start_idx + len(df) // 20
        for i in range(start_idx, peak1_idx):
            progress = (i - start_idx) / (peak1_idx - start_idx)
            df.iloc[i, df.columns.get_loc('Close')] *= (1 - 0.05 * np.sin(progress * np.pi))
            df.iloc[i, df.columns.get_loc('Low')] = min(df.iloc[i, df.columns.get_loc('Low')], 
                                                      df.iloc[i, df.columns.get_loc('Close')] * 0.99)
        
        # Create head
        peak2_idx = (start_idx + end_idx) // 2
        for i in range(peak1_idx, peak2_idx + (peak2_idx - peak1_idx)):
            progress = (i - peak1_idx) / (peak2_idx - peak1_idx)
            if progress <= 1:
                df.iloc[i, df.columns.get_loc('Close')] *= (1 - 0.1 * np.sin(progress * np.pi))
            else:
                df.iloc[i, df.columns.get_loc('Close')] *= (1 - 0.1 * np.sin((2 - progress) * np.pi))
            df.iloc[i, df.columns.get_loc('Low')] = min(df.iloc[i, df.columns.get_loc('Low')], 
                                                      df.iloc[i, df.columns.get_loc('Close')] * 0.99)
        
        # Create right shoulder
        peak3_idx = end_idx - len(df) // 20
        for i in range(peak2_idx + (peak2_idx - peak1_idx), end_idx):
            progress = (i - (peak2_idx + (peak2_idx - peak1_idx))) / (end_idx - (peak2_idx + (peak2_idx - peak1_idx)))
            df.iloc[i, df.columns.get_loc('Close')] *= (1 - 0.05 * np.sin(progress * np.pi))
            df.iloc[i, df.columns.get_loc('Low')] = min(df.iloc[i, df.columns.get_loc('Low')], 
                                                      df.iloc[i, df.columns.get_loc('Close')] * 0.99)
        
        # Adjust other columns
        for i in range(start_idx, end_idx):
            df.iloc[i, df.columns.get_loc('High')] = max(df.iloc[i, df.columns.get_loc('High')], 
                                                       df.iloc[i, df.columns.get_loc('Close')] * 1.01)
            df.iloc[i, df.columns.get_loc('Open')] = df.iloc[i, df.columns.get_loc('Close')] * (0.99 + 0.02 * np.random.random())
    
    elif pattern_type == 'cup_and_handle':
        # Cup and Handle pattern (bullish continuation)
        # U-shaped cup followed by a small downward drift (the handle)
        start_idx = len(df) // 4
        cup_end_idx = start_idx + len(df) // 3
        handle_end_idx = cup_end_idx + len(df) // 10
        
        # Create cup (U-shape)
        for i in range(start_idx, cup_end_idx):
            progress = (i - start_idx) / (cup_end_idx - start_idx)
            df.iloc[i, df.columns.get_loc('Close')] *= (1 - 0.08 * np.sin(progress * np.pi))
            df.iloc[i, df.columns.get_loc('Low')] = min(df.iloc[i, df.columns.get_loc('Low')], 
                                                      df.iloc[i, df.columns.get_loc('Close')] * 0.99)
        
        # Create handle (small downward drift)
        for i in range(cup_end_idx, handle_end_idx):
            progress = (i - cup_end_idx) / (handle_end_idx - cup_end_idx)
            df.iloc[i, df.columns.get_loc('Close')] *= (1 - 0.03 * np.sin(progress * np.pi))
            df.iloc[i, df.columns.get_loc('Low')] = min(df.iloc[i, df.columns.get_loc('Low')], 
                                                      df.iloc[i, df.columns.get_loc('Close')] * 0.99)
        
        # Adjust other columns
        for i in range(start_idx, handle_end_idx):
            df.iloc[i, df.columns.get_loc('High')] = max(df.iloc[i, df.columns.get_loc('High')], 
                                                       df.iloc[i, df.columns.get_loc('Close')] * 1.01)
            df.iloc[i, df.columns.get_loc('Open')] = df.iloc[i, df.columns.get_loc('Close')] * (0.99 + 0.02 * np.random.random())
    
    elif pattern_type == 'double_top':
        # Double Top pattern (bearish reversal)
        # Two peaks of similar height with a moderate trough in between
        start_idx = len(df) // 3
        mid_idx = start_idx + len(df) // 6
        end_idx = start_idx + len(df) // 3
        
        # Create first peak
        peak1_idx = start_idx + len(df) // 20
        for i in range(start_idx, peak1_idx):
            progress = (i - start_idx) / (peak1_idx - start_idx)
            df.iloc[i, df.columns.get_loc('Close')] *= (1 + 0.08 * np.sin(progress * np.pi))
            df.iloc[i, df.columns.get_loc('High')] = max(df.iloc[i, df.columns.get_loc('High')], 
                                                       df.iloc[i, df.columns.get_loc('Close')] * 1.01)
        
        # Create trough
        trough_idx = (peak1_idx + mid_idx) // 2
        for i in range(peak1_idx, trough_idx):
            progress = (i - peak1_idx) / (trough_idx - peak1_idx)
            df.iloc[i, df.columns.get_loc('Close')] *= (1 + 0.08 * (1 - progress))
            df.iloc[i, df.columns.get_loc('High')] = max(df.iloc[i, df.columns.get_loc('High')], 
                                                       df.iloc[i, df.columns.get_loc('Close')] * 1.01)
        
        # Create second peak
        peak2_idx = mid_idx + len(df) // 20
        for i in range(trough_idx, peak2_idx):
            progress = (i - trough_idx) / (peak2_idx - trough_idx)
            df.iloc[i, df.columns.get_loc('Close')] *= (1 + 0.08 * progress)
            df.iloc[i, df.columns.get_loc('High')] = max(df.iloc[i, df.columns.get_loc('High')], 
                                                       df.iloc[i, df.columns.get_loc('Close')] * 1.01)
        
        # Create decline
        for i in range(peak2_idx, end_idx):
            progress = (i - peak2_idx) / (end_idx - peak2_idx)
            df.iloc[i, df.columns.get_loc('Close')] *= (1 + 0.08 * (1 - progress))
            df.iloc[i, df.columns.get_loc('High')] = max(df.iloc[i, df.columns.get_loc('High')], 
                                                       df.iloc[i, df.columns.get_loc('Close')] * 1.01)
        
        # Adjust other columns
        for i in range(start_idx, end_idx):
            df.iloc[i, df.columns.get_loc('Low')] = min(df.iloc[i, df.columns.get_loc('Low')], 
                                                      df.iloc[i, df.columns.get_loc('Close')] * 0.99)
            df.iloc[i, df.columns.get_loc('Open')] = df.iloc[i, df.columns.get_loc('Close')] * (0.99 + 0.02 * np.random.random())
    
    elif pattern_type == 'double_bottom':
        # Double Bottom pattern (bullish reversal)
        # Two troughs of similar depth with a moderate peak in between
        start_idx = len(df) // 3
        mid_idx = start_idx + len(df) // 6
        end_idx = start_idx + len(df) // 3
        
        # Create first trough
        trough1_idx = start_idx + len(df) // 20
        for i in range(start_idx, trough1_idx):
            progress = (i - start_idx) / (trough1_idx - start_idx)
            df.iloc[i, df.columns.get_loc('Close')] *= (1 - 0.08 * np.sin(progress * np.pi))
            df.iloc[i, df.columns.get_loc('Low')] = min(df.iloc[i, df.columns.get_loc('Low')], 
                                                      df.iloc[i, df.columns.get_loc('Close')] * 0.99)
        
        # Create peak
        peak_idx = (trough1_idx + mid_idx) // 2
        for i in range(trough1_idx, peak_idx):
            progress = (i - trough1_idx) / (peak_idx - trough1_idx)
            df.iloc[i, df.columns.get_loc('Close')] *= (1 - 0.08 * (1 - progress))
            df.iloc[i, df.columns.get_loc('Low')] = min(df.iloc[i, df.columns.get_loc('Low')], 
                                                      df.iloc[i, df.columns.get_loc('Close')] * 0.99)
        
        # Create second trough
        trough2_idx = mid_idx + len(df) // 20
        for i in range(peak_idx, trough2_idx):
            progress = (i - peak_idx) / (trough2_idx - peak_idx)
            df.iloc[i, df.columns.get_loc('Close')] *= (1 - 0.08 * progress)
            df.iloc[i, df.columns.get_loc('Low')] = min(df.iloc[i, df.columns.get_loc('Low')], 
                                                      df.iloc[i, df.columns.get_loc('Close')] * 0.99)
        
        # Create recovery
        for i in range(trough2_idx, end_idx):
            progress = (i - trough2_idx) / (end_idx - trough2_idx)
            df.iloc[i, df.columns.get_loc('Close')] *= (1 - 0.08 * (1 - progress))
            df.iloc[i, df.columns.get_loc('Low')] = min(df.iloc[i, df.columns.get_loc('Low')], 
                                                      df.iloc[i, df.columns.get_loc('Close')] * 0.99)
        
        # Adjust other columns
        for i in range(start_idx, end_idx):
            df.iloc[i, df.columns.get_loc('High')] = max(df.iloc[i, df.columns.get_loc('High')], 
                                                       df.iloc[i, df.columns.get_loc('Close')] * 1.01)
            df.iloc[i, df.columns.get_loc('Open')] = df.iloc[i, df.columns.get_loc('Close')] * (0.99 + 0.02 * np.random.random())
    
    elif pattern_type == 'triangle':
        # Triangle pattern (can be ascending, descending, or symmetrical)
        # Here we'll implement a symmetrical triangle
        start_idx = len(df) // 3
        end_idx = start_idx + len(df) // 3
        
        high_start = df.iloc[start_idx, df.columns.get_loc('Close')] * 1.1
        low_start = df.iloc[start_idx, df.columns.get_loc('Close')] * 0.9
        
        for i in range(start_idx, end_idx):
            progress = (i - start_idx) / (end_idx - start_idx)
            high_val = high_start * (1 - 0.5 * progress)
            low_val = low_start * (1 + 0.5 * progress)
            
            # Oscillate between high and low
            oscillation = np.sin(progress * 5 * np.pi)
            df.iloc[i, df.columns.get_loc('Close')] = low_val + (high_val - low_val) * (0.5 + 0.5 * oscillation)
            df.iloc[i, df.columns.get_loc('High')] = max(df.iloc[i, df.columns.get_loc('Close')] * 1.01, min(high_val, high_val * (1 - 0.4 * progress)))
            df.iloc[i, df.columns.get_loc('Low')] = min(df.iloc[i, df.columns.get_loc('Close')] * 0.99, max(low_val, low_val * (1 + 0.4 * progress)))
            df.iloc[i, df.columns.get_loc('Open')] = df.iloc[i, df.columns.get_loc('Close')] * (0.99 + 0.02 * np.random.random())
    
    elif pattern_type == 'flag':
        # Flag pattern (continuation pattern after a sharp move)
        # First create a sharp move, then a channel against the trend
        start_idx = len(df) // 3
        pole_end_idx = start_idx + len(df) // 10
        flag_end_idx = pole_end_idx + len(df) // 6
        
        # Create the pole (sharp move up)
        for i in range(start_idx, pole_end_idx):
            progress = (i - start_idx) / (pole_end_idx - start_idx)
            df.iloc[i, df.columns.get_loc('Close')] *= (1 + 0.15 * progress)
            df.iloc[i, df.columns.get_loc('High')] = max(df.iloc[i, df.columns.get_loc('High')], 
                                                       df.iloc[i, df.columns.get_loc('Close')] * 1.01)
        
        # Create the flag (channel sloping against the trend)
        for i in range(pole_end_idx, flag_end_idx):
            progress = (i - pole_end_idx) / (flag_end_idx - pole_end_idx)
            df.iloc[i, df.columns.get_loc('Close')] *= (1 + 0.15 - 0.05 * progress)
            
            # Add oscillation within the channel
            oscillation = np.sin(progress * 6 * np.pi)
            df.iloc[i, df.columns.get_loc('Close')] *= (1 + 0.02 * oscillation)
            
            df.iloc[i, df.columns.get_loc('High')] = max(df.iloc[i, df.columns.get_loc('High')], 
                                                       df.iloc[i, df.columns.get_loc('Close')] * 1.01)
            df.iloc[i, df.columns.get_loc('Low')] = min(df.iloc[i, df.columns.get_loc('Low')], 
                                                      df.iloc[i, df.columns.get_loc('Close')] * 0.99)
        
        # Adjust other columns
        for i in range(start_idx, flag_end_idx):
            df.iloc[i, df.columns.get_loc('Open')] = df.iloc[i, df.columns.get_loc('Close')] * (0.99 + 0.02 * np.random.random())
    
    elif pattern_type == 'pennant':
        # Pennant pattern (similar to flag but with converging trendlines)
        # First create a sharp move, then a symmetrical triangle
        start_idx = len(df) // 3
        pole_end_idx = start_idx + len(df) // 10
        pennant_end_idx = pole_end_idx + len(df) // 6
        
        # Create the pole (sharp move up)
        for i in range(start_idx, pole_end_idx):
            progress = (i - start_idx) / (pole_end_idx - start_idx)
            df.iloc[i, df.columns.get_loc('Close')] *= (1 + 0.15 * progress)
            df.iloc[i, df.columns.get_loc('High')] = max(df.iloc[i, df.columns.get_loc('High')], 
                                                       df.iloc[i, df.columns.get_loc('Close')] * 1.01)
        
        # Create the pennant (converging trendlines)
        high_start = df.iloc[pole_end_idx, df.columns.get_loc('Close')] * 1.05
        low_start = df.iloc[pole_end_idx, df.columns.get_loc('Close')] * 0.95
        
        for i in range(pole_end_idx, pennant_end_idx):
            progress = (i - pole_end_idx) / (pennant_end_idx - pole_end_idx)
            high_val = high_start * (1 - 0.3 * progress)
            low_val = low_start * (1 + 0.3 * progress)
            
            # Oscillate between high and low
            oscillation = np.sin(progress * 6 * np.pi)
            df.iloc[i, df.columns.get_loc('Close')] = low_val + (high_val - low_val) * (0.5 + 0.5 * oscillation)
            df.iloc[i, df.columns.get_loc('High')] = max(df.iloc[i, df.columns.get_loc('Close')] * 1.01, min(high_val, high_val * (1 - 0.3 * progress)))
            df.iloc[i, df.columns.get_loc('Low')] = min(df.iloc[i, df.columns.get_loc('Close')] * 0.99, max(low_val, low_val * (1 + 0.3 * progress)))
            df.iloc[i, df.columns.get_loc('Open')] = df.iloc[i, df.columns.get_loc('Close')] * (0.99 + 0.02 * np.random.random())
    
    return df

def save_chart_image(df, pattern_type, index, save_dir='chart_images'):
    """
    Save a chart image of the data.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
        pattern_type (str): Type of pattern embedded in the data
        index (int): Index number for the filename
        save_dir (str): Directory to save the image
        
    Returns:
        str: Path to the saved image
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create filename
    filename = f"{save_dir}/{pattern_type}_{index}.png"
    
    # Plot and save the chart
    mpf.plot(
        df,
        type='candle',
        style='charles',
        title=f'{pattern_type.replace("_", " ").title()} Pattern',
        ylabel='Price',
        volume=False,
        figsize=(10, 6),
        savefig=filename
    )
    
    return filename

def generate_training_data_for_pattern(pattern_type, num_samples=100):
    """
    Generate training data for a specific pattern.
    
    Args:
        pattern_type (str): Type of pattern to generate
        num_samples (int): Number of samples to generate
        
    Returns:
        tuple: (X_paths, y_labels) where X_paths is a list of image paths and y_labels is a list of labels
    """
    X_paths = []
    y_labels = []
    
    # Generate pattern examples
    for i in range(num_samples // 2):
        # Generate data with the pattern - use a unique seed for each sample
        seed = random.randint(0, 1000000) + i
        df = generate_synthetic_price_data(pattern_type=pattern_type, seed=seed)
        
        # Save chart image
        image_path = save_chart_image(df, pattern_type, i)
        
        X_paths.append(image_path)
        y_labels.append(1)  # 1 = pattern present
    
    # Generate non-pattern examples
    for i in range(num_samples // 2):
        # Generate data without any pattern - use a unique seed for each sample
        seed = random.randint(1000000, 2000000) + i
        df = generate_synthetic_price_data(seed=seed)
        
        # Save chart image
        image_path = save_chart_image(df, f"non_{pattern_type}", i)
        
        X_paths.append(image_path)
        y_labels.append(0)  # 0 = pattern not present
    
    return X_paths, y_labels

def generate_all_training_data(patterns, num_samples=100):
    """
    Generate training data for all specified patterns.
    
    Args:
        patterns (list): List of pattern types to generate
        num_samples (int): Number of samples to generate per pattern
        
    Returns:
        dict: Dictionary mapping pattern types to (X_paths, y_labels) tuples
    """
    results = {}
    
    for pattern in patterns:
        print(f"Generating training data for {pattern}...")
        X_paths, y_labels = generate_training_data_for_pattern(pattern, num_samples)
        results[pattern] = (X_paths, y_labels)
        print(f"Generated {len(X_paths)} images for {pattern}")
    
    return results

if __name__ == "__main__":
    # List of patterns to generate
    patterns = [
        'head_and_shoulders',
        'inverse_head_and_shoulders',
        'cup_and_handle',
        'double_top',
        'double_bottom',
        'triangle',
        'flag',
        'pennant'
    ]
    
    # Generate training data for all patterns (100 samples per pattern)
    training_data = generate_all_training_data(patterns, num_samples=100)
    
    # Print summary
    print("\nTraining data generation complete!")
    print(f"Total patterns: {len(patterns)}")
    print(f"Total images: {sum(len(result[0]) for result in training_data.values())}")
    
    # Print detailed summary per pattern
    for pattern, (X_paths, y_labels) in training_data.items():
        print(f"{pattern}: {len(X_paths)} images, {sum(y_labels)} positive examples, {len(y_labels) - sum(y_labels)} negative examples")
    
    print("\nImages saved to 'chart_images' directory")
