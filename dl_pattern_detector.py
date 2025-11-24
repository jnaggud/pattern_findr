import logging
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# TensorFlow CPU-only mode already set by app.py environment variables
# Import quietly without additional configuration
import tensorflow as tf
from tensorflow.keras import layers, models
from datetime import datetime
import glob
from pattern_data_generator import generate_all_training_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create directory for chart images if it doesn't exist
os.makedirs('chart_images', exist_ok=True)

def create_chart_image(data, index, window_size=60, save_path=None):
    """
    Creates a candlestick chart image from a segment of stock data.

    Args:
        data (pd.DataFrame): The full historical data for the stock.
        index (int): The ending index in the DataFrame for the chart window.
        window_size (int): The number of days to include in the chart.
        save_path (str, optional): The path to save the image file. If None, the image is not saved.

    Returns:
        str: The path to the saved image, or None if not saved.
    """
    try:
        # Ensure index is valid
        if index < window_size:
            logging.warning(f"Index {index} is less than window_size {window_size}. Adjusting window.")
            window_size = index
        
        # Extract the window of data
        start_idx = max(0, index - window_size)
        end_idx = index
        window_data = data.iloc[start_idx:end_idx+1].copy()
        
        # Ensure we have a proper date index for mplfinance
        if 'Date' in window_data.columns:
            window_data = window_data.set_index('Date')
        
        # Use mplfinance to create a candlestick chart and save it directly
        if save_path:
            mpf.plot(
                window_data,
                type='candle',
                style='charles',
                title=f'Candlestick Chart',
                ylabel='Price',
                volume=False,
                figsize=(10, 6),
                savefig=save_path
            )
            logging.info(f"Chart image saved to {save_path}")
            return save_path
        else:
            # Generate a default filename based on timestamp
            os.makedirs('chart_images', exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"chart_images/chart_{timestamp}_{index}.png"
            mpf.plot(
                window_data,
                type='candle',
                style='charles',
                title=f'Candlestick Chart',
                ylabel='Price',
                volume=False,
                figsize=(10, 6),
                savefig=filename
            )
            logging.info(f"Chart image saved to {filename}")
            return filename
            
    except Exception as e:
        logging.error(f"Error creating chart image: {e}", exc_info=True)
        return None

def generate_training_data(data, pattern_indices, non_pattern_indices, window_size=60, pattern_name=None):
    """
    Generates training data for the deep learning model by creating chart images
    for both pattern and non-pattern examples.
    
    Args:
        data (pd.DataFrame): The full historical data for the stock.
        pattern_indices (list): List of indices where the pattern appears.
        non_pattern_indices (list): List of indices where the pattern does not appear.
        window_size (int): The number of days to include in each chart.
        pattern_name (str, optional): Name of the pattern for file naming.
        
    Returns:
        tuple: (X_train, y_train) where X_train is a list of image paths and y_train is a list of labels.
    """
    X_train = []
    y_train = []
    
    pattern_prefix = pattern_name if pattern_name else "pattern"
    
    # Generate pattern examples
    for idx in pattern_indices:
        save_path = f"chart_images/{pattern_prefix}_{idx}.png"
        create_chart_image(data, idx, window_size, save_path)
        X_train.append(save_path)
        y_train.append(1)  # 1 = pattern present
    
    # Generate non-pattern examples
    for idx in non_pattern_indices:
        save_path = f"chart_images/non_{pattern_prefix}_{idx}.png"
        create_chart_image(data, idx, window_size, save_path)
        X_train.append(save_path)
        y_train.append(0)  # 0 = pattern not present
    
    return X_train, y_train

def build_cnn_model(input_shape=(224, 224, 3)):
    """
    Builds a simple CNN model for image classification.
    
    Args:
        input_shape (tuple): The shape of the input images.
        
    Returns:
        tf.keras.Model: The compiled CNN model.
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def use_synthetic_data_for_training(pattern_name, num_samples=50):
    """
    Use synthetic data for training a model for a specific pattern.
    
    Args:
        pattern_name (str): Name of the pattern to train for (e.g., 'head_and_shoulders')
        num_samples (int): Number of samples to generate
        
    Returns:
        tuple: (X_train, y_train) where X_train is a list of image paths and y_train is a list of labels
    """
    # Convert pattern name to the format used in pattern_data_generator
    pattern_type = pattern_name.lower().replace(' ', '_').replace('-', '_')
    
    # Check if we already have generated data for this pattern
    pattern_files = glob.glob(f"chart_images/{pattern_type}_*.png")
    non_pattern_files = glob.glob(f"chart_images/non_{pattern_type}_*.png")
    
    # If we don't have enough data, generate it
    if len(pattern_files) < num_samples // 2 or len(non_pattern_files) < num_samples // 2:
        logging.info(f"Generating synthetic data for {pattern_name}...")
        training_data = generate_all_training_data([pattern_type], num_samples)
        X_paths, y_labels = training_data[pattern_type]
    else:
        logging.info(f"Using existing synthetic data for {pattern_name}...")
        X_paths = pattern_files + non_pattern_files
        y_labels = [1] * len(pattern_files) + [0] * len(non_pattern_files)
    
    return X_paths, y_labels

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """
    Loads and preprocesses an image for the CNN model.
    
    Args:
        image_path (str): Path to the image file.
        target_size (tuple): The target size for the image.
        
    Returns:
        np.array: The preprocessed image.
    """
    try:
        img = tf.keras.preprocessing.image.load_img(
            image_path, 
            target_size=target_size
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = img_array / 255.0  # Normalize to [0,1]
        return img_array
    except Exception as e:
        logging.error(f"Error loading image {image_path}: {e}")
        return None
