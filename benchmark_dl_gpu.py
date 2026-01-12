#!/usr/bin/env python3
"""
Benchmark: CPU vs MPS (Apple Silicon GPU) for DL Feature Extraction

Tests both approaches and reports which is faster for your M3 Ultra.
"""

import os
import sys
import time
import subprocess

def run_benchmark(mode: str, epochs: int = 15, sequence_length: int = 30) -> dict:
    """Run benchmark in a subprocess with specific device mode."""

    script = f'''
import os
import time
import warnings
warnings.filterwarnings('ignore')

# Set device mode BEFORE importing TensorFlow
MODE = "{mode}"
if MODE == "cpu":
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # Disable Metal
    os.environ['TF_DISABLE_MLC'] = '1'
elif MODE == "mps":
    # Enable Metal Performance Shaders
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        del os.environ['CUDA_VISIBLE_DEVICES']
    os.environ['TF_DISABLE_MLC'] = '0'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf

# Check device
print(f"TensorFlow version: {{tf.__version__}}")
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs available: {{gpus}}")

if MODE == "mps" and not gpus:
    print("WARNING: MPS not available, falling back to CPU")

# Import after setting env vars
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, Dropout, Concatenate, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

# Generate synthetic data similar to real use case
np.random.seed(42)
n_samples = 1500  # ~6 years of daily data
n_features = 20   # Typical feature count

print(f"\\nBenchmark Configuration:")
print(f"  Mode: {{MODE.upper()}}")
print(f"  Samples: {{n_samples}}")
print(f"  Features: {{n_features}}")
print(f"  Sequence Length: {sequence_length}")
print(f"  Epochs: {epochs}")

# Create synthetic price/indicator data
data = np.random.randn(n_samples, n_features).astype(np.float32)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Create sequences
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i-seq_len:i])
        y.append(data[i, 0])  # Predict first feature
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, {sequence_length})
print(f"  X shape: {{X.shape}}")
print(f"  y shape: {{y.shape}}")

# Build model (same architecture as DLFeatureExtractor)
def build_model(seq_len, n_features, encoding_dim=8):
    inputs = Input(shape=(seq_len, n_features))

    # Multi-scale CNN branches
    x30 = tf.keras.layers.Lambda(lambda x: x[:, -30:, :])(inputs)
    x30 = Conv1D(32, 3, activation='relu', padding='same')(x30)
    x30 = Dropout(0.2)(x30)
    x30 = Conv1D(32, 3, activation='relu', padding='same')(x30)
    x30 = GlobalAveragePooling1D()(x30)

    merged = Dense(64, activation='relu')(x30)
    merged = Dropout(0.4)(merged)
    bottleneck = Dense(encoding_dim, activation='linear', name='embedding')(merged)
    output = Dense(1, activation='linear')(bottleneck)

    model = Model(inputs, output)
    model.compile(optimizer='adam', loss='mse')
    return model

# Warmup run (compile graph)
print("\\nWarming up...")
warmup_model = build_model({sequence_length}, n_features)
warmup_model.fit(X[:100], y[:100], epochs=1, batch_size=32, verbose=0)
del warmup_model

# Benchmark training
print("\\nBenchmarking training...")
model = build_model({sequence_length}, n_features)

train_start = time.time()
history = model.fit(
    X, y,
    epochs={epochs},
    batch_size=32,
    validation_split=0.2,
    callbacks=[EarlyStopping(patience=3, restore_best_weights=True)],
    verbose=0
)
train_time = time.time() - train_start
actual_epochs = len(history.history['loss'])

# Benchmark inference
print("Benchmarking inference...")
n_inference_runs = 5
inference_times = []
for _ in range(n_inference_runs):
    inf_start = time.time()
    _ = model.predict(X, verbose=0)
    inference_times.append(time.time() - inf_start)

avg_inference = np.mean(inference_times)

# Results
print(f"\\n{'='*50}")
print(f"RESULTS - {{MODE.upper()}}")
print(f"{'='*50}")
print(f"Training time: {{train_time:.2f}}s ({{actual_epochs}} epochs)")
print(f"Time per epoch: {{train_time/actual_epochs:.2f}}s")
print(f"Inference time: {{avg_inference:.2f}}s (avg of {{n_inference_runs}} runs)")
print(f"Samples/second (train): {{n_samples * actual_epochs / train_time:.0f}}")
print(f"Samples/second (infer): {{n_samples / avg_inference:.0f}}")

# Output for parsing
print(f"\\n__RESULTS__:{{MODE}},{{train_time:.3f}},{{actual_epochs}},{{avg_inference:.3f}}")
'''

    # Run in subprocess
    result = subprocess.run(
        [sys.executable, '-c', script],
        capture_output=True,
        text=True,
        timeout=300
    )

    print(result.stdout)
    if result.stderr:
        # Filter out TF warnings
        stderr_lines = [l for l in result.stderr.split('\n')
                       if l and not any(x in l for x in ['WARNING', 'INFO', 'metal', 'cpu_feature_guard'])]
        if stderr_lines:
            print("STDERR:", '\n'.join(stderr_lines[:5]))

    # Parse results
    for line in result.stdout.split('\n'):
        if line.startswith('__RESULTS__:'):
            parts = line.split(':')[1].split(',')
            return {
                'mode': parts[0],
                'train_time': float(parts[1]),
                'epochs': int(parts[2]),
                'inference_time': float(parts[3])
            }

    return None


def main():
    print("=" * 60)
    print("DL FEATURE EXTRACTOR - CPU vs GPU BENCHMARK")
    print("=" * 60)
    print(f"System: Apple Silicon M3 Ultra")
    print(f"Testing: TensorFlow with Metal Performance Shaders (MPS)")
    print("=" * 60)

    results = {}

    # Test CPU
    print("\n" + "=" * 60)
    print("TEST 1: CPU MODE")
    print("=" * 60)
    try:
        results['cpu'] = run_benchmark('cpu')
    except Exception as e:
        print(f"CPU benchmark failed: {e}")
        results['cpu'] = None

    # Test MPS (GPU)
    print("\n" + "=" * 60)
    print("TEST 2: MPS (GPU) MODE")
    print("=" * 60)
    try:
        results['mps'] = run_benchmark('mps')
    except Exception as e:
        print(f"MPS benchmark failed: {e}")
        results['mps'] = None

    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    if results.get('cpu') and results.get('mps'):
        cpu = results['cpu']
        mps = results['mps']

        print(f"\n{'Metric':<25} {'CPU':>12} {'MPS (GPU)':>12} {'Winner':>10}")
        print("-" * 60)

        # Training
        cpu_train = cpu['train_time']
        mps_train = mps['train_time']
        train_winner = "CPU" if cpu_train < mps_train else "MPS"
        train_speedup = max(cpu_train, mps_train) / min(cpu_train, mps_train)
        print(f"{'Training Time':<25} {cpu_train:>10.2f}s {mps_train:>10.2f}s {train_winner:>10}")

        # Inference
        cpu_inf = cpu['inference_time']
        mps_inf = mps['inference_time']
        inf_winner = "CPU" if cpu_inf < mps_inf else "MPS"
        inf_speedup = max(cpu_inf, mps_inf) / min(cpu_inf, mps_inf)
        print(f"{'Inference Time':<25} {cpu_inf:>10.2f}s {mps_inf:>10.2f}s {inf_winner:>10}")

        # Overall recommendation
        print("\n" + "=" * 60)
        print("RECOMMENDATION")
        print("=" * 60)

        if train_winner == "MPS" and train_speedup > 1.2:
            print(f"\n✅ USE MPS (GPU) - {train_speedup:.1f}x faster training")
            print("\nTo enable MPS, edit dl_feature_extractor.py:")
            print("  Remove or comment out lines 14-16:")
            print("    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'")
        elif train_winner == "CPU" and train_speedup > 1.2:
            print(f"\n✅ KEEP CPU - {train_speedup:.1f}x faster than GPU")
            print("\nCPU is optimal for this model size on M3 Ultra.")
        else:
            print(f"\n⚖️  SIMILAR PERFORMANCE - difference is only {train_speedup:.1f}x")
            print("\nKeep CPU for stability (avoids potential Metal issues).")

    else:
        print("\nCould not complete both benchmarks.")
        if results.get('cpu'):
            print(f"CPU results: {results['cpu']}")
        if results.get('mps'):
            print(f"MPS results: {results['mps']}")


if __name__ == "__main__":
    main()
