import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from analysis import analyze_window_and_overlap
from visualization import visualize_results, get_and_print_optimal_params

# CONSTANTS AND SETTINGS
AUDIO_DATA = [
    {
        "mixture": Path("/home/ubnt/Real-Time-Voice-Separation-with-Multiple-Speakers/audio/mixture_input.wav"),
        "clean1": Path("/home/ubnt/Real-Time-Voice-Separation-with-Multiple-Speakers/audio/clean1.wav"),
        "clean2": Path("/home/ubnt/Real-Time-Voice-Separation-with-Multiple-Speakers/audio/clean2.wav"),
    },
    {
        "mixture": Path("/home/ubnt/storage/Libri2Mix/wav8k/min/test/mix_both/61-70968-0000_8455-210777-0012.wav"),
        "clean1": Path("/home/ubnt/storage/Libri2Mix/wav8k/min/test/s1/61-70968-0000_8455-210777-0012.wav"),
        "clean2": Path("/home/ubnt/storage/Libri2Mix/wav8k/min/test/s2/61-70968-0000_8455-210777-0012.wav"),
    },
]
MODEL_NAMES = ["resepformer-wsj02mix"]
WINDOW_DURATIONS = np.arange(1, 3, 1)
OVERLAP_RATIOS = np.arange(0.0, 0.5, 0.3)
ANALYZE_BY_CHUNK = True

# DEVICE SETUP
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("CUDA is available. Using GPU.")
else:
    DEVICE = torch.device("cpu")
    print("CUDA is not available. Using CPU.")

# RESULTS DIRECTORY
TIMESTAMP = datetime.now().strftime("%m-%d_%H-%M")
RESULTS_NAME = f"window_overlap_analysis_{TIMESTAMP}"
RESULTS_DIR = os.path.join("results", RESULTS_NAME)
os.makedirs(RESULTS_DIR, exist_ok=True)

# INITIALIZE ACCUMULATORS
all_results_df = []
all_chunk_metrics_df = []

# PROCESS AUDIO DATA
for audio_files in AUDIO_DATA:
    print(f"\nProcessing mixture: {audio_files['mixture']}")
    results_df, chunk_metrics_df = analyze_window_and_overlap(
        DEVICE,
        audio_files["mixture"],
        audio_files["clean1"],
        audio_files["clean2"],
        MODEL_NAMES,
        RESULTS_DIR,
        WINDOW_DURATIONS,
        OVERLAP_RATIOS,
        ANALYZE_BY_CHUNK,
    )
    if results_df is not None:
        all_results_df.append(results_df)
    if ANALYZE_BY_CHUNK and chunk_metrics_df is not None:
        all_chunk_metrics_df.append(chunk_metrics_df)

# AVERAGE AND VISUALIZE RESULTS
if all_results_df:
    # Group by model, window_duration, and overlap_ratio, then calculate mean of numeric columns
    numeric_columns = [
        "sdr",
        "sir",
        "sar",
        "total_separation_time",
        "average_window_time",
        "std_window_time",
    ]
    average_results_df = pd.concat(all_results_df).groupby(
        ["model", "window_duration", "overlap_ratio"]
    )[numeric_columns].mean().reset_index()

    # Handle chunk metrics if they exist
    average_chunk_metrics_df = None
    if ANALYZE_BY_CHUNK and all_chunk_metrics_df:
        numeric_chunk_columns = ["chunk_sdr", "chunk_sir", "chunk_sar"]
        average_chunk_metrics_df = pd.concat(all_chunk_metrics_df).groupby(
            ["model", "window_duration", "overlap_ratio", "chunk_index"]
        )[numeric_chunk_columns].mean().reset_index()

    # Visualize averaged results and get optimal parameters
    get_and_print_optimal_params(
        average_results_df, average_chunk_metrics_df if ANALYZE_BY_CHUNK else None
    )
    visualize_results(
        average_results_df, RESULTS_DIR, average_chunk_metrics_df if ANALYZE_BY_CHUNK else None
    )
else:
    print("No results to analyze. Check if the analysis produced any valid data.")
