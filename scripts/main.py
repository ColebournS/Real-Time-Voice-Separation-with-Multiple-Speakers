import os
from datetime import datetime
from analysis import analyze_window_and_overlap
from visualization import visualize_results, get_and_print_optimal_params
import numpy as np

# Set up paths to audio files
mixture_path = r"C:\Users\samco\WPI\MQP\separation_test\audio\mixture_input.wav"
clean1_path = r"C:\Users\samco\WPI\MQP\separation_test\audio\clean1.wav"
clean2_path = r"C:\Users\samco\WPI\MQP\separation_test\audio\clean2.wav"

# Create a results directory with a timestamp
timestamp = datetime.now().strftime("%m-%d_%H-%M")
results_name = f"window_overlap_analysis_{timestamp}"
results_dir = os.path.join("results", results_name)
os.makedirs(results_dir, exist_ok=True)

#Window and Overlap settings (min, max, increment)
window_durations = np.arange(3, 3.5, .1)
overlap_ratios = np.arange(0.0, 0.8, 0.1)
# window_durations = np.array([2.0])
# overlap_ratios = np.array([0.0])

# Run the analysis
results_df = analyze_window_and_overlap(mixture_path, clean1_path, clean2_path, results_dir, window_durations, overlap_ratios)

# Visualize
get_and_print_optimal_params(results_df)
visualize_results(results_df, results_dir)
