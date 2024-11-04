import os
import warnings
import torchaudio
import torch
from speechbrain.inference import SepformerSeparation
import time
import numpy as np
import mir_eval
from math import ceil
import pandas as pd
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

warnings.simplefilter(action='ignore', category=FutureWarning)
# Create results directory

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_name = f"window_analysis_{timestamp}"
results_dir = os.path.join("results", results_name)
os.makedirs(results_dir, exist_ok=True)

def separate_audio_windows(mixture_path, clean1_path, clean2_path, window_duration_seconds, overlap_ratio=0.5):
    """
    Separate audio sources using windows of specified duration with overlap.
    
    Args:
        mixture_path (str): Path to the mixture audio file
        clean1_path (str): Path to the first clean audio file
        clean2_path (str): Path to the second clean audio file
        window_duration_seconds (float): Duration of each processing window in seconds
        overlap_ratio (float): Ratio of overlap between consecutive windows (0 to 1)
    
    Returns:
        dict: Results dictionary containing separation metrics and timing information
    """
    model_names = ["resepformer-wsj02mix"]
    
    os.makedirs("models", exist_ok=True)

    # Load audio files
    clean1, sr1 = torchaudio.load(clean1_path)
    clean2, sr2 = torchaudio.load(clean2_path)
    mixture, sr_mix = torchaudio.load(mixture_path)

    if sr1 != sr2 or sr1 != sr_mix:
        raise ValueError("Sample rates for audio files do not match.")

    # Calculate window parameters
    window_size = int(window_duration_seconds * sr1)
    hop_size = int(window_size * (1 - overlap_ratio))  # Calculate hop size based on overlap
    total_samples = mixture.shape[1]
    
    # Calculate number of windows with overlap
    num_windows = ceil((total_samples - window_size) / hop_size) + 1

    results = {}
    for model_name in model_names:
        print(f"\nProcessing with {model_name} - Window duration: {window_duration_seconds:.1f}s, Overlap: {overlap_ratio:.1%}")
        model_dir = os.path.join("models", model_name)
        output_dir = os.path.join(results_dir, model_name, f"window_{window_duration_seconds:.1f}s_overlap_{overlap_ratio:.1%}")
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        model = SepformerSeparation.from_hparams(source=f"speechbrain/{model_name}", 
                                                savedir=model_dir)

        # Initialize output tensors with zeros
        separated_1 = torch.zeros(1, total_samples, device=mixture.device)
        separated_2 = torch.zeros(1, total_samples, device=mixture.device)
        
        # Initialize normalization tensor to track overlaps
        normalization = torch.zeros(1, total_samples, device=mixture.device)
        
        total_separation_time = 0
        window_times = []
        
        # Create window function (Hanning window)
        window_function = torch.hann_window(window_size).unsqueeze(0)
        
        # Process each window
        for i in range(num_windows):
            start_sample = i * hop_size
            end_sample = min(start_sample + window_size, total_samples)
            current_window_size = end_sample - start_sample
            
            # Extract window from mixture
            mixture_window = mixture[:, start_sample:end_sample]
            
            # Apply window function (trim if needed for last window)
            if current_window_size < window_size:
                window = window_function[:, :current_window_size]
            else:
                window = window_function
            
            mixture_window = mixture_window * window
            
            # Save window to temporary file
            temp_mixture_path = f"temp_mixture_window_{i}.wav"
            torchaudio.save(temp_mixture_path, mixture_window, sr1)
            
            # Process window
            start_time = time.time()
            est_sources = model.separate_file(path=temp_mixture_path)
            window_time = time.time() - start_time
            window_times.append(window_time)
            total_separation_time += window_time
            
            # Apply window function to separated sources
            est_sources = est_sources * window.unsqueeze(1)
            
            # Add to output tensors with proper alignment
            if current_window_size < window_size:
                separated_1[:, start_sample:end_sample] += est_sources[:, :current_window_size, 0].detach().cpu()
                separated_2[:, start_sample:end_sample] += est_sources[:, :current_window_size, 1].detach().cpu()
                normalization[:, start_sample:end_sample] += window[:, :current_window_size]
            else:
                separated_1[:, start_sample:end_sample] += est_sources[:, :, 0].detach().cpu()
                separated_2[:, start_sample:end_sample] += est_sources[:, :, 1].detach().cpu()
                normalization[:, start_sample:end_sample] += window
            
            # Clean up temporary file
            os.remove(temp_mixture_path)
            
            print(f"Window {i+1}/{num_windows} processed in {window_time:.2f} seconds")

        # Normalize output by overlap-add weights
        eps = 1e-10  # Small value to prevent division by zero
        separated_1 = separated_1 / (normalization + eps)
        separated_2 = separated_2 / (normalization + eps)

        # Save full separated sources
        torchaudio.save(os.path.join(output_dir, "separated_source_1.wav"), 
                       separated_1, sr1)
        torchaudio.save(os.path.join(output_dir, "separated_source_2.wav"), 
                       separated_2, sr1)

        # Evaluate SDR, SAR, and SIR on full audio
        clean_reference = np.stack([clean1.squeeze().numpy(), clean2.squeeze().numpy()])
        separated_sources = np.stack([separated_1.squeeze().numpy(), 
                                    separated_2.squeeze().numpy()])
        
        sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(clean_reference, 
                                                              separated_sources)
        
        results[model_name] = {
            'window_duration': window_duration_seconds,
            'overlap_ratio': overlap_ratio,
            'num_windows': num_windows,
            'total_separation_time': total_separation_time,
            'average_window_time': np.mean(window_times),
            'std_window_time': np.std(window_times),
            'sdr': float(np.mean(sdr)),
            'sir': float(np.mean(sir)),
            'sar': float(np.mean(sar))
        }

    return results

def analyze_window_durations(mixture_path, clean1_path, clean2_path):
    """
    Analyze performance across different window durations.
    """
    all_results = []
    window_durations = np.arange(0.5, 3.5, 0.5)
    
    for duration in window_durations:
        print(f"\nTesting window duration: {duration:.1f} seconds")
        results = separate_audio_windows(mixture_path, clean1_path, clean2_path, duration)
        
        # Add results to list
        for model_name, model_results in results.items():
            result_entry = {
                'model': model_name,
                **model_results
            }
            all_results.append(result_entry)
            
        # Save intermediate results after each duration
        df = pd.DataFrame(all_results)
        df.to_csv(os.path.join(results_dir, 'window_analysis_results.csv'), index=False)
        
        # Save detailed results as JSON
        with open(os.path.join(results_dir, 'window_analysis_results.json'), 'w') as f:
            json.dump(all_results, f, indent=4)
    
    return df

def visualize_results(df):
    """
    Create comprehensive visualizations of the window analysis results.
    """
    # Set style parameters directly instead of using a style preset
    plt.rcParams['figure.figsize'] = (20, 12)  # Adjusted figure size for 4 plots
    plt.rcParams['axes.grid'] = True
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['lines.markersize'] = 8
    
    # Create figure with GridSpec for better control of subplot sizes
    fig = plt.figure()
    gs = GridSpec(2, 2, figure=fig)  # Changed to 2x2 grid
    
    # Color palette for consistent colors across plots
    colors = ['#FF6B6B', '#4ECDC4']
    
    # 1. Processing Time Analysis
    ax1 = fig.add_subplot(gs[0, 0])
    for idx, model in enumerate(df['model'].unique()):
        model_data = df[df['model'] == model]
        ax1.plot(model_data['window_duration'], 
                model_data['total_separation_time'], 
                marker='o', 
                label=model,
                color=colors[idx])
    ax1.set_xlabel('Window Duration (s)')
    ax1.set_ylabel('Total Processing Time (s)')
    ax1.set_title('Total Processing Time vs Window Duration')
    ax1.legend()
    
    # 2. Quality Metrics (SDR, SIR, SAR)
    ax2 = fig.add_subplot(gs[0, 1])
    for idx, model in enumerate(df['model'].unique()):
        model_data = df[df['model'] == model]
        ax2.plot(model_data['window_duration'], 
                model_data['sdr'], 
                marker='o', 
                color=colors[idx],
                label=f'{model} (SDR)')
        ax2.plot(model_data['window_duration'], 
                model_data['sir'], 
                marker='s', 
                linestyle='--',
                color=colors[idx],
                alpha=0.7,
                label=f'{model} (SIR)')
        ax2.plot(model_data['window_duration'], 
                model_data['sar'], 
                marker='^', 
                linestyle=':',
                color=colors[idx],
                alpha=0.5,
                label=f'{model} (SAR)')
    ax2.set_xlabel('Window Duration (s)')
    ax2.set_ylabel('Quality Metric (dB)')
    ax2.set_title('Separation Quality Metrics vs Window Duration')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 3. Average Window Processing Time
    ax3 = fig.add_subplot(gs[1, 0])
    for idx, model in enumerate(df['model'].unique()):
        model_data = df[df['model'] == model]
        ax3.plot(model_data['window_duration'], 
                model_data['average_window_time'], 
                marker='o', 
                label=model,
                color=colors[idx])
    ax3.set_xlabel('Window Duration (s)')
    ax3.set_ylabel('Average Time per Window (s)')
    ax3.set_title('Average Processing Time per Window vs Window Duration')
    ax3.legend()
    
    # 4. Efficiency Metric (Quality per Time)
    ax4 = fig.add_subplot(gs[1, 1])
    for idx, model in enumerate(df['model'].unique()):
        model_data = df[df['model'] == model]
        efficiency = model_data['sdr'] / model_data['total_separation_time']
        ax4.plot(model_data['window_duration'], 
                efficiency, 
                marker='o', 
                label=model,
                color=colors[idx])
    ax4.set_xlabel('Window Duration (s)')
    ax4.set_ylabel('SDR/Processing Time (dB/s)')
    ax4.set_title('Processing Efficiency vs Window Duration')
    ax4.legend()
    
    # Adjust layout and save
    plt.tight_layout()
    plots_path = os.path.join(results_dir, 'overlap_analysis_plots.png')
    plt.savefig(plots_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Create correlation heatmaps
    metrics = ['window_duration', 'total_separation_time', 'average_window_time', 
               'sdr', 'sir', 'sar', 'num_windows']
    
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        correlation = model_data[metrics].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation, 
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   fmt='.2f')
        plt.title(f'Correlation Matrix - {model}')
        plt.tight_layout()
        heatmap_path = os.path.join(results_dir, f'correlation_heatmap_{model}.png')
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()

def analyze_and_visualize_windows(mixture_path, clean1_path, clean2_path):
    """
    Run the analysis and create visualizations.
    """
    # Run the analysis
    results_df = analyze_window_durations(mixture_path, clean1_path, clean2_path)
    
    # Create visualizations
    visualize_results(results_df)
    
    # Print summary statistics
    print("\nSummary by model:")
    for model in results_df['model'].unique():
        model_data = results_df[results_df['model'] == model]
        print(f"\nModel: {model}")
        print("Best performance by metric:")
        print(f"Fastest total time: {model_data['total_separation_time'].min():.2f}s "
              f"(window: {model_data.loc[model_data['total_separation_time'].idxmin(), 'window_duration']:.1f}s)")
        print(f"Best SDR: {model_data['sdr'].max():.2f} "
              f"(window: {model_data.loc[model_data['sdr'].idxmax(), 'window_duration']:.1f}s)")
        print(f"Best SIR: {model_data['sir'].max():.2f} "
              f"(window: {model_data.loc[model_data['sir'].idxmax(), 'window_duration']:.1f}s)")
        print(f"Best SAR: {model_data['sar'].max():.2f} "
              f"(window: {model_data.loc[model_data['sar'].idxmax(), 'window_duration']:.1f}s)")
    
    return results_df

if __name__ == "__main__":
    # Set your paths here
    mixture_path = r"C:\Users\samco\WPI\MQP\separation_test\audio\mixture_input.wav"
    clean1_path = r"C:\Users\samco\WPI\MQP\separation_test\audio\clean1.wav"
    clean2_path = r"C:\Users\samco\WPI\MQP\separation_test\audio\clean2.wav"
    
    # Run analysis and create visualizations
    results_df = analyze_and_visualize_windows(mixture_path, clean1_path, clean2_path)