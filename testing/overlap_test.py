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
results_name = f"window_overlap_analysis_{timestamp}"
results_dir = os.path.join("results", results_name)
os.makedirs(results_dir, exist_ok=True)

def separate_audio_windows(mixture_path, clean1_path, clean2_path, window_duration_seconds, overlap_ratio=0.5):
    """
    Separate audio sources using windows of specified duration with overlap.
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
    hop_size = int(window_size * (1 - overlap_ratio))
    total_samples = mixture.shape[1]
    num_windows = ceil((total_samples - window_size) / hop_size) + 1

    results = {}
    for model_name in model_names:
        print(f"\nProcessing with {model_name} - Window duration: {window_duration_seconds:.1f}s, Overlap: {overlap_ratio:.1%}")
        model_dir = os.path.join("models", model_name)
        output_dir = os.path.join(results_dir, model_name, 
                                f"window_{window_duration_seconds:.1f}s_overlap_{overlap_ratio:.1%}")
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        model = SepformerSeparation.from_hparams(source=f"speechbrain/{model_name}", 
                                                savedir=model_dir)

        # Initialize output tensors
        separated_1 = torch.zeros(1, total_samples, device=mixture.device)
        separated_2 = torch.zeros(1, total_samples, device=mixture.device)
        normalization = torch.zeros(1, total_samples, device=mixture.device)
        
        total_separation_time = 0
        window_times = []
        
        # Create window function
        window_function = torch.ones(window_size).unsqueeze(0)
        
        for i in range(num_windows):
            start_sample = i * hop_size
            end_sample = min(start_sample + window_size, total_samples)
            current_window_size = end_sample - start_sample
            
            mixture_window = mixture[:, start_sample:end_sample]
            
            if current_window_size < window_size:
                window = window_function[:, :current_window_size]
            else:
                window = window_function
            
            mixture_window = mixture_window * window
            
            temp_mixture_path = f"temp_mixture_window_{i}.wav"
            torchaudio.save(temp_mixture_path, mixture_window, sr1)
            
            start_time = time.time()
            est_sources = model.separate_file(path=temp_mixture_path)
            window_time = time.time() - start_time
            window_times.append(window_time)
            total_separation_time += window_time
            
            window_expanded = window.unsqueeze(-1)
            est_sources = est_sources * window_expanded
            
            if current_window_size < window_size:
                separated_1[:, start_sample:end_sample] += est_sources[:, :current_window_size, 0].detach().cpu()
                separated_2[:, start_sample:end_sample] += est_sources[:, :current_window_size, 1].detach().cpu()
                normalization[:, start_sample:end_sample] += window[:, :current_window_size]
            else:
                separated_1[:, start_sample:end_sample] += est_sources[:, :, 0].detach().cpu()
                separated_2[:, start_sample:end_sample] += est_sources[:, :, 1].detach().cpu()
                normalization[:, start_sample:end_sample] += window
            
            os.remove(temp_mixture_path)
            print(f"Window {i+1}/{num_windows} processed in {window_time:.2f} seconds")

        eps = 1e-10
        separated_1 = separated_1 / (normalization + eps)
        separated_2 = separated_2 / (normalization + eps)

        torchaudio.save(os.path.join(output_dir, "separated_source_1.wav"), 
                       separated_1, sr1)
        torchaudio.save(os.path.join(output_dir, "separated_source_2.wav"), 
                       separated_2, sr1)

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

def analyze_window_and_overlap(mixture_path, clean1_path, clean2_path):
    """
    Analyze performance across different window durations and overlap ratios.
    """
    all_results = []
    window_durations = np.arange(1, 5, 2)
    overlap_ratios = np.arange(0.0, 0.8, 0.3)
    
    total_combinations = len(window_durations) * len(overlap_ratios)
    current_combination = 0
    
    for duration in window_durations:
        for overlap in overlap_ratios:
            current_combination += 1
            print(f"\nProcessing combination {current_combination}/{total_combinations}")
            print(f"Window duration: {duration:.1f}s, Overlap: {overlap:.1%}")
            
            try:
                results = separate_audio_windows(mixture_path, clean1_path, clean2_path, 
                                              duration, overlap)
                
                for model_name, model_results in results.items():
                    result_entry = {
                        'model': model_name,
                        **model_results
                    }
                    all_results.append(result_entry)
                
                # Save intermediate results
                df = pd.DataFrame(all_results)
                df.to_csv(os.path.join(results_dir, 'window_overlap_analysis_results.csv'), 
                         index=False)
                
                with open(os.path.join(results_dir, 'window_overlap_analysis_results.json'), 'w') as f:
                    json.dump(all_results, f, indent=4, default=lambda x: int(x) if isinstance(x, (np.integer, np.int64)) else float(x) if isinstance(x, np.floating) else x)

                    
            except Exception as e:
                print(f"Error processing combination: {e}")
                continue
    
    return pd.DataFrame(all_results)

def visualize_results(df):
    """
    Create comprehensive visualizations of the window and overlap analysis results.
    """
    # Set the style using seaborn's set_style instead of plt.style.use
    sns.set_style("whitegrid")
    
    # Define metrics and their display names
    metrics = ['sdr', 'sir', 'sar', 'total_separation_time']
    metric_names = {'sdr': 'SDR (dB)', 'sir': 'SIR (dB)', 'sar': 'SAR (dB)', 
                   'total_separation_time': 'Processing Time (s)'}
    
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        
        # Create figure with subplots for each metric
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(2, 2, figure=fig)
        fig.suptitle(f'Performance Metrics for {model}', size=16, y=0.95)
        
        for idx, metric in enumerate(metrics):
            ax = fig.add_subplot(gs[idx // 2, idx % 2])
            
            # Pivot data for heatmap
            heatmap_data = model_data.pivot(index='window_duration', 
                                          columns='overlap_ratio',
                                          values=metric)
            
            # Create heatmap with improved aesthetics
            sns.heatmap(heatmap_data, 
                       annot=True, 
                       fmt='.2f', 
                       cmap='viridis' if metric == 'total_separation_time' else 'RdYlBu',
                       ax=ax,
                       cbar_kws={'label': metric_names[metric]})
            
            ax.set_title(f'{metric_names[metric]} vs Window Duration and Overlap')
            ax.set_xlabel('Overlap Ratio')
            ax.set_ylabel('Window Duration (s)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'heatmap_analysis_{model}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create 3D surface plots with improved aesthetics
        for metric in metrics:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            X = model_data['window_duration'].values
            Y = model_data['overlap_ratio'].values
            Z = model_data[metric].values
            
            # Create triangulated surface plot
            surf = ax.plot_trisurf(X, Y, Z, 
                                 cmap='viridis',
                                 edgecolor='none',
                                 alpha=0.8)
            
            ax.set_xlabel('Window Duration (s)')
            ax.set_ylabel('Overlap Ratio')
            ax.set_zlabel(metric_names[metric])
            ax.set_title(f'{metric_names[metric]} Surface Plot - {model}')
            
            # Add colorbar with label
            cbar = fig.colorbar(surf, ax=ax, pad=0.1)
            cbar.set_label(metric_names[metric])
            
            # Improve 3D plot viewing angle
            ax.view_init(elev=30, azim=45)
            
            plt.savefig(os.path.join(results_dir, f'surface_plot_{metric}_{model}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()

def find_optimal_parameters(df):
    """
    Find optimal parameters based on different criteria.
    """
    optimal_results = {}
    
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        
        optimal_results[model] = {
            'best_quality': {
                'parameters': model_data.loc[model_data['sdr'].idxmax()],
                'criterion': 'Highest SDR'
            },
            'best_efficiency': {
                'parameters': model_data.loc[model_data['sdr'].div(model_data['total_separation_time']).idxmax()],
                'criterion': 'Best SDR/Time ratio'
            },
            'fastest': {
                'parameters': model_data.loc[model_data['total_separation_time'].idxmin()],
                'criterion': 'Fastest processing time'
            }
        }
    
    return optimal_results

def analyze_and_visualize_windows(mixture_path, clean1_path, clean2_path):
    """
    Run the complete analysis and create visualizations.
    """
    # Run the analysis
    results_df = analyze_window_and_overlap(mixture_path, clean1_path, clean2_path)
    
    # Create visualizations
    visualize_results(results_df)
    
    # Find optimal parameters
    optimal_params = find_optimal_parameters(results_df)
    
    # Print summary and recommendations
    print("\nAnalysis Summary:")
    for model, results in optimal_params.items():
        print(f"\nModel: {model}")
        for criterion, result in results.items():
            params = result['parameters']
            print(f"\n{result['criterion']}:")
            print(f"Window Duration: {params['window_duration']:.1f}s")
            print(f"Overlap Ratio: {params['overlap_ratio']:.1%}")
            print(f"SDR: {params['sdr']:.2f} dB")
            print(f"Processing Time: {params['total_separation_time']:.2f}s")
    
    return results_df, optimal_params

if __name__ == "__main__":
    mixture_path = r"C:\Users\samco\WPI\MQP\separation_test\audio\mixture_input.wav"
    clean1_path = r"C:\Users\samco\WPI\MQP\separation_test\audio\clean1.wav"
    clean2_path = r"C:\Users\samco\WPI\MQP\separation_test\audio\clean2.wav"
    
    results_df, optimal_params = analyze_and_visualize_windows(mixture_path, clean1_path, clean2_path)