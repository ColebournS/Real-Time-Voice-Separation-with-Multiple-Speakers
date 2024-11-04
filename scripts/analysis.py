import os
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from separation import separate_audio_windows

import os
import pandas as pd
import json
import numpy as np

def analyze_window_and_overlap(mixture_path, clean1_path, clean2_path, results_dir, window_durations, overlap_ratios):
    """
    Analyze performance across different window durations and overlap ratios.
    
    Parameters:
        mixture_path (str): Path to the mixture audio file.
        clean1_path (str): Path to the first clean audio file.
        clean2_path (str): Path to the second clean audio file.
        results_dir (str): Directory to store results.
        window_durations (list or np.array): Array of window durations in seconds.
        overlap_ratios (list or np.array): Array of overlap ratios.
    """
    all_results = []
    total_combinations = len(window_durations) * len(overlap_ratios)
    current_combination = 0

    # Check if files and directory paths exist
    if not os.path.exists(mixture_path):
        raise FileNotFoundError(f"Mixture file not found: {mixture_path}")
    if not os.path.exists(clean1_path):
        raise FileNotFoundError(f"Clean file 1 not found: {clean1_path}")
    if not os.path.exists(clean2_path):
        raise FileNotFoundError(f"Clean file 2 not found: {clean2_path}")
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
        print(f"Created results directory at: {results_dir}")

    for duration in window_durations:
        for overlap in overlap_ratios:
            current_combination += 1
            print(f"\nProcessing combination {current_combination}/{total_combinations}")
            print(f"Window duration: {duration:.1f}s, Overlap: {overlap:.1%}")

            try:
                # Attempt to run separation with the provided parameters
                results = separate_audio_windows(mixture_path, clean1_path, clean2_path, 
                                                  duration, overlap, results_dir)

                for model_name, model_results in results.items():
                    result_entry = {
                        'model': model_name,
                        **model_results
                    }
                    all_results.append(result_entry)

                # Save intermediate results to CSV and JSON
                df = pd.DataFrame(all_results)
                df.to_csv(os.path.join(results_dir, 'window_overlap_analysis_results.csv'), index=False)
                
                with open(os.path.join(results_dir, 'window_overlap_analysis_results.json'), 'w') as f:
                    json.dump(all_results, f, indent=4, default=lambda x: int(x) if isinstance(x, (np.integer, np.int64)) else float(x) if isinstance(x, np.floating) else x)

            except FileNotFoundError as fnf_error:
                print(f"File error: {fnf_error}")
            except ValueError as val_error:
                print(f"Value error: {val_error}")
            except TypeError as type_error:
                print(f"Type error: {type_error}")
            except json.JSONDecodeError as json_error:
                print(f"JSON error: {json_error}")
            except Exception as e:
                print(f"Unexpected error processing combination: {e}")
                continue

    return pd.DataFrame(all_results)

def visualize_results(df, results_dir):
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
            
            ax.plot_trisurf(X, Y, Z, cmap='viridis' if metric == 'total_separation_time' else 'RdYlBu', 
                             linewidth=0.2, antialiased=True)
            
            ax.set_title(f'3D Surface Plot of {metric_names[metric]}')
            ax.set_xlabel('Window Duration (s)')
            ax.set_ylabel('Overlap Ratio')
            ax.set_zlabel(metric_names[metric])
            plt.savefig(os.path.join(results_dir, f'surface_plot_{model}_{metric}.png'), 
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