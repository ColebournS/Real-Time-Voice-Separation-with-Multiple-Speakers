import os
import numpy as np
import pandas as pd
import json
from separation import separate_audio_windows

def analyze_window_and_overlap(device, mixture_path, clean1_path, clean2_path, model_names, results_dir, window_durations, overlap_ratios, analyze_by_chunk):
    """
    Analyze performance across different window durations and overlap ratios.
    
    Parameters:
        mixture_path (str): Path to the mixture audio file.
        clean1_path (str): Path to the first clean audio file.
        clean2_path (str): Path to the second clean audio file.
        results_dir (str): Directory to store results.
        window_durations (list or np.array): Array of window durations in seconds.
        overlap_ratios (list or np.array): Array of overlap ratios.
        analyze_by_chunk (bool): True if you want to analyze by chunk false if want to analyze full separation
        
    Returns:
        tuple: (overall_results_df, chunk_metrics_df)
            - overall_results_df: DataFrame containing overall metrics
            - chunk_metrics_df: DataFrame containing chunk-wise metrics (None if analyze_by_chunk is False)
    """
    all_results = []
    all_chunk_metrics = [] if analyze_by_chunk else None
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

            results = {}
            # Attempt to run separation with the provided parameters
            try:
                results = separate_audio_windows(device, mixture_path, clean1_path, clean2_path, model_names,
                                                duration, overlap, results_dir, analyze_by_chunk)
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

            for model_name, model_results in results.items():
                # Extract and store overall results
                overall_result = {
                    'model': model_name,
                    'window_duration': duration,
                    'overlap_ratio': overlap
                }
                
                # Add all metrics except chunk_metrics
                for key, value in model_results.items():
                    if key != 'chunk_metrics':
                        overall_result[key] = value
                
                all_results.append(overall_result)
                
                # Process chunk metrics if available
                if analyze_by_chunk and 'chunk_metrics' in model_results:
                    for chunk_idx, chunk_data in enumerate(model_results['chunk_metrics']):
                        chunk_entry = {
                            'model': model_name,
                            'window_duration': duration,
                            'overlap_ratio': overlap,
                            'chunk_index': chunk_idx,
                            'chunk_sdr': chunk_data['sdr'],
                            'chunk_sir': chunk_data['sir'],
                            'chunk_sar': chunk_data['sar'],
                            'start_sample': chunk_data['start_sample'],
                            'end_sample': chunk_data['end_sample']
                        }
                        all_chunk_metrics.append(chunk_entry)

            # Save intermediate results to JSON
            results_to_save = {
                'overall_results': all_results,
                'chunk_metrics': all_chunk_metrics if analyze_by_chunk else None
            }
            
            with open(os.path.join(results_dir, 'window_overlap_analysis_results.json'), 'w') as f:
                json.dump(results_to_save, f, indent=4, 
                        default=lambda x: int(x) if isinstance(x, (np.integer, np.int64)) 
                                        else float(x) if isinstance(x, np.floating) else x)

    # Convert results to DataFrames
    overall_df = pd.DataFrame(all_results)
    chunk_df = pd.DataFrame(all_chunk_metrics) if analyze_by_chunk else None

    return overall_df, chunk_df