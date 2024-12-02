import os
import time
import torchaudio
import torch
import numpy as np
import mir_eval
from math import ceil
from speechbrain.inference import SepformerSeparation
import warnings
from pathlib import Path
import logging

logging.getLogger("speechbrain.utils.fetching").setLevel(logging.WARNING)
logging.getLogger('speechbrain').setLevel(logging.WARNING)
warnings.simplefilter(action='ignore', category=FutureWarning)

def compute_similarity_measures(x, y, start_index):
    try:
        min_length = min(x.size(0), y.size(0))

        # Slice the larger tensor only to keep the desired segment
        if x.size(0) > y.size(0):
            #print(f"\t\t\tSlicing x from index {start_index} to {start_index + min_length}")
            x = x[start_index:start_index + min_length]
        else:
            #print(f"\t\t\tSlicing y from index {start_index} to {start_index + min_length}")
            y = y[start_index:start_index + min_length]

        # Center both tensors for correlation computation
        x_mean = x.mean()
        y_mean = y.mean()
        
        x_centered = x - x_mean
        y_centered = y - y_mean
        
        # Compute normalized cross-correlation
        cross_corr = torch.sum(x_centered * y_centered) / (torch.sqrt(torch.sum(x_centered**2) * torch.sum(y_centered**2)) + 1e-10)

        # Compute Euclidean Distance
        euclidean_distance = torch.norm(x - y)

        # Compute Cosine Similarity
        cosine_similarity = torch.nn.functional.cosine_similarity(x.view(1, -1), y.view(1, -1))

        # Compute Mean Squared Error
        mse = torch.mean((x - y) ** 2)

        return {
            'cross_correlation': cross_corr,
            'euclidean_distance': euclidean_distance,
            'cosine_similarity': cosine_similarity,
            'mean_squared_error': mse
        }

    except Exception as e:
        print(f"Error computing similarity measures: {e}")
        return {
            'cross_correlation': torch.tensor(0.0),
            'euclidean_distance': torch.tensor(float('inf')),
            'cosine_similarity': torch.tensor(0.0),
            'mean_squared_error': torch.tensor(float('inf'))
        }


def verify_and_swap_sources(est_sources, separated1, separated2, start_index):
    """
    Verify that the estimated sources match the original clean sources in terms of similarity.
    
    Args:
        est_sources (torch.Tensor): Estimated separated sources
        separated1 (torch.Tensor): Separated first source so far
        separated2 (torch.Tensor): Separated second source so far
    
    Returns:
        torch.Tensor: Potentially reordered estimated sources
    """
    
    # Extract the estimated sources
    source1 = est_sources[:, :, 0]
    source2 = est_sources[:, :, 1]
    
    # Compute similarity measures with clean sources
    sim1_source1 = compute_similarity_measures(source1.squeeze(), separated1.squeeze(), start_index)
    sim1_source2 = compute_similarity_measures(source1.squeeze(), separated2.squeeze(), start_index)
    sim2_source1 = compute_similarity_measures(source2.squeeze(), separated1.squeeze(), start_index)
    sim2_source2 = compute_similarity_measures(source2.squeeze(), separated2.squeeze(), start_index)
    
    # Print the similarity measures
    # print(f"Similarity measures for Source 1 with separated1: Cross-correlation = {sim1_source1['cross_correlation'].item():.4f}, "
    #       f"Euclidean Distance = {sim1_source1['euclidean_distance'].item():.4f}, "
    #       f"Cosine Similarity = {sim1_source1['cosine_similarity'].item():.4f}, "
    #       f"MSE = {sim1_source1['mean_squared_error'].item():.4f}")
    
    # print(f"Similarity measures for Source 1 with separated2: Cross-correlation = {sim1_source2['cross_correlation'].item():.4f}, "
    #       f"Euclidean Distance = {sim1_source2['euclidean_distance'].item():.4f}, "
    #       f"Cosine Similarity = {sim1_source2['cosine_similarity'].item():.4f}, "
    #       f"MSE = {sim1_source2['mean_squared_error'].item():.4f}")

    # print(f"Similarity measures for Source 2 with separated1: Cross-correlation = {sim2_source1['cross_correlation'].item():.4f}, "
    #       f"Euclidean Distance = {sim2_source1['euclidean_distance'].item():.4f}, "
    #       f"Cosine Similarity = {sim2_source1['cosine_similarity'].item():.4f}, "
    #       f"MSE = {sim2_source1['mean_squared_error'].item():.4f}")
    
    # print(f"Similarity measures for Source 2 with separated2: Cross-correlation = {sim2_source2['cross_correlation'].item():.4f}, "
    #       f"Euclidean Distance = {sim2_source2['euclidean_distance'].item():.4f}, "
    #       f"Cosine Similarity = {sim2_source2['cosine_similarity'].item():.4f}, "
    #       f"MSE = {sim2_source2['mean_squared_error'].item():.4f}")
    
    # Initialize scores
    score_source1 = 0
    score_source2 = 0

    # Increment scores based on better similarity metrics
    # For Source 1 with separated1
    if (sim1_source2['cross_correlation'] > sim1_source1['cross_correlation']):
        score_source1 += 1
    if (sim1_source2['euclidean_distance'] < sim1_source1['euclidean_distance']):
        score_source1 += 1
    if (sim1_source2['cosine_similarity'] > sim1_source1['cosine_similarity']):
        score_source1 += 1
    if (sim1_source2['mean_squared_error'] < sim1_source1['mean_squared_error']):
        score_source1 += 1

    # For Source 2 with separated2
    if (sim2_source1['cross_correlation'] > sim2_source2['cross_correlation']):
        score_source2 += 1
    if (sim2_source1['euclidean_distance'] < sim2_source2['euclidean_distance']):
        score_source2 += 1
    if (sim2_source1['cosine_similarity'] > sim2_source2['cosine_similarity']):
        score_source2 += 1
    if (sim2_source1['mean_squared_error'] < sim2_source2['mean_squared_error']):
        score_source2 += 1
    
    # Determine if sources need to be swapped based on scores
    if score_source1 > score_source2:
        est_sources = torch.stack([est_sources[:, :, 1], est_sources[:, :, 0]], dim=-1)
    
    return est_sources


def separate_audio_windows(device, mixture_path, clean1_path, clean2_path, model_names, window_duration_seconds, overlap_ratio, results_dir, analyze_by_chunk, save_separated_files):
    results = {}
    
    # print(mixture_path)
    # print(clean1_path)
    # print(clean2_path)

    try:
        os.makedirs("models", exist_ok=True)

        # Load audio files
        clean1, sr1 = torchaudio.load(str(clean1_path))
        clean2, sr2 = torchaudio.load(str(clean2_path))
        mixture, sr_mix = torchaudio.load(str(mixture_path))

        if sr1 != sr2 or sr1 != sr_mix:
            raise ValueError("Sample rates for audio files do not match.")
    except Exception as e:
        print(f"Error loading audio files: {e}")
        return {}

    # Calculate window parameters
    try:
        window_size = int(window_duration_seconds * sr1)
        hop_size = int(window_size * (1 - overlap_ratio))
        total_samples = mixture.shape[1]
        num_windows = ceil((total_samples - window_size) / hop_size) + 1
    except Exception as e:
        print(f"Error calculating window parameters: {e}")
        return {}

    for model_name in model_names:
        try:
            print(f"Processing with {model_name} - Window duration: {window_duration_seconds:.1f}s, Overlap: {overlap_ratio:.1%}")
            model_dir = os.path.join("models", model_name)
            output_dir = os.path.join(results_dir, model_name, f"window_{window_duration_seconds:.1f}s_overlap_{overlap_ratio:.1%}")
            
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(model_dir, exist_ok=True)

            model = SepformerSeparation.from_hparams(source=f"speechbrain/{model_name}", savedir=model_dir)

            separated_1 = torch.zeros(1, total_samples, device=mixture.device)
            separated_2 = torch.zeros(1, total_samples, device=mixture.device)
            normalization = torch.zeros(1, total_samples, device=mixture.device)
            
            total_separation_time = 0
            window_times = []
            
            window_function = torch.ones(window_size).unsqueeze(0)
            
            # Lists to store per-chunk metrics if analyze_by_chunk is True
            chunk_metrics = [] if analyze_by_chunk else None
            
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

                temp_mixture_path = Path("temp_mixture_window_.wav")
                temp_mixture_path_str = "temp_mixture_window_.wav"

                try:
                    # Save the temporary mixture window file
                    torchaudio.save(temp_mixture_path_str, mixture_window, sr1)

                    if not temp_mixture_path.exists():
                        raise FileNotFoundError(f"Temporary file {temp_mixture_path_str} was not created.")

                    start_time = time.time()
                    
                    # Run the separation on the current window
                    try:
                            est_sources = model.separate_file(path=temp_mixture_path_str)
                    except Exception as e:
                        print(f"Error separating audio in window {i + 1}: {e}")

                except FileNotFoundError as fnf_error:
                    print(f"File error: {fnf_error}")
                except PermissionError as perm_error:
                    print(f"Permission error: {perm_error}")
                except Exception as e:
                    print(f"Unexpected error in window {i + 1}: {e}")

                if i != 0:
                    prev_window_start = max(0, start_sample - current_window_size)
                    est_sources = verify_and_swap_sources(est_sources, separated_1, separated_2, prev_window_start)
                
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

                # Calculate metrics for this chunk if analyze_by_chunk is True
                if analyze_by_chunk:
                    clean1_chunk = clean1[:, start_sample:end_sample]
                    clean2_chunk = clean2[:, start_sample:end_sample]
                    clean_reference_chunk = np.stack([clean1_chunk.squeeze().numpy(), 
                                                    clean2_chunk.squeeze().numpy()])
                    
                    # Use the current window's separated sources
                    if current_window_size < window_size:
                        sep1_chunk = est_sources[:, :current_window_size, 0].detach().cpu()
                        sep2_chunk = est_sources[:, :current_window_size, 1].detach().cpu()
                    else:
                        sep1_chunk = est_sources[:, :, 0].detach().cpu()
                        sep2_chunk = est_sources[:, :, 1].detach().cpu()
                    
                    separated_sources_chunk = np.stack([sep1_chunk.squeeze().numpy(), 
                                                      sep2_chunk.squeeze().numpy()])
                    
                    swapped_separated_sources_chunk = np.stack([sep2_chunk.squeeze().numpy(), 
                                                      sep1_chunk.squeeze().numpy()])
                  
                    separated_metrics = mir_eval.separation.bss_eval_sources(clean_reference_chunk, separated_sources_chunk)
                    swapped_metrics = mir_eval.separation.bss_eval_sources(clean_reference_chunk, swapped_separated_sources_chunk)

                    chunk_sdr = max(np.max(separated_metrics[0]), np.max(swapped_metrics[0]))  
                    chunk_sir = max(np.max(separated_metrics[1]), np.max(swapped_metrics[1]))  
                    chunk_sar = max(np.max(separated_metrics[2]), np.max(swapped_metrics[2]))

                    chunk_sdr = 0 if np.isnan(chunk_sdr) else chunk_sdr
                    chunk_sir = 0 if np.isnan(chunk_sir) else chunk_sir
                    chunk_sar = 0 if np.isnan(chunk_sar) else chunk_sar

                    chunk_metrics.append({
                        'window_idx': i,
                        'start_sample': start_sample,
                        'end_sample': end_sample,
                        'sdr': float(np.mean(chunk_sdr)),
                        'sir': float(np.mean(chunk_sir)),
                        'sar': float(np.mean(chunk_sar))
                    })

                os.remove(temp_mixture_path)
                #print(f"Window {i + 1}/{num_windows} processed in {window_time:.2f} seconds")

            eps = 1e-10
            separated_1 = separated_1 / (normalization + eps)
            separated_2 = separated_2 / (normalization + eps)

            if save_separated_files:
                mixture_name = Path(mixture_path).stem
                output_subdir = os.path.join(output_dir, mixture_name)
                os.makedirs(output_subdir, exist_ok=True)

                # Save audio files
                torchaudio.save(os.path.join(output_subdir, "separated_source_1.wav"), separated_1, sr1)
                torchaudio.save(os.path.join(output_subdir, "separated_source_2.wav"), separated_2, sr1)

            # Calculate overall metrics
            clean_reference = np.stack([clean1.squeeze().numpy(), clean2.squeeze().numpy()])
            separated_sources = np.stack([separated_1.squeeze().numpy(), separated_2.squeeze().numpy()])

            sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(clean_reference, separated_sources)

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

            # Add chunk metrics if analyze_by_chunk is True
            if analyze_by_chunk:
                results[model_name]['chunk_metrics'] = chunk_metrics
                results[model_name]['chunk_average_metrics'] = {
                    'sdr': float(np.mean([m['sdr'] for m in chunk_metrics])),
                    'sir': float(np.mean([m['sir'] for m in chunk_metrics])),
                    'sar': float(np.mean([m['sar'] for m in chunk_metrics]))
                }

        except Exception as e:
            print(f"Error processing model {model_name}: {e}")

    return results
