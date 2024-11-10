import os
import time
import torchaudio
import torch
import numpy as np
import mir_eval
from math import ceil
from speechbrain.inference import SepformerSeparation
import warnings
from dtw import dtw
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from concurrent.futures import ThreadPoolExecutor

warnings.simplefilter(action='ignore', category=FutureWarning)

import torch

def compute_dtw(x, y):
    """Compute approximate DTW using FastDTW for faster execution."""
    try:
        # Use FastDTW with Euclidean distance
        distance, _ = fastdtw(x, y, dist=euclidean)
        return distance
    except Exception as e:
        print(f"Error computing DTW: {e}")
        return float('inf')  # Return a high DTW distance in case of error

def verify_and_swap_sources(est_sources, separated1, separated2, start_index):
    # Slice once at the beginning to align lengths
    min_length = min(est_sources.size(0), separated1.size(0), separated2.size(0))
    source1 = est_sources[:min_length, :, 0].squeeze().cpu().numpy()
    source2 = est_sources[:min_length, :, 1].squeeze().cpu().numpy()
    separated1 = separated1[:min_length].squeeze().cpu().numpy()
    separated2 = separated2[:min_length].squeeze().cpu().numpy()
    
    # Perform DTW calculations in parallel
    with ThreadPoolExecutor() as executor:
        dtw_source1_separated1 = executor.submit(compute_dtw, source1, separated1)
        dtw_source1_separated2 = executor.submit(compute_dtw, source1, separated2)
        dtw_source2_separated1 = executor.submit(compute_dtw, source2, separated1)
        dtw_source2_separated2 = executor.submit(compute_dtw, source2, separated2)

        dtw_source1_separated1 = dtw_source1_separated1.result()
        dtw_source1_separated2 = dtw_source1_separated2.result()
        dtw_source2_separated1 = dtw_source2_separated1.result()
        dtw_source2_separated2 = dtw_source2_separated2.result()
    
    # Print the DTW distances
    print(f"DTW for Source 1 with separated1: {dtw_source1_separated1:.4f}")
    print(f"DTW for Source 1 with separated2: {dtw_source1_separated2:.4f}")
    print(f"DTW for Source 2 with separated1: {dtw_source2_separated1:.4f}")
    print(f"DTW for Source 2 with separated2: {dtw_source2_separated2:.4f}")
    
    # Determine if sources need to be swapped based on DTW distances
    if (dtw_source1_separated2 < dtw_source1_separated1) and (dtw_source2_separated1 < dtw_source2_separated2):
        est_sources = torch.stack([est_sources[:, :, 1], est_sources[:, :, 0]], dim=-1)
        print("Sources swapped.")
    
    return est_sources



def separate_audio_windows(mixture_path, clean1_path, clean2_path, window_duration_seconds, overlap_ratio=0.5, results_dir=None):
    model_names = ["resepformer-wsj02mix"]
    
    try:
        os.makedirs("models", exist_ok=True)

        # Load audio files
        clean1, sr1 = torchaudio.load(clean1_path)
        clean2, sr2 = torchaudio.load(clean2_path)
        mixture, sr_mix = torchaudio.load(mixture_path)

        if sr1 != sr2 or sr1 != sr_mix:
            raise ValueError("Sample rates for audio files do not match.")

    except FileNotFoundError as e:
        print(f"Audio file not found: {e}")
        return {}
    except ValueError as e:
        print(f"Audio file sampling rate mismatch: {e}")
        return {}
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

    results = {}
    for model_name in model_names:
        try:
            print(f"Processing with {model_name} - Window duration: {window_duration_seconds:.1f}s, Overlap: {overlap_ratio:.1%}")
            model_dir = os.path.join("models", model_name)
            output_dir = os.path.join(results_dir, model_name, 
                                       f"window_{window_duration_seconds:.1f}s_overlap_{overlap_ratio:.1%}")
            
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(model_dir, exist_ok=True)

            model = SepformerSeparation.from_hparams(source=f"speechbrain/{model_name}", 
                                                     savedir=model_dir)
            
            separated_1 = torch.zeros(1, total_samples, device=mixture.device)
            separated_2 = torch.zeros(1, total_samples, device=mixture.device)
            normalization = torch.zeros(1, total_samples, device=mixture.device)
            
            total_separation_time = 0
            window_times = []
            
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
                try:
                    est_sources = model.separate_file(path=temp_mixture_path)
                except Exception as e:
                    print(f"Error separating audio in window {i + 1}: {e}")
                    os.remove(temp_mixture_path)
                    continue  # Skip this window if separation fails
                
                #if the first window then we dont need to try to swap the sources
                if(i != 0):
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

                os.remove(temp_mixture_path)
                print(f"Window {i + 1}/{num_windows} processed in {window_time:.2f} seconds")

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

        except Exception as e:
            print(f"Error processing model {model_name}: {e}")

    return results
