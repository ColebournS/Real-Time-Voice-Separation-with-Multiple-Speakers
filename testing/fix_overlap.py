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

warnings.simplefilter(action='ignore', category=FutureWarning)

# Create results directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_name = f"fix_overlap_analysis_{timestamp}"
results_dir = os.path.join("results", results_name)
os.makedirs(results_dir, exist_ok=True)

import torch

def generate_window_function(window_type, window_size):
    """
    Generates a window function of a specified type and size.
    """
    if window_type == "hann":
        return torch.hann_window(window_size)
    elif window_type == "hamming":
        return torch.hamming_window(window_size)
    elif window_type == "rectangular":
        return torch.ones(window_size)
    elif window_type == "triangular":
        return torch.bartlett_window(window_size)
    elif window_type == "blackman":
        return torch.blackman_window(window_size)
    elif window_type == "flat-top":
        return flat_top_window(window_size)
    elif window_type == "gaussian":
        return gaussian_window(window_size, std=0.4)
    else:
        raise ValueError(f"Unknown window type: {window_type}")

def gaussian_window(window_size, std=0.4):
    """
    Generates a Gaussian window.
    """
    n = torch.arange(0, window_size) - (window_size - 1) / 2
    gauss_win = torch.exp(-0.5 * (n / (std * (window_size / 2)))**2)
    return gauss_win / gauss_win.max()

def flat_top_window(window_size):
    """
    Generates a flat-top window.
    """
    a0, a1, a2, a3, a4 = 1.0, 1.93, 1.29, 0.388, 0.028
    n = torch.arange(window_size)
    term1 = a0 - a1 * torch.cos(2 * torch.pi * n / (window_size - 1))
    term2 = a2 * torch.cos(4 * torch.pi * n / (window_size - 1))
    term3 = a3 * torch.cos(6 * torch.pi * n / (window_size - 1))
    term4 = a4 * torch.cos(8 * torch.pi * n / (window_size - 1))
    return term1 + term2 - term3 + term4


def separate_audio_windows(mixture_path, clean1_path, clean2_path, window_duration_seconds, overlap_ratio=0.5):
    """
    Separate audio sources using windows of specified duration with overlap, testing multiple window types.
    """
    model_name = "resepformer-wsj02mix"
    window_types = ["hann", "hamming", "rectangular", "triangular", "blackman", "flat-top", "gaussian"]
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
    for window_type in window_types:
        print(f"\nProcessing with {model_name}_{window_type} - Window duration: {window_duration_seconds:.1f}s, Overlap: {overlap_ratio:.1%}")
        
        model_dir = os.path.join("models", model_name)
        output_dir = os.path.join(results_dir, model_name, 
                                  f"{window_type}_window_{window_duration_seconds:.1f}s_overlap_{overlap_ratio:.1%}")
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        model = SepformerSeparation.from_hparams(source=f"speechbrain/{model_name}", savedir=model_dir)

        # Initialize output tensors
        separated_1 = torch.zeros(1, total_samples, device=mixture.device)
        separated_2 = torch.zeros(1, total_samples, device=mixture.device)
        normalization = torch.zeros(1, total_samples, device=mixture.device)
        
        total_separation_time = 0
        window_times = []
        
        # Create window function
        window_function = generate_window_function(window_type, window_size).unsqueeze(0)
        
        for i in range(num_windows):
            start_sample = i * hop_size
            end_sample = min(start_sample + window_size, total_samples)
            current_window_size = end_sample - start_sample
            
            mixture_window = mixture[:, start_sample:end_sample]
            
            # Apply window function
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

        torchaudio.save(os.path.join(output_dir, "separated_source_1.wav"), separated_1, sr1)
        torchaudio.save(os.path.join(output_dir, "separated_source_2.wav"), separated_2, sr1)

        clean_reference = np.stack([clean1.squeeze().numpy(), clean2.squeeze().numpy()])
        separated_sources = np.stack([separated_1.squeeze().numpy(), separated_2.squeeze().numpy()])
        
        sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(clean_reference, separated_sources)
        
        results[f"{model_name}_{window_type}"] = {
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

    print_results(results)
    return results

def print_results(results):
    """
    Prints results in a formatted manner for each model and window type.
    """
    print("\n======= Results Summary =======")
    for model_type, metrics in results.items():
        print(f"\nModel: {model_type}")
        print(f" - Window Duration: {metrics['window_duration']} seconds")
        print(f" - Overlap Ratio: {metrics['overlap_ratio'] * 100:.1f}%")
        print(f" - Number of Windows: {metrics['num_windows']}")
        print(f" - Total Separation Time: {metrics['total_separation_time']:.2f} seconds")
        print(f" - Average Window Processing Time: {metrics['average_window_time']:.2f} seconds")
        print(f" - Std Dev of Window Processing Time: {metrics['std_window_time']:.2f}")
        print(f" - SDR (Signal-to-Distortion Ratio): {metrics['sdr']:.2f} dB")
        print(f" - SIR (Signal-to-Interference Ratio): {metrics['sir']:.2f} dB")
        print(f" - SAR (Signal-to-Artifacts Ratio): {metrics['sar']:.2f} dB")
    print("======= End of Results =======\n")

if __name__ == "__main__":
    mixture_path = r"C:\Users\samco\WPI\MQP\separation_test\audio\mixture_input.wav"
    clean1_path = r"C:\Users\samco\WPI\MQP\separation_test\audio\clean1.wav"
    clean2_path = r"C:\Users\samco\WPI\MQP\separation_test\audio\clean2.wav"
    
    results = separate_audio_windows(mixture_path, clean1_path, clean2_path, window_duration_seconds=2.7, overlap_ratio=0.5)
