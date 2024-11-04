import os
import warnings
import torchaudio
from speechbrain.inference import SepformerSeparation
import time
import numpy as np
import mir_eval

warnings.simplefilter(action='ignore', category=FutureWarning)

model_names = ["resepformer-wsj02mix", "sepformer-wsj02mix"]

os.makedirs("separated_files", exist_ok=True)
os.makedirs("models", exist_ok=True)

mixture_path = r"C:\Users\samco\WPI\MQP\separation_test\audio\mixture_input.wav"
clean1_path = r"C:\Users\samco\WPI\MQP\separation_test\audio\clean1.wav"
clean2_path = r"C:\Users\samco\WPI\MQP\separation_test\audio\clean2.wav"

clean1, sr1 = torchaudio.load(clean1_path)
clean2, sr2 = torchaudio.load(clean2_path)

if sr1 != sr2:
    raise ValueError("Sample rates for clean audio files do not match.")


clean_reference = np.stack([clean1.squeeze().numpy(), clean2.squeeze().numpy()])

for model_name in model_names:
    model_dir = os.path.join("models", model_name)
    output_dir = os.path.join("separated_files", model_name)
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    model = SepformerSeparation.from_hparams(source=f"speechbrain/{model_name}", savedir=model_dir)

    start_time = time.time()
    est_sources = model.separate_file(path=mixture_path)
    separation_time = time.time() - start_time
    print(f"Audio separation time for {model_name}: {separation_time:.2f} seconds")

    torchaudio.save(os.path.join(output_dir, "separated_source_1.wav"), est_sources[:, :, 0].detach().cpu(), sr1)
    torchaudio.save(os.path.join(output_dir, "separated_source_2.wav"), est_sources[:, :, 1].detach().cpu(), sr1)

    # Evaluate SDR, SAR, and SIR
    separated_sources = est_sources.squeeze(dim=0).detach().cpu().numpy().T  # Transpose to match shape
    sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(clean_reference, separated_sources)
    print(f"SDR: {np.mean(sdr):.2f}, SIR: {np.mean(sir):.2f}, SAR: {np.mean(sar):.2f}")