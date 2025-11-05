# -*- coding: utf-8 -*-
"""
STEP 2: Generate CWT Scalograms & Phasograms for PTB-XL (Scarpiniti et al. 2024 Style)

Features:
- 12-lead ECG (PTB-XL)
- 100 Hz sampling rate
- Morlet wavelet ('morl')
- Log-spaced scales: 0.5 Hz → 40 Hz (128 scales)
- Per-sample min-max normalization → uint8
- No filtering, no log, no outlier removal
- OpenCV resize + memory-mapped output
- Matches: https://doi.org/10.3390/s24248043

Output:
- train/val/test _scalograms.npy  (uint8, shape: N×12×224×224)
- train/val/test _phasograms.npy  (uint8, shape: N×12×224×224)
"""

import os
import pickle
import numpy as np
import pywt
import cv2
from tqdm import tqdm
from numpy.lib.format import open_memmap

# ============================================================================
# CONFIGURATION
# ============================================================================
PROCESSED_PATH = '../santosh_lab/shared/KagoziA/wavelets/xresnet_baseline/'
WAVELETS_PATH = '../santosh_lab/shared/KagoziA/wavelets/cwt/processed_wavelets_ptbxl_scarpiniti/'

SAMPLING_RATE = 100        # PTB-XL native rate
IMAGE_SIZE = 224
BATCH_SIZE = 100
WAVELET = 'morl'           
N_SCALES = 128
FMIN = 0.5                 # Hz
FMAX = 40.0                # Hz

os.makedirs(WAVELETS_PATH, exist_ok=True)

print("="*80)
print("STEP 2: CWT GENERATION (PTB-XL + Scarpiniti et al. 2024)")
print("="*80)
print(f"Sampling Rate: {SAMPLING_RATE} Hz")
print(f"Wavelet: {WAVELET}")
print(f"Scales: {N_SCALES} (log-spaced: {FMIN}–{FMAX} Hz)")
print(f"Output: {IMAGE_SIZE}×{IMAGE_SIZE} uint8")
print(f"Normalization: Per-sample min-max → [0,255]")
print(f"Output Path: {WAVELETS_PATH}")
print("="*80)

# ============================================================================
# LOG-SPACED SCALES (Frequency-Aware)
# ============================================================================
def get_log_scales(sampling_rate, wavelet, n_scales=128, fmin=0.5, fmax=40.0):
    """Convert frequency range [fmin, fmax] Hz → CWT scales (log-spaced)"""
    central_freq = pywt.central_frequency(wavelet)  # ~0.8125 for 'morl'
    freqs = np.logspace(np.log10(fmin), np.log10(fmax), n_scales)
    scales = central_freq * sampling_rate / (4 * np.pi * freqs)
    return scales.astype(np.float64)

# ============================================================================
# CWT GENERATOR
# ============================================================================
class CWTGenerator:
    def __init__(self, sampling_rate=100, image_size=224, wavelet='morl'):
        self.sampling_rate = sampling_rate
        self.image_size = image_size
        self.wavelet = wavelet

        # Log-spaced scales covering 0.5–40 Hz
        self.scales = get_log_scales(sampling_rate, wavelet, N_SCALES, FMIN, FMAX)

        print(f"\nGenerator Initialized:")
        print(f"  Wavelet: {wavelet}")
        print(f"  Scales: {len(self.scales)} (log: {FMIN}–{FMAX} Hz)")
        print(f"  Output: {image_size}×{image_size} uint8")
        print(f"  Normalization: Per-sample min-max → [0,255]")

    def compute_cwt(self, signal_1d):
        """Compute CWT with error handling"""
        try:
            coeffs, _ = pywt.cwt(signal_1d, self.scales, self.wavelet)
            return coeffs
        except Exception as e:
            print(f"\nWarning: CWT failed: {e}")
            return None

    def to_uint8(self, matrix):
        """Min-max normalize → uint8 (per image)"""
        matrix = np.nan_to_num(matrix)
        mn, mx = matrix.min(), matrix.max()
        if mx - mn > 1e-10:
            matrix = (matrix - mn) / (mx - mn)
        else:
            matrix = np.zeros_like(matrix)
        return (255 * matrix).astype(np.uint8)

    def resize(self, mat):
        """Resize to IMAGE_SIZE × IMAGE_SIZE"""
        return cv2.resize(mat, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)

    def process_12_lead_ecg(self, ecg_12_lead):
        """
        Input: ecg_12_lead shape (12, T) or (T, 12)
        Output: scalograms, phasograms (12, 224, 224) uint8
        """
        if ecg_12_lead.shape[0] != 12:
            ecg_12_lead = ecg_12_lead.T  # (T,12) → (12,T)

        scalograms = []
        phasograms = []

        for lead in ecg_12_lead:
            coeffs = self.compute_cwt(lead)
            if coeffs is None:
                z = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
                scalograms.append(z)
                phasograms.append(z)
                continue

            # Scalogram: magnitude
            Wm = np.abs(coeffs)
            Wm = self.resize(Wm)
            Wm = self.to_uint8(Wm)

            # Phasogram: phase (no unwrap – matches paper)
            Wp = np.angle(coeffs)
            Wp = self.resize(Wp)
            Wp = self.to_uint8(Wp)

            scalograms.append(Wm)
            phasograms.append(Wp)

        return np.stack(scalograms), np.stack(phasograms)

    def process_dataset_batched(self, X, out_s_path, out_p_path, batch_size=100):
        """Process full split with memory mapping"""
        n_samples = len(X)
        shape = (n_samples, 12, self.image_size, self.image_size)

        print(f"\nProcessing {n_samples} samples → shape {shape} (uint8)")

        # Create memmap files
        scalos = open_memmap(out_s_path, mode='w+', dtype='uint8', shape=shape)
        phasos = open_memmap(out_p_path, mode='w+', dtype='uint8', shape=shape)

        for start_idx in tqdm(range(0, n_samples, batch_size), desc="Batches"):
            end_idx = min(start_idx + batch_size, n_samples)
            batch = X[start_idx:end_idx]

            for i, ecg in enumerate(batch):
                s, p = self.process_12_lead_ecg(ecg)
                scalos[start_idx + i] = s
                phasos[start_idx + i] = p

            scalos.flush()
            phasos.flush()

        # Close and verify
        del scalos, phasos
        print(f"Saved:")
        print(f"  Scalograms: {out_s_path}")
        print(f"  Phasograms: {out_p_path}")

        # Quick verification
        s_check = np.load(out_s_path, mmap_mode='r')
        p_check = np.load(out_p_path, mmap_mode='r')
        print(f"  Shape: {s_check.shape}, dtype: {s_check.dtype}")
        print(f"  Value range: [{s_check.min()}, {s_check.max()}]")

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    np.random.seed(42)  # Reproducibility

    # Load metadata
    print("\n[1/4] Loading metadata...")
    meta_path = os.path.join(PROCESSED_PATH, 'metadata.pkl')
    with open(meta_path, 'rb') as f:
        metadata = pickle.load(f)

    print(f"Classes: {metadata['num_classes']} (multi-label)")
    print(f"Train: {metadata['train_size']} | Val: {metadata['val_size']} | Test: {metadata['test_size']}")
    print(f"Signal shape: {metadata['signal_shape']}")

    # Initialize generator
    print("\n[2/4] Initializing CWT Generator (PTB-XL + Scarpiniti Style)...")
    cwt_gen = CWTGenerator(
        sampling_rate=SAMPLING_RATE,
        image_size=IMAGE_SIZE,
        wavelet=WAVELET
    )

    # Process each split
    splits = ['train', 'val', 'test']
    for split in splits:
        print(f"\n[3/4] Processing {split.upper()} set...")
        input_path = os.path.join(PROCESSED_PATH, f'{split}_standardized.npy')
        X = np.load(input_path, mmap_mode='r')

        cwt_gen.process_dataset_batched(
            X=X,
            out_s_path=os.path.join(WAVELETS_PATH, f'{split}_scalograms.npy'),
            out_p_path=os.path.join(WAVELETS_PATH, f'{split}_phasograms.npy'),
            batch_size=BATCH_SIZE
        )
        del X

    print("\n" + "="*80)
    print("STEP 2 COMPLETE! (PTB-XL + Scarpiniti Pipeline)")
    print("="*80)
    print(f"All files saved to: {WAVELETS_PATH}")
    print("\nFiles created:")
    for split in splits:
        print(f"  - {split}_scalograms.npy")
        print(f"  - {split}_phasograms.npy")
    print("\nNext step: Train XResNet with BCEWithLogitsLoss (multi-label)")
    print("Tip: Stack scalogram + phasogram → 24-channel input or train separately")
    print("="*80)

if __name__ == '__main__':
    main()