"""
Master audio feature extraction orchestrator for Member 2.
Coordinates extraction of rhythm, energy, spectral, tonal, and repetition (RQA) features.
"""

import os
import sys
import subprocess

def run_step(description, script_name):
    print(f"\n=======================================================")
    print(f"BƯỚC: {description} ({script_name})")
    print(f"=======================================================")
    cmd = [sys.executable, os.path.join("src", "audio", script_name)]
    result = subprocess.run(cmd, check=True)
    if result.returncode != 0:
        raise RuntimeError(f"Lỗi khi thực thi {script_name}!")

def main():
    print("Khởi chạy pipeline trích xuất toàn diện đặc trưng âm học...")
    
    # 1. Trích xuất nhịp điệu (Tempo, Beat variability, Onset rate)
    run_step("Trích xuất nhịp điệu (Rhythm)", "extract_rhythm.py")
    
    # 2. Trích xuất năng lượng (RMS, Dynamic range, Loudness proxy)
    run_step("Trích xuất năng lượng (Energy)", "extract_energy.py")
    
    # 3. Trích xuất phổ tần số (Spectral centroid, bandwidth)
    run_step("Trích xuất phổ tần số (Spectral)", "extract_spectral.py")
    
    # 4. Trích xuất lặp lại cấu trúc (Recurrence Quantification Analysis - RQA)
    run_step("Trích xuất cấu trúc lặp lại RQA (Repetition)", "extract_repetition.py")
    
    # 5. Trích xuất cao độ và hòa âm (Key, Mode, Chord change rate)
    run_step("Trích xuất hòa âm và điệu thức (Tonal)", "extract_tonal.py")
    
    # 6. Hợp nhất Master Dataset
    run_step("Hợp nhất Master Audio Features Dataset", "build_audio_features_master.py")
    
    print("\nToàn bộ quy trình trích xuất âm học đã hoàn thành xuất sắc!")

if __name__ == "__main__":
    main()
