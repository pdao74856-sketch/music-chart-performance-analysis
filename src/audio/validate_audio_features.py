"""
Audio features validation script for Member 2.
Verifies data integrity, value bounds, and null-rates from Supabase database.
Generates logs/audio_extraction_log.csv.
"""

import os
import io
import sys

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

def main():
    load_dotenv()
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        print("Lỗi: SUPABASE_URL hoặc SUPABASE_KEY không tồn tại trong .env")
        sys.exit(1)

    supabase = create_client(url, key)
    os.makedirs('logs', exist_ok=True)

    print("Đang truy vấn bảng audio_features từ Supabase...")
    all_rows = []
    step = 1000
    start = 0
    while True:
        res = supabase.table('audio_features').select('*').range(start, start + step - 1).execute()
        all_rows.extend(res.data)
        if len(res.data) < step:
            break
        start += step

    df = pd.DataFrame(all_rows)
    total_tracks = len(df)
    print(f"Tổng số track âm học trong bảng audio_features: {total_tracks}")

    # Danh sách các cột cần validate
    required_cols = [
        'tempo_bpm', 'beat_interval_variability', 'onset_rate',
        'rms_energy', 'rms_variability', 'dynamic_range', 'loudness_proxy',
        'spectral_centroid', 'spectral_bandwidth', 'key', 'mode',
        'key_confidence', 'chord_change_rate',
        'chroma_repetition_determinism', 'chroma_mean_pattern_duration', 'chroma_longest_pattern_duration',
        'mfcc_repetition_determinism', 'mfcc_mean_pattern_duration', 'mfcc_longest_pattern_duration'
    ]

    log_records = []
    for idx, row in df.iterrows():
        track_id = row.get('spotify_track_id', f'unknown_{idx}')
        yt_id = row.get('youtube_mapping_id')
        
        # Kiểm tra nulls
        missing = [c for c in required_cols if pd.isnull(row.get(c))]
        is_valid = len(missing) == 0

        # Kiểm tra ngưỡng hợp lệ (sanity checks)
        range_errors = []
        if is_valid:
            if not (40 <= float(row['tempo_bpm']) <= 260):
                range_errors.append('tempo_out_of_range')
            if not (0 <= float(row['chroma_repetition_determinism']) <= 1.0):
                range_errors.append('chroma_det_out_of_range')
            if not (0 <= float(row['mfcc_repetition_determinism']) <= 1.0):
                range_errors.append('mfcc_det_out_of_range')
            if not (0 <= float(row['key_confidence']) <= 1.0):
                range_errors.append('key_confidence_out_of_range')

        status = 'VALID' if (is_valid and len(range_errors) == 0) else 'INVALID'
        error_msg = ';'.join(missing + range_errors) if status == 'INVALID' else 'OK'

        log_records.append({
            'spotify_track_id': track_id,
            'youtube_mapping_id': yt_id,
            'status': status,
            'details': error_msg
        })

    log_df = pd.DataFrame(log_records)
    log_path = 'logs/audio_extraction_log.csv'
    log_df.to_csv(log_path, index=False, encoding='utf-8')
    print(f"Đã lưu log kiểm định tại: {log_path}")

    valid_count = (log_df['status'] == 'VALID').sum()
    invalid_count = (log_df['status'] == 'INVALID').sum()
    print(f"Kết quả kiểm định:")
    print(f" - Hợp lệ (Valid): {valid_count} / {total_tracks} ({(valid_count / total_tracks)*100:.1f}%)")
    print(f" - Lỗi (Invalid): {invalid_count} / {total_tracks} ({(invalid_count / total_tracks)*100:.1f}%)")

if __name__ == '__main__':
    main()
