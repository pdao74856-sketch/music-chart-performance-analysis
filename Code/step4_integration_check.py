"""
step4_integration_check.py
-----------------------------
Member 4 - Data Integration Check

Mục đích:
  Kiểm tra dữ liệu sampling của CHÍNH BẠN (song_quarter_sampling) có
  liên kết đúng với songs không, và (nếu Member 2/3 đã có tiến độ)
  kiểm tra luôn việc liên kết với youtube_mapping.

  Chạy được NGAY BÂY GIỜ (chỉ cần songs + song_quarter_sampling).
  Phần check youtube_mapping sẽ tự động bỏ qua nếu bảng đó chưa tồn tại
  hoặc còn trống - không cần chờ Member 2 xong mới chạy được script này.

Cách chạy: python step4_integration_check.py
"""

from __future__ import annotations

import os
import sys

import pandas as pd
from dotenv import load_dotenv

try:
    from supabase import create_client, Client
except ImportError:
    print("Cần cài: pip install supabase --break-system-packages")
    sys.exit(1)


def get_client() -> Client:
    load_dotenv()
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError("Thiếu SUPABASE_URL hoặc SUPABASE_KEY trong file .env")
    return create_client(url, key)


def fetch_table_paginated(client: Client, table: str, page_size: int = 1000) -> pd.DataFrame:
    all_rows = []
    start = 0
    while True:
        resp = client.table(table).select("*").range(start, start + page_size - 1).execute()
        rows = resp.data
        if not rows:
            break
        all_rows.extend(rows)
        if len(rows) < page_size:
            break
        start += page_size
    return pd.DataFrame(all_rows)


def try_fetch_table(client: Client, table: str) -> pd.DataFrame | None:
    """Trả về None nếu bảng chưa tồn tại, thay vì crash cả script."""
    try:
        return fetch_table_paginated(client, table)
    except Exception as e:
        print(f"[BỎ QUA] Không đọc được bảng '{table}': {e}")
        return None


def main() -> None:
    client = get_client()

    print("Đang tải 'songs'...")
    songs = fetch_table_paginated(client, "songs")
    print(f"  -> {len(songs)} dòng")

    print("Đang tải 'song_quarter_sampling'...")
    sampling = fetch_table_paginated(client, "song_quarter_sampling")
    print(f"  -> {len(sampling)} dòng\n")

    print("=" * 70)
    print("DATA INTEGRATION CHECK - MEMBER 4")
    print("=" * 70)

    # Check 1: mọi sampled track_id phải tồn tại trong songs
    song_ids = set(songs["spotify_track_id"])
    sampled_ids = set(sampling["spotify_track_id"].dropna())
    orphaned = sampled_ids - song_ids
    passed = len(orphaned) == 0
    print(f"\n[{'PASS' if passed else 'FAIL'}] Sampled songs tồn tại trong bảng songs")
    print(f"        {len(orphaned)} spotify_track_id trong sampling không có trong songs")
    if orphaned:
        print(f"        Ví dụ: {list(orphaned)[:10]}")

    # Check 2: không có duplicate sampling record
    key_cols = ["spotify_track_id", "market", "year", "quarter"]
    n_dup = sampling.duplicated(subset=key_cols).sum()
    print(f"\n[{'PASS' if n_dup == 0 else 'FAIL'}] Không duplicate sampling records")
    print(f"        {n_dup} dòng duplicate theo (track_id, market, year, quarter)")

    # Check 3: mỗi market-quarter đúng 40 bài
    counts = sampling.groupby(["market", "quarter"]).size()
    n_wrong = (counts != 40).sum()
    print(f"\n[{'PASS' if n_wrong == 0 else 'CẢNH BÁO'}] Mỗi market-quarter có đúng 40 bài")
    print(f"        {n_wrong}/{len(counts)} market-quarter không đúng 40 bài")

    # Check 4 (tùy chọn): sampled songs có chart record tương ứng trong quarter đó không
    # (đối chiếu ngược lại chart_weekly - đảm bảo sampling không "bịa" ra track_id lạ)
    chart = try_fetch_table(client, "chart_weekly")
    if chart is not None:
        chart_keys = set(zip(chart["spotify_track_id"], chart["market"]))
        sampling_keys = set(zip(sampling["spotify_track_id"], sampling["market"]))
        missing_in_chart = sampling_keys - chart_keys
        print(f"\n[{'PASS' if len(missing_in_chart) == 0 else 'FAIL'}] Sampled songs có chart record tương ứng")
        print(f"        {len(missing_in_chart)} cặp (track_id, market) trong sampling "
              f"không tìm thấy trong chart_weekly")

    # Check 5 (chỉ chạy nếu Member 2 đã có tiến độ): youtube_mapping liên kết được về track_id
    youtube = try_fetch_table(client, "youtube_mapping")
    if youtube is not None and len(youtube) > 0:
        yt_ids = set(youtube["spotify_track_id"].dropna())
        no_youtube_yet = sampled_ids - yt_ids
        pct_done = 100 * (1 - len(no_youtube_yet) / len(sampled_ids)) if sampled_ids else 0
        print(f"\n[INFO] Tiến độ Member 2 (YouTube mapping): {pct_done:.1f}% "
              f"({len(sampled_ids) - len(no_youtube_yet)}/{len(sampled_ids)} bài đã có mapping)")
    else:
        print("\n[INFO] Bảng 'youtube_mapping' chưa có dữ liệu - Member 2 chưa bắt đầu hoặc chưa insert.")

    # Check 6 (chỉ chạy nếu Member 3 đã có tiến độ): lyrics collection liên kết được về track_id
    # (giả định tên bảng lyrics metadata - đổi tên bảng nếu Member 3 đặt tên khác)
    lyrics = try_fetch_table(client, "lyrics_features")
    if lyrics is not None and len(lyrics) > 0:
        lyrics_ids = set(lyrics["spotify_track_id"].dropna())
        no_lyrics_yet = sampled_ids - lyrics_ids
        pct_done = 100 * (1 - len(no_lyrics_yet) / len(sampled_ids)) if sampled_ids else 0
        print(f"[INFO] Tiến độ Member 3 (Lyrics): {pct_done:.1f}% "
              f"({len(sampled_ids) - len(no_lyrics_yet)}/{len(sampled_ids)} bài đã có lyrics metadata)")
    else:
        print("[INFO] Bảng 'lyrics_features' chưa có dữ liệu - Member 3 chưa bắt đầu hoặc chưa insert.")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()