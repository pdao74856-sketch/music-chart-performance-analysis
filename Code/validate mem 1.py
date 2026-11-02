"""
validate_member1_data.py
--------------------------
Member 4 - Data Integration Check

Mục đích:
  Kiểm tra dữ liệu Member 1 đã nạp lên Supabase (bảng `songs` và
  `chart_weekly`) có đạt yêu cầu để Member 4 tiến hành sampling hay
  không. Đây là bước bắt buộc TRƯỚC KHI chạy stratified sampling -
  sampling trên data bẩn sẽ cho kết quả sai ngay từ gốc.

Yêu cầu:
  pip install supabase python-dotenv pandas --break-system-packages

Cách dùng:
  1. Tạo file .env cùng thư mục, nội dung:
       SUPABASE_URL=https://xxxx.supabase.co
       SUPABASE_KEY=your_anon_or_service_key
  2. Chạy: python validate_member1_data.py
  3. Đọc báo cáo in ra cuối file - phần nào FAIL thì báo lại Member 1.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

try:
    from supabase import create_client, Client
except ImportError:
    print("Cần cài: pip install supabase --break-system-packages")
    sys.exit(1)


# ---------------------------------------------------------------------------
# 0. Kết nối Supabase
# ---------------------------------------------------------------------------
def get_client() -> Client:
    load_dotenv()
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError(
            "Thiếu SUPABASE_URL hoặc SUPABASE_KEY trong file .env. "
            "Lấy 2 giá trị này ở Project Settings > API trên Supabase dashboard."
        )
    return create_client(url, key)


def fetch_table_paginated(client: Client, table: str, page_size: int = 1000) -> pd.DataFrame:
    """
    Supabase mặc định giới hạn 1000 dòng/lần query, nên phải fetch theo
    trang (pagination) để lấy hết toàn bộ bảng - đặc biệt quan trọng với
    chart_weekly vì bảng này có 161k+ dòng.
    """
    all_rows = []
    start = 0
    while True:
        resp = (
            client.table(table)
            .select("*")
            .range(start, start + page_size - 1)
            .execute()
        )
        rows = resp.data
        if not rows:
            break
        all_rows.extend(rows)
        if len(rows) < page_size:
            break
        start += page_size
    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# 1. Các hàm kiểm tra riêng lẻ - mỗi hàm trả về (passed: bool, message: str)
# ---------------------------------------------------------------------------
VALID_MARKETS = {"VN", "US", "KR"}


def check_songs_pk_unique(songs: pd.DataFrame) -> tuple[bool, str]:
    """spotify_track_id trong bảng songs phải UNIQUE (đúng constraint PK)."""
    n_total = len(songs)
    n_unique = songs["spotify_track_id"].nunique()
    passed = n_total == n_unique
    msg = f"songs.spotify_track_id: {n_total} dòng, {n_unique} giá trị unique"
    if not passed:
        n_dup = n_total - n_unique
        msg += f" -> FAIL: có {n_dup} spotify_track_id bị trùng!"
    return passed, msg


def check_songs_required_fields(songs: pd.DataFrame) -> tuple[bool, str]:
    """title, artist không được rỗng/null."""
    required = ["spotify_track_id", "title", "artist"]
    missing_counts = {col: int(songs[col].isna().sum()) for col in required if col in songs.columns}
    n_missing_total = sum(missing_counts.values())
    passed = n_missing_total == 0
    msg = f"songs - thiếu field bắt buộc: {missing_counts}"
    return passed, msg


def check_chart_market_valid(chart: pd.DataFrame) -> tuple[bool, str]:
    """Chỉ được có 3 market: VN, US, KR."""
    actual_markets = set(chart["market"].dropna().unique())
    invalid = actual_markets - VALID_MARKETS
    passed = len(invalid) == 0
    msg = f"chart_weekly.market - các giá trị thực tế: {sorted(actual_markets)}"
    if not passed:
        msg += f" -> FAIL: có market lạ ngoài VN/US/KR: {invalid}"
    return passed, msg


def check_chart_date_valid(chart: pd.DataFrame) -> tuple[bool, str]:
    """chart_date phải parse được và nằm trong khoảng 2021-01-01 .. 2026-12-31."""
    parsed = pd.to_datetime(chart["chart_date"], errors="coerce")
    n_invalid = int(parsed.isna().sum())
    in_range = parsed.between("2021-01-01", "2026-12-31")
    n_out_of_range = int((~in_range & parsed.notna()).sum())
    passed = n_invalid == 0 and n_out_of_range == 0
    msg = (f"chart_weekly.chart_date: {n_invalid} không parse được, "
           f"{n_out_of_range} nằm ngoài khoảng 2021-2026")
    return passed, msg


def check_rank_valid(chart: pd.DataFrame) -> tuple[bool, str]:
    """rank phải là số nguyên dương (thường 1-200 tùy chart Kworb)."""
    rank_numeric = pd.to_numeric(chart["rank"], errors="coerce")
    n_invalid = int(rank_numeric.isna().sum())
    n_non_positive = int((rank_numeric <= 0).sum())
    passed = n_invalid == 0 and n_non_positive == 0
    msg = f"chart_weekly.rank: {n_invalid} không phải số, {n_non_positive} <= 0"
    return passed, msg


def check_chart_duplicates(chart: pd.DataFrame) -> tuple[bool, str]:
    """Không được trùng (spotify_track_id, market, chart_date) - đúng UNIQUE constraint."""
    key_cols = ["spotify_track_id", "market", "chart_date"]
    n_dup = int(chart.duplicated(subset=key_cols).sum())
    passed = n_dup == 0
    msg = f"chart_weekly - duplicate theo (track_id, market, date): {n_dup} dòng"
    return passed, msg


def check_foreign_key_integrity(songs: pd.DataFrame, chart: pd.DataFrame) -> tuple[bool, str]:
    """Mọi spotify_track_id trong chart_weekly phải tồn tại trong songs."""
    song_ids = set(songs["spotify_track_id"])
    chart_ids = set(chart["spotify_track_id"].dropna())
    orphaned = chart_ids - song_ids
    passed = len(orphaned) == 0
    msg = f"FK integrity: {len(orphaned)} spotify_track_id trong chart_weekly không có trong songs"
    if not passed and len(orphaned) <= 10:
        msg += f" (ví dụ: {list(orphaned)[:10]})"
    return passed, msg


def check_records_per_market_week(chart: pd.DataFrame) -> tuple[bool, str]:
    """
    Cảnh báo (không fail cứng) nếu số record/market/week lệch nhau quá nhiều -
    dấu hiệu 1 market bị thu thập thiếu tuần nào đó.
    """
    counts = chart.groupby(["market", "chart_date"]).size().reset_index(name="n")
    per_market_avg = counts.groupby("market")["n"].agg(["mean", "min", "max"])
    msg = f"Số record/tuần theo market:\n{per_market_avg}"
    # Không có tiêu chí pass/fail cứng ở đây, chỉ để bạn tự nhìn và đánh giá
    return True, msg


# ---------------------------------------------------------------------------
# 2. Chạy toàn bộ và in báo cáo
# ---------------------------------------------------------------------------
def run_all_checks(songs: pd.DataFrame, chart: pd.DataFrame) -> None:
    checks = [
        ("Songs - PK unique", lambda: check_songs_pk_unique(songs)),
        ("Songs - required fields", lambda: check_songs_required_fields(songs)),
        ("Chart - market hợp lệ", lambda: check_chart_market_valid(chart)),
        ("Chart - chart_date hợp lệ", lambda: check_chart_date_valid(chart)),
        ("Chart - rank hợp lệ", lambda: check_rank_valid(chart)),
        ("Chart - duplicate record", lambda: check_chart_duplicates(chart)),
        ("Foreign key integrity (chart -> songs)", lambda: check_foreign_key_integrity(songs, chart)),
    ]

    print("=" * 70)
    print("BÁO CÁO KIỂM TRA DỮ LIỆU MEMBER 1")
    print("=" * 70)
    print(f"Tổng số songs: {len(songs)}")
    print(f"Tổng số chart_weekly records: {len(chart)}\n")

    all_passed = True
    for name, fn in checks:
        passed, msg = fn()
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"[{status}] {name}")
        print(f"        {msg}\n")

    # Check thông tin - không tính pass/fail
    _, info_msg = check_records_per_market_week(chart)
    print(f"[INFO] Phân bố record theo market/week")
    print(f"        {info_msg}\n")

    print("=" * 70)
    if all_passed:
        print(">>> TẤT CẢ CHECK BẮT BUỘC ĐỀU PASS - có thể tiến hành sampling.")
    else:
        print(">>> CÓ CHECK FAIL - báo lại Member 1 trước khi sampling, "
              "đừng sampling trên data lỗi vì kết quả sẽ sai ngay từ gốc.")
    print("=" * 70)


def main() -> None:
    client = get_client()
    print("Đang tải bảng 'songs'...")
    songs = fetch_table_paginated(client, "songs")
    print(f"  -> {len(songs)} dòng")

    print("Đang tải bảng 'chart_weekly' (có thể mất chút thời gian vì ~161k dòng)...")
    chart = fetch_table_paginated(client, "chart_weekly")
    print(f"  -> {len(chart)} dòng\n")

    run_all_checks(songs, chart)


if __name__ == "__main__":
    main()

