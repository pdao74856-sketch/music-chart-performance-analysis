"""
step3_stratified_sampling.py
------------------------------
Member 4 - Bước 3: Stratified Sampling

Mục đích:
  Với mỗi market x quarter, lấy 40 bài theo 4 tầng rank:
    - 10 bài từ rank 1-10
    - 10 bài từ rank 11-20
    - 10 bài từ rank 21-50
    - 10 bài từ rank 51+
  Kiểm tra đủ 40 bài, không trùng lặp, rồi insert kết quả vào bảng
  song_quarter_sampling trên Supabase.

Input:
  data/processed/chart_weekly_with_quarter.csv (đã tạo ở bước 2)

Lưu ý về "1 bài có thể xuất hiện nhiều lần trong 1 quarter":
  Một bài hát có thể lên chart nhiều tuần trong cùng 1 quarter, với rank
  khác nhau mỗi tuần. Ta cần MỘT rank đại diện cho bài đó trong quarter
  đó trước khi chia tầng - ở đây dùng rank TỐT NHẤT (nhỏ nhất) mà bài
  đạt được trong quarter, vì đó phản ánh đúng "vị trí cao nhất" bài đạt
  được trong giai đoạn đó. Nếu team muốn dùng rank trung bình thay vì
  rank tốt nhất, báo lại để mình đổi logic.

Cách chạy: python step3_stratified_sampling.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

try:
    from supabase import create_client, Client
except ImportError:
    print("Cần cài: pip install supabase --break-system-packages")
    sys.exit(1)


RANDOM_SEED = 42  # cố định seed để kết quả sampling có thể tái tạo lại được
SONGS_PER_STRATUM = 10

# 4 tầng rank theo đúng spec: (tên tầng, rank_min, rank_max)
# rank_max = None nghĩa là "trở lên" (51+)
RANK_STRATA = [
    ("1-10", 1, 10),
    ("11-20", 11, 20),
    ("21-50", 21, 50),
    ("51+", 51, None),
]


# ---------------------------------------------------------------------------
# 1. Kết nối Supabase
# ---------------------------------------------------------------------------
def get_client() -> Client:
    load_dotenv()
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError("Thiếu SUPABASE_URL hoặc SUPABASE_KEY trong file .env")
    return create_client(url, key)


# ---------------------------------------------------------------------------
# 2. Chuẩn bị: mỗi (spotify_track_id, market, quarter) -> rank đại diện
# ---------------------------------------------------------------------------
def build_song_quarter_rank(chart: pd.DataFrame) -> pd.DataFrame:
    """
    Từ chart_weekly (nhiều dòng/tuần), rút gọn về 1 dòng cho mỗi
    (spotify_track_id, market, quarter), lấy rank NHỎ NHẤT (tốt nhất)
    mà bài đó đạt được trong quarter.
    """
    df = chart.dropna(subset=["quarter"]).copy()
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")

    best_rank = (
        df.groupby(["market", "quarter", "spotify_track_id"], as_index=False)["rank"]
        .min()
    )
    return best_rank


def assign_stratum(rank: float) -> str | None:
    for label, lo, hi in RANK_STRATA:
        if hi is None:
            if rank >= lo:
                return label
        else:
            if lo <= rank <= hi:
                return label
    return None  # phòng trường hợp rank <= 0 hoặc lỗi, không nên xảy ra sau khi validate


# ---------------------------------------------------------------------------
# 3. Stratified sampling cho 1 market-quarter
# ---------------------------------------------------------------------------
def sample_one_market_quarter(
    group: pd.DataFrame, market: str, quarter: str, rng: "pd.core.groupby.generic"
) -> tuple[pd.DataFrame, list[str]]:
    """
    Trả về (sampled_rows, warnings) cho 1 market-quarter.
    sampled_rows có cột: spotify_track_id, market, quarter, rank_stratum
    """
    warnings: list[str] = []
    sampled_parts = []

    for label, lo, hi in RANK_STRATA:
        if hi is None:
            stratum_songs = group[group["rank"] >= lo]
        else:
            stratum_songs = group[(group["rank"] >= lo) & (group["rank"] <= hi)]

        n_available = len(stratum_songs)
        if n_available < SONGS_PER_STRATUM:
            warnings.append(
                f"{market} {quarter} tầng {label}: chỉ có {n_available} bài "
                f"(cần {SONGS_PER_STRATUM}) -> lấy hết số có, THIẾU {SONGS_PER_STRATUM - n_available}"
            )
            picked = stratum_songs
        else:
            picked = stratum_songs.sample(n=SONGS_PER_STRATUM, random_state=RANDOM_SEED)

        picked = picked.copy()
        picked["rank_stratum"] = label
        sampled_parts.append(picked)

    result = pd.concat(sampled_parts, ignore_index=True) if sampled_parts else pd.DataFrame()
    return result, warnings


# ---------------------------------------------------------------------------
# 4. Chạy sampling cho TOÀN BỘ market x quarter
# ---------------------------------------------------------------------------
def run_full_sampling(song_quarter_rank: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    all_samples = []
    all_warnings: list[str] = []

    groups = song_quarter_rank.groupby(["market", "quarter"])
    print(f"Tổng số market-quarter cần sample: {len(groups)}")

    for (market, quarter), group in groups:
        sampled, warnings = sample_one_market_quarter(group, market, quarter, None)
        sampled["market"] = market
        sampled["quarter"] = quarter
        all_samples.append(sampled)
        all_warnings.extend(warnings)

    result = pd.concat(all_samples, ignore_index=True)

    # Tách quarter thành year + quarter_num để khớp đúng schema
    # song_quarter_sampling (year, quarter riêng biệt theo spec)
    result["year"] = result["quarter"].str.split("-").str[0].astype(int)

    # Cột "sampled" theo schema - đánh dấu True cho mọi dòng đã sample
    result["sampled"] = True

    final_cols = ["spotify_track_id", "market", "year", "quarter", "rank_stratum", "sampled"]
    result = result[final_cols]

    return result, all_warnings


# ---------------------------------------------------------------------------
# 5. Kiểm tra kết quả trước khi insert
# ---------------------------------------------------------------------------
def validate_sampling_result(result: pd.DataFrame) -> bool:
    print("\n" + "=" * 70)
    print("KIỂM TRA KẾT QUẢ SAMPLING")
    print("=" * 70)

    # Check 1: không duplicate theo đúng UNIQUE constraint của bảng
    key_cols = ["spotify_track_id", "market", "year", "quarter"]
    n_dup = result.duplicated(subset=key_cols).sum()
    print(f"[{'PASS' if n_dup == 0 else 'FAIL'}] Duplicate theo (track_id, market, year, quarter): {n_dup}")

    # Check 2: mỗi market-quarter có đúng 40 bài (trừ khi bị thiếu do warning)
    counts = result.groupby(["market", "quarter"]).size()
    n_not_40 = (counts != 40).sum()
    print(f"[{'PASS' if n_not_40 == 0 else 'CẢNH BÁO'}] Số market-quarter KHÔNG đúng 40 bài: {n_not_40}")
    if n_not_40 > 0:
        print("  Chi tiết các market-quarter không đủ 40:")
        print(counts[counts != 40].to_string())

    print(f"\nTổng số dòng sample: {len(result)}")
    print(f"Tổng số market-quarter: {result.groupby(['market', 'quarter']).ngroups}")
    print("=" * 70)

    return n_dup == 0


# ---------------------------------------------------------------------------
# 6. Insert vào Supabase (tạo bảng trước nếu chưa có - xem ghi chú SQL cuối file)
# ---------------------------------------------------------------------------
def insert_to_supabase(client: Client, result: pd.DataFrame, batch_size: int = 500) -> None:
    records = result.to_dict(orient="records")
    total = len(records)
    print(f"\nBắt đầu insert {total} bản ghi vào 'song_quarter_sampling'...")

    for start in range(0, total, batch_size):
        batch = records[start : start + batch_size]
        client.table("song_quarter_sampling").upsert(
            batch, on_conflict="spotify_track_id,market,year,quarter"
        ).execute()
        end = min(start + batch_size, total)
        print(f"  -> Đã nạp {start + 1} đến {end} ({end}/{total})")

    print(f"[+] Hoàn thành insert vào 'song_quarter_sampling'! Thành công: {total}/{total}")


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------
def main() -> None:
    input_path = Path("data/processed/chart_weekly_with_quarter.csv")
    if not input_path.exists():
        print(f"Không tìm thấy {input_path}. Chạy step2_assign_quarter.py trước.")
        sys.exit(1)

    print(f"Đang đọc {input_path}...")
    chart = pd.read_csv(input_path)
    print(f"  -> {len(chart)} dòng\n")

    print("Đang tính rank đại diện cho mỗi (bài, market, quarter)...")
    song_quarter_rank = build_song_quarter_rank(chart)
    print(f"  -> {len(song_quarter_rank)} tổ hợp (bài, market, quarter)\n")

    print("Đang chạy stratified sampling...")
    result, warnings = run_full_sampling(song_quarter_rank)
    print(f"  -> Sample được {len(result)} dòng\n")

    if warnings:
        print(f"[CẢNH BÁO] Có {len(warnings)} tầng bị thiếu bài (không đủ 10):")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("[OK] Không có tầng nào bị thiếu bài.")

    is_valid = validate_sampling_result(result)

    # Lưu ra CSV để xem lại / báo cáo, dù có insert Supabase hay không
    output_path = Path("data/processed/song_quarter_sampling.csv")
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n[+] Đã lưu kết quả sampling vào: {output_path}")

    if not is_valid:
        print("\n>>> CÓ DUPLICATE - KHÔNG insert vào Supabase. Kiểm tra lại logic trước.")
        return

    answer = input("\nBạn có muốn insert kết quả này vào Supabase ngay bây giờ không? (y/n): ")
    if answer.strip().lower() == "y":
        client = get_client()
        insert_to_supabase(client, result)
    else:
        print("Bỏ qua insert. File CSV đã sẵn sàng, chạy lại script này khi muốn insert.")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# GHI CHÚ: SQL để tạo bảng song_quarter_sampling trên Supabase (chạy 1 lần
# trong Supabase SQL Editor TRƯỚC KHI insert, nếu bảng chưa tồn tại):
#
# CREATE TABLE song_quarter_sampling (
#     id BIGSERIAL PRIMARY KEY,
#     spotify_track_id TEXT NOT NULL REFERENCES songs(spotify_track_id),
#     market TEXT NOT NULL,
#     year INTEGER NOT NULL,
#     quarter TEXT NOT NULL,
#     rank_stratum TEXT NOT NULL,
#     sampled BOOLEAN NOT NULL DEFAULT TRUE,
#     UNIQUE (spotify_track_id, market, year, quarter)
# );
# ---------------------------------------------------------------------------