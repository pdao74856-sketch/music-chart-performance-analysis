import re
import pandas as pd
import ftfy

src = "data/processed/sample_list_v1.csv"
dst = "data/processed/sample_list_v2.csv"

df = pd.read_csv(src, encoding="utf-8-sig", dtype=str, keep_default_na=False)
n_rows = len(df)

before = df[["title", "artist"]].copy()
for c in ["title", "artist"]:
    df[c] = df[c].map(ftfy.fix_text)

changed = (before != df[["title", "artist"]]).any(axis=1).sum()
print("Số dòng:", n_rows)
print("Số dòng đã sửa:", changed)

# Kiểm tra còn dòng nào nghi lỗi không
pat = re.compile(r"Ã|Ä|Â|â€|\ufffd")
bad = df[df["title"].str.contains(pat) | df["artist"].str.contains(pat)]
print("Số dòng còn nghi lỗi:", len(bad))
print(bad[["spotify_track_id", "title", "artist"]].head(20).to_string())

# Xem thử vài dòng đã sửa
print(df[df["title"].str.contains("Thu")][["title", "artist"]].head(5).to_string())

assert len(df) == n_rows
df.to_csv(dst, index=False, encoding="utf-8-sig")
print("Đã lưu:", dst)