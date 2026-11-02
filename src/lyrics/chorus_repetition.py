
import argparse
import csv
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

# Thu muc chua cac file lyrics
INPUT_DIR = Path("data/raw/lyrics")

# File luu ket qua
OUTPUT_FILE = Path("data/processed/chorus_repetition_results.csv")

# Nguong tuong dong toi thieu: 80%
SIMILARITY_THRESHOLD = 0.80

# Do dai doan lyrics theo so tu
MIN_BLOCK_WORDS = 8
MAX_BLOCK_WORDS = 100

# Cho phep hai doan chenh lech so tu toi da 20%
MAX_LENGTH_DIFFERENCE_RATIO = 0.20


def normalize_lyrics(text):
    """
    Chuan hoa lyrics:
    - Chuan hoa Unicode
    - Chuyen ve chu thuong
    - Loai bo nhan [Chorus], [Verse], ...
    - Loai bo dau cau
    - Gop nhieu khoang trang va xuong dong thanh mot khoang trang

    Vi vay, lyrics co cach dong hay khong cach dong deu duoc xu ly
    theo cung mot dinh dang.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.lower()

    # Loai bo nhan cau truc bai hat trong ngoac vuong
    text = re.sub(r"\[[^\]]*\]", " ", text)

    # Loai bo dau cau, giu chu, so va ky tu Unicode
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)

    # Chuan hoa tat ca khoang trang va xuong dong
    text = re.sub(r"\s+", " ", text).strip()

    return text.split()


def load_lyrics(file_path):
    """Doc file lyrics va chuan hoa thanh danh sach tu."""
    with open(file_path, "r", encoding="utf-8-sig") as file:
        text = file.read()

    return normalize_lyrics(text)


def calculate_similarity(tokens_a, tokens_b):
    """
    Tinh do tuong dong giua hai doan lyrics.
    Gia tri nam trong khoang 0 den 1.
    """
    if not tokens_a or not tokens_b:
        return 0.0

    return SequenceMatcher(
        None,
        tokens_a,
        tokens_b,
        autojunk=False
    ).ratio()


def find_repeated_sections(tokens):
    """
    Tim cac doan lyrics lap lai voi do tuong dong >= 80%.

    Hai doan co the khac nhau ve so tu, nhung do chenh lech
    khong duoc vuot qua 20% do dai doan ngan hon.

    Cac doan duoc chon khong chong lan nhau de giam dem trung.
    """
    total_words = len(tokens)

    if total_words < MIN_BLOCK_WORDS * 2:
        return []

    candidates = []

    # Thu cac do dai doan tu ngan den dai
    max_length = min(MAX_BLOCK_WORDS, total_words // 2)

    for length_a in range(MIN_BLOCK_WORDS, max_length + 1):

        # Khong can xet doan dai hon phan con lai
        for start_a in range(0, total_words - length_a + 1):
            end_a = start_a + length_a
            block_a = tokens[start_a:end_a]

            # Doan thu hai phai nam sau va khong chong len doan dau
            min_length_b = max(
                MIN_BLOCK_WORDS,
                int(length_a * (1 - MAX_LENGTH_DIFFERENCE_RATIO))
            )

            max_length_b = min(
                MAX_BLOCK_WORDS,
                int(length_a * (1 + MAX_LENGTH_DIFFERENCE_RATIO)),
                total_words - end_a
            )

            for start_b in range(end_a, total_words - min_length_b + 1):
                for length_b in range(min_length_b, max_length_b + 1):
                    end_b = start_b + length_b

                    if end_b > total_words:
                        continue

                    block_b = tokens[start_b:end_b]

                    similarity = calculate_similarity(block_a, block_b)

                    if similarity >= SIMILARITY_THRESHOLD:
                        candidates.append({
                            "start_a": start_a,
                            "end_a": end_a,
                            "start_b": start_b,
                            "end_b": end_b,
                            "similarity": similarity,
                            "length": min(length_a, length_b)
                        })

    # Uu tien doan dai va co do tuong dong cao
    candidates.sort(
        key=lambda item: (
            item["length"],
            item["similarity"]
        ),
        reverse=True
    )

    used_words = [False] * total_words
    selected_sections = []

    for candidate in candidates:
        start_a = candidate["start_a"]
        end_a = candidate["end_a"]
        start_b = candidate["start_b"]
        end_b = candidate["end_b"]

        # Bo qua neu mot trong hai doan da duoc tinh
        if any(used_words[start_a:end_a]):
            continue

        if any(used_words[start_b:end_b]):
            continue

        selected_sections.append(candidate)

        for index in range(start_a, end_a):
            used_words[index] = True

        for index in range(start_b, end_b):
            used_words[index] = True

    return selected_sections


def calculate_chorus_repetition(tokens):
    """
    Dem so cap doan lyrics lap lai.

    Mot cap gom hai lan xuat hien:
    - Lan dau: doan goc
    - Lan sau: mot lan lap them

    Luu y: day la uoc luong cac doan lap, chua dam bao
    tat ca cac doan duoc tim thay deu la chorus.
    """
    if not tokens:
        return None, "empty_lyrics"

    if len(tokens) < MIN_BLOCK_WORDS * 2:
        return None, "insufficient_lyrics"

    repeated_sections = find_repeated_sections(tokens)

    repetition_count = len(repeated_sections)

    return repetition_count, "success"


def process_song(file_path):
    """Xu ly mot bai hat."""
    tokens = load_lyrics(file_path)

    if not tokens:
        return None, "empty_lyrics"

    return calculate_chorus_repetition(tokens)



def main():
    import time

    parser = argparse.ArgumentParser(
        description="Detect repeated lyric sections similar to choruses."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="So bai muon xu ly, vi du: --limit 20"
    )

    args = parser.parse_args()

    if not INPUT_DIR.exists():
        print(f"Khong tim thay thu muc: {INPUT_DIR}")
        return

    files = sorted(INPUT_DIR.glob("*.txt"))

    if args.limit is not None:
        files = files[:args.limit]

    if not files:
        print("Khong tim thay file lyrics nao.")
        return

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    results = []
    total = len(files)
    start_all = time.time()

    print(f"Tong so bai can xu ly: {total}")
    print("Bat dau trich xuat chorus repetition...\n")

    for index, file_path in enumerate(files, start=1):
        track_id = file_path.stem
        start_song = time.time()

        print(
            f"[{index}/{total}] "
            f"({index / total * 100:.1f}%) "
            f"Dang xu ly: {track_id}",
            flush=True
        )

        try:
            count, status = process_song(file_path)
            error = ""

        except Exception as exception:
            count = None
            status = "error"
            error = str(exception)

        elapsed_song = time.time() - start_song
        elapsed_all = time.time() - start_all
        average_time = elapsed_all / index
        remaining = average_time * (total - index)

        results.append({
            "spotify_track_id": track_id,
            "chorus_repetition_count": count,
            "lyrics_status": status,
            "error": error
        })

        print(
            f"    Ket qua: {count} ({status}) | "
            f"Thoi gian bai nay: {elapsed_song:.1f}s | "
            f"Da chay: {elapsed_all / 60:.1f} phut | "
            f"Uoc tinh con: {remaining / 60:.1f} phut",
            flush=True
        )

        # Luu checkpoint sau moi bai de han che mat ket qua neu bi dung
        with open(
            OUTPUT_FILE,
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as file:
            fieldnames = [
                "spotify_track_id",
                "chorus_repetition_count",
                "lyrics_status",
                "error"
            ]

            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    total_elapsed = time.time() - start_all

    print("\nDa xu ly xong!")
    print(f"So bai da xu ly: {len(results)}/{total}")
    print(f"Tong thoi gian: {total_elapsed / 60:.2f} phut")
    print(f"Ket qua: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()