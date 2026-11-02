#!/usr/bin/env python3
"""Language detection pipeline (v2.1.0) for raw lyrics files.

Key principles:
- Deterministic seeding (DetectorFactory.seed = 0).
- Line-by-line script + langdetect weighting (threshold = 0.70 for main language,
  >= 2 languages with >= 0.30 for 'mixed').
- Evidence-based song-level script fallback ONLY when all individual lines are
  shorter than MIN_TEXT_LENGTH (15 chars), which occurs in syllabic/logographic
  scripts (Hangul 'ko', Hanzi 'zh', Kana 'ja') where 1 character = 1 syllable
  and lines are typically 8-14 characters long.
- Never artificially reduces 'mixed' labels (fixes the working-tree denominator
  bug where script_counts_total excluded Latin characters) and never assigns
  'unknown' to a specific language without >= 70% script character evidence.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter
from typing import Any

try:
    from langdetect import DetectorFactory, detect
    from langdetect.lang_detect_exception import LangDetectException

    DetectorFactory.seed = 0
except ImportError:
    print("Chưa cài langdetect. Chạy: pip install langdetect")
    raise

ALGORITHM_VERSION = "2.1.0"

RESULT_FILE = "data/raw/lyrics_collection_results.json"
LYRICS_FOLDER = "data/raw/lyrics"
CHECKPOINT_FILE = "data/raw/language_detection_v2_checkpoint.json"

MIN_TEXT_LENGTH = 15
MAIN_LANGUAGE_THRESHOLD = 0.70
MIXED_THRESHOLD = 0.30

# Các nhãn/metadata của Genius không phải nội dung lyrics.
SECTION_PATTERN = re.compile(
    r"^\s*\[(?:verse|pre-chorus|chorus|post-chorus|bridge|intro|outro|hook|refrain|instrumental|interlude|break|part|spoken)[^\]]*\]\s*$",
    re.IGNORECASE,
)
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MARKDOWN_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\([^)]*\)")
BRACKET_PATTERN = re.compile(r"\[([^\]]+)\]")


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_atomic(path: str, data: Any) -> None:
    temp_path = path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(temp_path, path)


def clean_line(line: str) -> str:
    """Remove Genius/markdown metadata while keeping actual lyric text."""
    line = line.strip()
    if not line:
        return ""

    # Bỏ section labels: [Verse 1], [Chorus], ...
    if SECTION_PATTERN.match(line):
        return ""

    # [text](url) -> text
    line = MARKDOWN_LINK_PATTERN.sub(r"\1", line)

    # Xóa URL còn sót lại
    line = URL_PATTERN.sub("", line)

    # Một số Genius lines có [text] nhưng không phải section label.
    # Giữ text bên trong ngoặc để không mất lyrics.
    line = BRACKET_PATTERN.sub(r"\1", line)

    return line.strip()


def clean_lyrics(text: str) -> list[str]:
    lines = []
    for raw_line in text.splitlines():
        line = clean_line(raw_line)
        if line:
            lines.append(line)
    return lines


def count_script_chars(text: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for ch in text:
        code = ord(ch)
        if 0xAC00 <= code <= 0xD7AF or 0x1100 <= code <= 0x11FF or 0x3130 <= code <= 0x318F:
            counts["ko"] += 1
        elif (
            0x3040 <= code <= 0x30FF
            or 0x31F0 <= code <= 0x31FF
            or 0xFF66 <= code <= 0xFF9D
        ):
            counts["ja"] += 1
        elif 0x4E00 <= code <= 0x9FFF:
            counts["zh"] += 1
    return counts


def latin_letter_count(text: str) -> int:
    return len(re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]", text))


def detect_line_language(line: str) -> str | None:
    """Detect a lyric line using script first, langdetect second."""
    script_counts = count_script_chars(line)
    script_total = sum(script_counts.values())

    # Nếu một script chiếm phần lớn ký tự script thì ưu tiên nó.
    if script_total > 0:
        top_script, top_count = script_counts.most_common(1)[0]
        if top_count / script_total >= 0.60:
            return top_script

    # Dòng không có đủ chữ thì bỏ qua.
    if latin_letter_count(line) < MIN_TEXT_LENGTH and script_total < MIN_TEXT_LENGTH:
        return None

    try:
        return detect(line)
    except LangDetectException:
        return None
    except Exception:
        return None


def detect_language(lyrics_text: str) -> tuple[str, dict[str, float]]:
    if not lyrics_text or not lyrics_text.strip():
        return "unknown", {}

    lines = clean_lyrics(lyrics_text)
    if not lines:
        return "unknown", {}

    weighted_counts: Counter[str] = Counter()
    total_weight = 0
    script_counts_total: Counter[str] = Counter()
    total_latin_chars = 0

    for line in lines:
        line_scripts = count_script_chars(line)
        line_latin = latin_letter_count(line)
        script_counts_total.update(line_scripts)
        total_latin_chars += line_latin

        language = detect_line_language(line)
        if not language:
            continue

        # Dùng số ký tự chữ làm trọng số thay vì mỗi dòng = 1.
        # Điều này giảm ảnh hưởng của các dòng cực ngắn như "Oh", "Yeah".
        weight = line_latin + sum(line_scripts.values())
        if weight < MIN_TEXT_LENGTH:
            continue

        weighted_counts[language] += weight
        total_weight += weight

    # Fallback có bằng chứng khi mọi dòng đơn lẻ đều < MIN_TEXT_LENGTH (15 ký tự),
    # thường gặp ở bài hát tiếng Hàn (Hangul) hoặc tiếng Trung (Hanzi) có câu ngắn
    # 8-14 âm tiết/dòng nhưng tổng toàn bài có hàng trăm ký tự script rõ ràng.
    if total_weight == 0:
        total_script_chars = sum(script_counts_total.values())
        total_all_chars = total_script_chars + total_latin_chars
        if total_all_chars >= MIN_TEXT_LENGTH and script_counts_total:
            top_script, top_script_count = script_counts_total.most_common(1)[0]
            # Mẫu số bắt buộc là tổng toàn bộ ký tự chữ (script + Latin)
            if top_script_count / total_all_chars >= MAIN_LANGUAGE_THRESHOLD:
                dist = {
                    lang: round(cnt / total_all_chars, 4)
                    for lang, cnt in script_counts_total.most_common()
                }
                if total_latin_chars > 0:
                    dist["latin_other"] = round(total_latin_chars / total_all_chars, 4)
                return top_script, dist
        return "unknown", {}

    distribution = {
        lang: round(count / total_weight, 4)
        for lang, count in weighted_counts.most_common()
    }

    ranked = weighted_counts.most_common()
    top_lang, top_count = ranked[0]
    top_ratio = top_count / total_weight

    # 1 ngôn ngữ chiếm >= 70% -> main language
    if top_ratio >= MAIN_LANGUAGE_THRESHOLD:
        return top_lang, distribution

    # 2 ngôn ngữ cùng >= 30% -> mixed
    significant = [
        lang for lang, count in ranked
        if count / total_weight >= MIXED_THRESHOLD
    ]
    if len(significant) >= 2:
        return "mixed", distribution

    # Không đủ rõ ràng: lấy ngôn ngữ đứng đầu.
    return top_lang, distribution


def get_lyrics_text(item: dict[str, Any], track_id: str, lyrics_folder: str = LYRICS_FOLDER) -> str | None:
    lyrics_file = item.get("lyrics_file")
    if lyrics_file:
        path = lyrics_file.replace("\\", os.sep)
    else:
        path = os.path.join(lyrics_folder, f"{track_id}.txt")

    if not os.path.exists(path):
        fallback = os.path.join(lyrics_folder, f"{track_id}.txt")
        if os.path.exists(fallback):
            path = fallback
        else:
            return None

    with open(path, "r", encoding="utf-8-sig") as f:
        return f.read()


def load_checkpoint(path: str = CHECKPOINT_FILE) -> dict[str, str]:
    if not os.path.exists(path):
        return {}
    return load_json(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run lyrics language detection (v2.1.0).")
    parser.add_argument("--result-file", default=RESULT_FILE, help="Path to lyrics_collection_results.json")
    parser.add_argument("--checkpoint-file", default=CHECKPOINT_FILE, help="Path to language detection checkpoint")
    parser.add_argument("--lyrics-folder", default=LYRICS_FOLDER, help="Folder containing raw .txt lyrics")
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute language labels instead of skipping tracks present in checkpoint",
    )
    parser.add_argument(
        "--only-unknown",
        action="store_true",
        help="Only recompute tracks currently labeled 'unknown' or missing in checkpoint",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not overwrite JSON/checkpoint files; only report changes",
    )
    args = parser.parse_args()

    results = load_json(args.result_file)
    checkpoint = load_checkpoint(args.checkpoint_file)

    print(f"Algorithm version: {ALGORITHM_VERSION}")
    print(f"Tổng số records: {len(results)}")
    print(f"Đã có checkpoint: {len(checkpoint)}")

    success_records = [
        (track_id, item)
        for track_id, item in results.items()
        if item.get("lyrics_status") == "success"
    ]

    changed_tracks: list[tuple[str, str, str, str, str]] = []
    processed_count = 0

    for index, (track_id, item) in enumerate(success_records, start=1):
        old_lang = item.get("language") or checkpoint.get(track_id) or "unknown"

        if not args.recompute and not args.only_unknown and track_id in checkpoint:
            continue
        if args.only_unknown and track_id in checkpoint and checkpoint.get(track_id) not in ("unknown", None, ""):
            continue

        processed_count += 1
        lyrics_text = get_lyrics_text(item, track_id, args.lyrics_folder)
        if not lyrics_text or not lyrics_text.strip():
            new_lang = "unknown"
            distribution: dict[str, float] = {}
        else:
            new_lang, distribution = detect_language(lyrics_text)

        if new_lang != old_lang:
            changed_tracks.append((
                track_id,
                str(item.get("title", "")),
                str(item.get("artist", "")),
                str(old_lang),
                new_lang,
            ))
            print(
                f"[{index}/{len(success_records)}] CHANGE: {track_id} | "
                f"{item.get('title')} - {item.get('artist')} | {old_lang} -> {new_lang} | {distribution}"
            )

        item["language"] = new_lang
        checkpoint[track_id] = new_lang

    # Ensure non-success tracks have None/unknown consistently
    for track_id, item in results.items():
        if item.get("lyrics_status") != "success":
            item["language"] = None

    if not args.dry_run and (processed_count > 0):
        save_json_atomic(args.checkpoint_file, checkpoint)
        save_json_atomic(args.result_file, results)

    lang_counts = Counter(
        (item.get("language") or "unknown")
        for item in results.values()
    )
    print("\n" + "=" * 60)
    print(f"LANGUAGE DETECTION {ALGORITHM_VERSION} COMPLETED")
    print("=" * 60)
    print(f"Success records: {len(success_records)}")
    print(f"Records evaluated in this run: {processed_count}")
    print(f"Total label changes: {len(changed_tracks)}")
    print("Language distribution across all 1,912 tracks:")
    for lang, cnt in lang_counts.most_common():
        print(f"  {lang}: {cnt}")
    if args.dry_run:
        print("DRY RUN: No files were modified.")
    else:
        print(f"Checkpoint updated: {args.checkpoint_file}")
        print(f"Results updated: {args.result_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
