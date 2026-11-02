#!/usr/bin/env python3
"""Extract lyrics features for ADY201m from metadata and raw lyric files.

Run from the repository root:
    python src/lyrics/extract_lyrics_features.py

The script reads paths and feature rules from language_rules.yaml unless overridden
with command-line arguments. It never modifies raw lyrics.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:
    raise SystemExit("Thiếu thư viện PyYAML. Cài bằng: python -m pip install pyyaml") from exc

try:
    import regex as unicode_re
except ImportError as exc:
    raise SystemExit(
        "Thiếu thư viện regex (hỗ trợ Unicode \\p{L}). "
        "Cài bằng: python -m pip install regex"
    ) from exc


SCRIPT_VERSION = "1.1.0"

DEFAULT_TITLE_STOPWORDS = {
    "en": set("""a an the and or but if then of to in on at for from with by is am are was were be been being do does did have has had i me my we us our you your he him his she her it they them their this that these those what which who whom as not so too very""".split()),
    "vi": set("""và là của cho với những các một đã đang sẽ thì mà ở trong trên dưới khi nếu vì do từ này đó ấy ai gì nào cũng không rất như về để được bị""".split()),
    "ko": set("""그 이 저 것 수 나 너 내 네 우리 저희 그리고 하지만 그래서 는 은 이 가 을 를 에 에서 로 으로 와 과 도 만""".split()),
    "es": set("""el la los las un una unos unas y o pero de a en por para con del al es soy eres somos son mi mis tu tus su sus yo me te se lo le les nos vosotros usted ustedes ellos ellas que quien cual como no""".split()),
}
SUPPORTED_LANGUAGES = {"en", "vi", "ko", "mixed", "es"}
PRONOUN_LANGUAGES = {"en", "vi", "ko", "es"}
FEATURE_NAMES = [
    "word_count",
    "unique_word_count",
    "lexical_diversity",
    "repetition_ratio",
    "avg_word_length",
    "line_count",
    "chorus_repetition_count",
    "title_word_in_lyrics",
    "punctuation_ratio",
    "pronoun_ratio",
]

CHORUS_TAG_RE = re.compile(
    r"\[\s*(?:chorus|refrain|hook|điệp khúc)\b[^\]]*\]",
    re.IGNORECASE,
)
ANY_SECTION_TAG_RE = re.compile(r"\[[^\]]*\]")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Cấu hình YAML không hợp lệ: {path}")
    return data


def normalize_text(text: str, config: dict[str, Any]) -> str:
    cfg = config.get("normalization", {})
    text = unicodedata.normalize(cfg.get("unicode_normalization", "NFC"), text)
    if cfg.get("strip_leading_trailing_whitespace", True):
        text = text.strip()
    if cfg.get("collapse_consecutive_whitespace", True):
        # Keep line boundaries for line_count and chorus rules.
        text = unicode_re.sub(r"[\t\f\v ]+", " ", text)
        text = unicode_re.sub(r" *\n *", "\n", text)
    return text


def clean_token(token: str, config: dict[str, Any]) -> str:
    cfg = config.get("normalization", {}).get("token_normalization", {})
    token = unicodedata.normalize(cfg.get("unicode_normalization", "NFC"), token)
    if cfg.get("lowercase", True):
        token = token.casefold()
    if cfg.get("trim_surrounding_punctuation", True):
        token = token.strip("\"'’‘“”.,!?;:()[]{}<>…—–-·")
    if cfg.get("ignore_tokens_without_letters_or_numbers", True):
        if not any(unicode_re.match(r"[\p{L}\p{N}]", char) for char in token):
            return ""
    return token


def tokenize(text: str, language: str, config: dict[str, Any]) -> list[str]:
    # Strip structural section headers like [Chorus], [Verse 1] so metadata tags
    # are not counted as lyric words.
    lyric_body = ANY_SECTION_TAG_RE.sub(" ", text or "")
    rules = config.get("tokenization", {})
    lang_rule = rules.get(language, rules.get("other", {}))
    method = lang_rule.get("method", "unicode_word_regex")
    if method == "whitespace_delimited_unicode_tokens":
        raw_tokens = lyric_body.split()
    else:
        pattern = lang_rule.get("pattern") or r"[\p{L}\p{N}]+(?:['’\-][\p{L}\p{N}]+)*"
        raw_tokens = unicode_re.findall(pattern, lyric_body)
    return [token for raw in raw_tokens if (token := clean_token(raw, config))]


def count_punctuation(text: str, config: dict[str, Any]) -> int:
    categories = set(
        config.get("normalization", {})
        .get("punctuation", {})
        .get("unicode_categories", ["Pc", "Pd", "Pe", "Pf", "Pi", "Po", "Ps"])
    )
    return sum(1 for char in text if unicodedata.category(char) in categories)


def get_title_stopwords(language: str, config: dict[str, Any] | None = None) -> set[str]:
    cfg_stopwords = (config or {}).get("title_stopwords", {})
    if language in cfg_stopwords and isinstance(cfg_stopwords[language], list):
        return {str(w).strip().casefold() for w in cfg_stopwords[language] if str(w).strip()}
    if language in DEFAULT_TITLE_STOPWORDS:
        return DEFAULT_TITLE_STOPWORDS[language]
    if language == "mixed":
        # Conservative union: mixed-language titles may contain tokens from several
        # supported languages. This limitation is documented in analysis.
        if cfg_stopwords:
            return {
                str(w).strip().casefold()
                for words in cfg_stopwords.values()
                if isinstance(words, list)
                for w in words
                if str(w).strip()
            }
        return set().union(*DEFAULT_TITLE_STOPWORDS.values())
    return set()


def title_tokens(title: str, language: str, config: dict[str, Any]) -> set[str]:
    tokens = tokenize(title or "", language, config)
    stopwords = get_title_stopwords(language, config)
    return {
        token for token in tokens
        if token not in stopwords and any(char.isalpha() for char in token)
    }


def count_pronoun_tokens(tokens: list[str], dictionary: Any, config: dict[str, Any]) -> int:
    """Count token positions matched by exact single- or multi-token pronouns.

    Longest phrase wins at a position, so a phrase such as 'chúng tôi' counts its
    two token positions once rather than double-counting 'tôi' as a second match.
    """
    if not isinstance(dictionary, list):
        return 0

    phrases: list[tuple[str, ...]] = []
    for item in dictionary:
        phrase = str(item).strip()
        if not phrase:
            continue
        phrase_tokens = tuple(
            clean_token(part, config)
            for part in phrase.split()
        )
        phrase_tokens = tuple(part for part in phrase_tokens if part)
        if phrase_tokens:
            phrases.append(phrase_tokens)

    phrases = sorted(set(phrases), key=lambda p: (-len(p), p))
    matched_positions = 0
    i = 0
    while i < len(tokens):
        matched_len = 0
        for phrase in phrases:
            n = len(phrase)
            if tuple(tokens[i:i + n]) == phrase:
                matched_len = n
                break
        if matched_len:
            matched_positions += matched_len
            i += matched_len
        else:
            i += 1
    return matched_positions


def _normalize_chorus_line(line: str) -> str:
    line = unicodedata.normalize("NFC", line).casefold()
    line = re.sub(r"[^\w\s]", " ", line, flags=re.UNICODE)
    return re.sub(r"\s+", " ", line).strip()


def detect_chorus_repetition(
    text: str, config: dict[str, Any] | None = None
) -> int | None:
    """Identify chorus repetition count using explicit tags or multi-line repetition.

    Rules:
    1. If explicit [Chorus]/[Refrain]/[Hook]/[Điệp khúc] tags exist:
       - If >= 2 tags exist, the chorus occurs at least len(tags) times
         (chorus_repetition_count >= len(tags) - 1).
       - If 1 tag exists and no repeated block is found in the body, returns 0
         (chorus was explicitly identified by tag and verified to have 0 additional
         repetitions).
    2. For unlabeled lyrics (or to check if a single tagged chorus repeats untagged):
       - Strip section tags, normalize lines, and search for non-overlapping
         multi-line blocks (3..10 lines, >= 2 unique lines, >= 6 words) repeating
         with token-level similarity >= 0.80.
       - If a repeated block is found, returns (occurrences - 1).
       - If NO explicit chorus tag exists and NO repeated multi-line block is found,
         returns None (NULL) because the chorus cannot be reliably identified.
         Never defaults to 0 when chorus is unidentified.
    """
    if not text or not text.strip():
        return None

    chorus_cfg = (config or {}).get("chorus_detection", {})
    min_lines = int(chorus_cfg.get("min_block_lines", 3))
    max_lines_cfg = int(chorus_cfg.get("max_block_lines", 10))
    min_unique_lines = int(chorus_cfg.get("min_unique_lines", 2))
    min_block_words = int(chorus_cfg.get("min_block_words", 6))
    sim_threshold = float(chorus_cfg.get("similarity_threshold", 0.80))

    chorus_tags = CHORUS_TAG_RE.findall(text)
    clean_body = ANY_SECTION_TAG_RE.sub("", text)
    lines = [nl for raw in clean_body.splitlines() if (nl := _normalize_chorus_line(raw))]

    if len(chorus_tags) >= 2:
        return len(chorus_tags) - 1

    n = len(lines)
    if n < min_lines * 2:
        if len(chorus_tags) == 1 and n >= 2:
            return 0
        return None

    max_lines = min(max_lines_cfg, n // 2)
    line_positions: dict[str, list[int]] = defaultdict(list)
    for idx, line in enumerate(lines):
        if len(line) >= 3:
            line_positions[line].append(idx)

    best_count = 0
    best_score = 0

    for idx, anchor_line in enumerate(lines[: n - min_lines + 1]):
        positions = line_positions.get(anchor_line, [])
        if len(positions) < 2:
            continue
        for length in range(min_lines, min(max_lines, n - idx) + 1):
            block_lines = lines[idx : idx + length]
            if len(set(block_lines)) < min_unique_lines:
                continue
            block_words = " ".join(block_lines).split()
            if len(block_words) < min_block_words:
                continue

            occurrences = [(idx, idx + length)]
            last_end = idx + length
            for cand_pos in positions:
                if cand_pos < last_end or cand_pos + min_lines > n:
                    continue
                cand_len = min(length, n - cand_pos)
                if abs(cand_len - length) > 1:
                    continue
                cand_lines = lines[cand_pos : cand_pos + cand_len]
                if cand_lines == block_lines:
                    sim = 1.0
                else:
                    cand_words = " ".join(cand_lines).split()
                    matcher = SequenceMatcher(None, block_words, cand_words)
                    sim = matcher.ratio() if matcher.quick_ratio() >= sim_threshold else 0.0
                if sim >= sim_threshold:
                    occurrences.append((cand_pos, cand_pos + cand_len))
                    last_end = cand_pos + cand_len

            if len(occurrences) >= 2:
                score = length * len(occurrences)
                if score > best_score:
                    best_score = score
                    best_count = len(occurrences) - 1

    if best_count >= 1:
        return best_count
    if len(chorus_tags) == 1:
        return 0
    return None


def calculate_features(
    text: str, title: str, language: str, config: dict[str, Any]
) -> dict[str, Any]:
    if text is None or not text.strip():
        return {name: None for name in FEATURE_NAMES}

    trimmed_text = text.strip()
    normalized = normalize_text(text, config)
    tokens = tokenize(normalized, language, config)
    word_count = len(tokens)
    unique_count = len(set(tokens))
    nonempty_lines = [line.strip() for line in text.splitlines() if line.strip()]

    lexical_diversity = (unique_count / word_count) if word_count > 0 else None
    repetition_ratio = ((word_count - unique_count) / word_count) if word_count > 0 else None
    avg_word_length = (
        sum(len(token) for token in tokens) / word_count
        if word_count > 0
        else None
    )

    chorus_repetition_count = (
        detect_chorus_repetition(text, config) if word_count > 0 else None
    )

    title_supported = language in SUPPORTED_LANGUAGES
    title_content_tokens = (
        title_tokens(title, language, config) if (title and title_supported) else set()
    )
    if word_count == 0 or not normalized or not title or not title_supported or not title_content_tokens:
        title_word_in_lyrics = None
    else:
        title_word_in_lyrics = bool(title_content_tokens.intersection(set(tokens)))

    punctuation_ratio = (
        count_punctuation(trimmed_text, config) / len(trimmed_text)
        if trimmed_text
        else None
    )

    pronoun_ratio = None
    pronoun_dictionary = config.get("pronouns", {}).get(language)
    if word_count > 0 and language in PRONOUN_LANGUAGES and isinstance(pronoun_dictionary, list):
        pronoun_count = count_pronoun_tokens(tokens, pronoun_dictionary, config)
        pronoun_ratio = pronoun_count / word_count

    return {
        "word_count": word_count,
        "unique_word_count": unique_count,
        "lexical_diversity": lexical_diversity,
        "repetition_ratio": repetition_ratio,
        "avg_word_length": avg_word_length,
        "line_count": len(nonempty_lines) if nonempty_lines else None,
        "chorus_repetition_count": chorus_repetition_count,
        "title_word_in_lyrics": title_word_in_lyrics,
        "punctuation_ratio": punctuation_ratio,
        "pronoun_ratio": pronoun_ratio,
    }


def resolve_lyrics_path(project_root: Path, raw_value: str, lyrics_dir: Path) -> Path | None:
    if not raw_value or not raw_value.strip():
        return None
    normalized_value = raw_value.replace("\\", "/")
    candidate = Path(normalized_value)
    if candidate.is_absolute():
        return candidate
    direct = project_root / candidate
    if direct.exists():
        return direct
    return lyrics_dir / candidate.name


def read_csv_rows(path: Path, config: dict[str, Any]) -> list[dict[str, str]]:
    input_cfg = config.get("input", {})
    encoding = input_cfg.get("encoding", "utf-8-sig")
    with path.open("r", encoding=encoding, newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError(f"CSV không có header: {path}")
        required = {
            input_cfg.get("track_id_column", "spotify_track_id"),
            input_cfg.get("title_column", "title"),
            input_cfg.get("language_column", "language"),
            input_cfg.get("lyrics_status_column", "lyrics_status"),
            input_cfg.get("lyrics_file_column", "lyrics_file"),
        }
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"CSV thiếu cột bắt buộc: {', '.join(sorted(missing))}")
        return list(reader)


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    return value


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Trích xuất lyrics features cho ADY201m")
    parser.add_argument("--root", default=".", help="Thư mục gốc repository")
    parser.add_argument("--config", default="src/lyrics/language_rules.yaml", help="Đường dẫn YAML rules")
    parser.add_argument("--input", default=None, help="Ghi đè đường dẫn metadata trong YAML")
    parser.add_argument("--lyrics-dir", default=None, help="Ghi đè thư mục lyrics trong YAML")
    parser.add_argument("--output", default="reports/report3/lyrics_features_extracted.csv", help="CSV features đầu ra")
    parser.add_argument("--log", default="logs/lyrics_extraction_log.csv", help="CSV log đầu ra")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    config_path = (root / args.config).resolve()
    if not config_path.exists():
        print(f"Lỗi: không tìm thấy file cấu hình: {config_path}", file=sys.stderr)
        return 2

    try:
        config = load_yaml(config_path)
        input_cfg = config.get("input", {})
        input_value = args.input or input_cfg.get("status_csv", "data/processed/lyrics_status.csv")
        lyrics_dir_value = args.lyrics_dir or input_cfg.get("lyrics_directory", "data/raw/lyrics")
        input_path = (root / input_value).resolve()
        lyrics_dir = (root / lyrics_dir_value).resolve()
        output_path = (root / args.output).resolve()
        log_path = (root / args.log).resolve()

        if not input_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file metadata: {input_path}")
        if not lyrics_dir.exists() or not lyrics_dir.is_dir():
            raise FileNotFoundError(f"Không tìm thấy thư mục lyrics: {lyrics_dir}")

        rows = read_csv_rows(input_path, config)
        fields = {
            "track_id": input_cfg.get("track_id_column", "spotify_track_id"),
            "title": input_cfg.get("title_column", "title"),
            "market": input_cfg.get("market_column", "market"),
            "language": input_cfg.get("language_column", "language"),
            "lyrics_status": input_cfg.get("lyrics_status_column", "lyrics_status"),
            "lyrics_file": input_cfg.get("lyrics_file_column", "lyrics_file"),
            "source": input_cfg.get("source_column", "source"),
        }

        feature_fields = [
            fields["track_id"],
            fields["title"],
            "artist",
            fields["market"],
            fields["language"],
            fields["source"],
            fields["lyrics_status"],
            "extraction_status",
            *FEATURE_NAMES,
            "rules_version",
        ]
        log_fields = [
            fields["track_id"],
            fields["title"],
            fields["language"],
            fields["lyrics_status"],
            "extraction_status",
            fields["lyrics_file"],
            "word_count",
            "error_reason",
            "processed_at",
            "rules_version",
        ]

        seen_ids: set[str] = set()
        feature_rows: list[dict[str, Any]] = []
        log_rows: list[dict[str, Any]] = []
        duplicate_count = 0
        success_count = 0
        language_limited_count = 0

        for row in rows:
            track_id = (row.get(fields["track_id"]) or "").strip()
            title = (row.get(fields["title"]) or "").strip()
            market = (row.get(fields["market"]) or "").strip()
            language = (row.get(fields["language"]) or "").strip().lower()
            language = language if language and language.casefold() != "nan" else "unknown"
            lyrics_status = (row.get(fields["lyrics_status"]) or "").strip().lower()
            lyrics_file_value = (row.get(fields["lyrics_file"]) or "").strip()
            processed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            error_reason = ""
            extraction_status = "failed"
            features: dict[str, Any] = {name: None for name in FEATURE_NAMES}

            duplicate_input_row = bool(track_id and track_id in seen_ids)
            if not track_id:
                extraction_status, error_reason = "invalid_track_id", "missing_spotify_track_id"
            elif duplicate_input_row:
                duplicate_count += 1
                extraction_status, error_reason = "failed", "duplicate_spotify_track_id_in_input"
            else:
                seen_ids.add(track_id)
                if lyrics_status != "success":
                    extraction_status = "not_found" if lyrics_status == "not_found" else "not_attempted"
                    error_reason = f"lyrics_status={lyrics_status or 'missing'}"
                else:
                    lyrics_path = resolve_lyrics_path(root, lyrics_file_value, lyrics_dir)
                    if not lyrics_file_value or lyrics_path is None:
                        extraction_status, error_reason = "missing_file", "lyrics_file_path_missing"
                    elif not lyrics_path.exists() or not lyrics_path.is_file():
                        extraction_status, error_reason = "missing_file", f"lyrics_file_not_found:{lyrics_path}"
                    else:
                        try:
                            raw_text = lyrics_path.read_text(
                                encoding=input_cfg.get("encoding", "utf-8-sig")
                            )
                        except (OSError, UnicodeError) as exc:
                            extraction_status = "read_error"
                            error_reason = f"{type(exc).__name__}:{exc}"
                        else:
                            if not raw_text.strip():
                                extraction_status, error_reason = "empty_file", "lyrics_file_is_empty"
                            else:
                                try:
                                    features = calculate_features(raw_text, title, language, config)
                                    if not features["word_count"]:
                                        extraction_status = "failed"
                                        error_reason = "no_tokens_recognized_after_tokenization"
                                    elif language not in SUPPORTED_LANGUAGES:
                                        extraction_status = "language_limited"
                                        error_reason = "basic_features_extracted_language_specific_features_limited"
                                        language_limited_count += 1
                                    else:
                                        extraction_status = "success"
                                        success_count += 1
                                except Exception as exc:
                                    extraction_status = "failed"
                                    error_reason = f"{type(exc).__name__}:{exc}"

            # Keep the feature table unique by Track ID. Duplicate input rows are logged,
            # but are not written as a second feature row.
            if track_id and not duplicate_input_row:
                feature_row = {
                    fields["track_id"]: track_id,
                    fields["title"]: title,
                    "artist": (row.get("artist") or "").strip(),
                    fields["market"]: market,
                    fields["language"]: language,
                    fields["source"]: (row.get(fields["source"]) or "").strip(),
                    fields["lyrics_status"]: lyrics_status,
                    "extraction_status": extraction_status,
                    **features,
                    "rules_version": config.get("version", SCRIPT_VERSION),
                }
                feature_rows.append({key: csv_value(value) for key, value in feature_row.items()})

            log_rows.append({
                fields["track_id"]: track_id,
                fields["title"]: title,
                fields["language"]: language,
                fields["lyrics_status"]: lyrics_status,
                "extraction_status": extraction_status,
                fields["lyrics_file"]: lyrics_file_value,
                "word_count": csv_value(features.get("word_count")),
                "error_reason": error_reason,
                "processed_at": processed_at,
                "rules_version": config.get("version", SCRIPT_VERSION),
            })

        write_csv(output_path, feature_fields, feature_rows)
        write_csv(log_path, log_fields, log_rows)

        statuses = Counter(log_row["extraction_status"] for log_row in log_rows)
        print("Lyrics feature extraction finished.")
        print(f"Input tracks: {len(rows)}")
        print(f"Unique valid Track IDs seen: {len(seen_ids)}")
        print(f"Feature extraction success (supported languages): {success_count}")
        print(f"Basic features extracted, language-limited: {language_limited_count}")
        print(f"Duplicate Track ID rows: {duplicate_count}")
        print("Extraction statuses:")
        for status, count in sorted(statuses.items()):
            print(f"  {status}: {count}")
        print(f"Feature rows written: {len(feature_rows)}")
        print(f"Features CSV: {output_path}")
        print(f"Extraction log: {log_path}")
        return 0
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
