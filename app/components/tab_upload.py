"""
app/components/tab_upload.py
Tab 3 — Upload Âm Thanh & Trích Xuất Librosa Thời Gian Thực
Project: What Makes a Hit Song? (Comparing Chart Performance: VN, US, KR)
Owner: Member 2 (Audio Features & Showcase Engineer)

Luồng:
  1. Upload file âm thanh (.mp3 / .wav / .flac / .ogg)
  2. Kiểm tra kích thước <= 15 MB trước khi xử lý
  3. Trích xuất 17 đặc trưng âm học qua audio_extractor (cached SHA-256)
  4. Hiển thị bảng đặc trưng & các metric nổi bật (đơn vị vật lý chuẩn xác)
  5. Người dùng chọn thị trường -> dự đoán điểm số song hành (Dual-Model: OLS + RF)
     kết hợp Fallback ca từ từ data/processed/lyrics_market_medians.json

Audio Contract v2 (bắt buộc tuân thủ):
  - OLS Model 3: Sử dụng 14 biến âm học liên tục + mode (34 cột design matrix).
  - Nghiêm cấm đưa loudness_proxy và spectral_bandwidth vào OLS (tránh VIF 20.96 và 6.26).
  - Random Forest Model 4: Sử dụng đủ 17 đặc trưng âm học để bắt quan hệ phi tuyến.
  - Fallback ca từ: Nạp trung vị 6 biến ca từ từ lyrics_market_medians.json (TV3 bàn giao).
"""

from __future__ import annotations

import hashlib
import json
import time
import warnings
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from app.components.market_position import render_market_position, render_prediction_context
from app.components.prediction_feedback import render_prediction_feedback

# ── Graceful Streamlit Import (cho phép chạy headless test khi chưa có UI) ──
try:
    import streamlit as st
except ImportError:
    class _MockStreamlitContext:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def __getattr__(self, name: str):
            return lambda *args, **kwargs: _MockStreamlitContext()

    class _MockStreamlit:
        session_state: Dict[str, Any] = {}

        @staticmethod
        def cache_data(*args, **kwargs):
            def decorator(fn):
                return fn
            return decorator

        def columns(self, spec, *args, **kwargs):
            n = len(spec) if isinstance(spec, (list, tuple)) else int(spec)
            return [_MockStreamlitContext() for _ in range(n)]

        def form(self, *args, **kwargs):
            return _MockStreamlitContext()

        def container(self, *args, **kwargs):
            return _MockStreamlitContext()

        def expander(self, *args, **kwargs):
            return _MockStreamlitContext()

        def spinner(self, *args, **kwargs):
            return _MockStreamlitContext()

        def selectbox(self, label, options, *args, **kwargs):
            return options[0] if options else None

        def text_area(self, label, *args, **kwargs):
            return ""

        def __getattr__(self, name: str):
            return lambda *args, **kwargs: None

    st = _MockStreamlit()  # type: ignore

# ── Root path resolution ────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[2]

# ── Constants ───────────────────────────────────────────────────────────────
MAX_FILE_SIZE_MB = 15
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
SUPPORTED_FORMATS = ["mp3", "wav", "flac", "ogg"]

MARKET_LABELS = {"VN": "🇻🇳 Việt Nam", "US": "🇺🇸 Hoa Kỳ", "KR": "🇰🇷 Hàn Quốc"}
MARKET_CODES = {"🇻🇳 Việt Nam": "VN", "🇺🇸 Hoa Kỳ": "US", "🇰🇷 Hàn Quốc": "KR"}

# Nhóm 17 đặc trưng để hiển thị
FEATURE_GROUPS: Dict[str, list[str]] = {
    "🥁 Nhịp điệu & Năng lượng": [
        "tempo",
        "onset_rate",
        "tempo_stability",
        "energy",
        "rms_variability",
    ],
    "🎵 Phổ & Hòa âm": [
        "spectral_centroid",
        "spectral_bandwidth",
        "loudness_proxy",
        "chord_change_rate",
        "key_confidence",
        "mode",
    ],
    "🔁 Cấu trúc lặp lại (Chroma & MFCC)": [
        "chroma_repetition_determinism",
        "chroma_mean_pattern_duration",
        "chroma_longest_pattern_duration",
        "mfcc_repetition_determinism",
        "mfcc_mean_pattern_duration",
        "mfcc_longest_pattern_duration",
    ],
}

FEATURE_META: Dict[str, Dict[str, str]] = {
    "tempo":                        {"label": "Tempo",                 "unit": "BPM",        "fmt": ".1f"},
    "onset_rate":                   {"label": "Onset Rate",            "unit": "onsets/min", "fmt": ".1f"},
    "tempo_stability":              {"label": "Độ ổn định nhịp",       "unit": "",           "fmt": ".4f"},
    "energy":                       {"label": "Energy (RMS²)",         "unit": "",           "fmt": ".4f"},
    "rms_variability":              {"label": "RMS Variability",       "unit": "",           "fmt": ".4f"},
    "spectral_centroid":            {"label": "Spectral Centroid",     "unit": "Hz",         "fmt": ".1f"},
    "spectral_bandwidth":           {"label": "Spectral Bandwidth",    "unit": "Hz",         "fmt": ".1f"},
    "loudness_proxy":               {"label": "Loudness Proxy",        "unit": "dBFS",       "fmt": ".2f"},
    "chord_change_rate":            {"label": "Chord Change Rate",     "unit": "chords/min", "fmt": ".1f"},
    "key_confidence":               {"label": "Độ tin cậy giọng điệu", "unit": "",           "fmt": ".4f"},
    "mode":                         {"label": "Mode (Trưởng/Thứ)",     "unit": "",           "fmt": ""},
    "chroma_repetition_determinism":{"label": "Chroma DET",            "unit": "",           "fmt": ".4f"},
    "chroma_mean_pattern_duration": {"label": "Chroma Mean Pattern",   "unit": "s",          "fmt": ".2f"},
    "chroma_longest_pattern_duration":{"label":"Chroma Longest Pattern","unit": "s",         "fmt": ".1f"},
    "mfcc_repetition_determinism":  {"label": "MFCC DET",              "unit": "",           "fmt": ".4f"},
    "mfcc_mean_pattern_duration":   {"label": "MFCC Mean Pattern",     "unit": "s",          "fmt": ".2f"},
    "mfcc_longest_pattern_duration":{"label": "MFCC Longest Pattern",  "unit": "s",          "fmt": ".1f"},
}


# ── Core Helpers ─────────────────────────────────────────────────────────────

def _sha256(data: bytes) -> str:
    """SHA-256 hash của bytes nội dung file — dùng làm cache key chống trùng lặp."""
    return hashlib.sha256(data).hexdigest()


def _inspect_audio_duration(file_bytes: bytes) -> Tuple[Optional[float], Optional[str]]:
    """
    Kiểm tra nhanh thời lượng âm thanh bằng soundfile.info mà không cần giải mã toàn bộ mảng sóng vào RAM.
    Bảo vệ ứng dụng trước các tệp dài bất thường (podcast, DJ mix) vượt quá 360 giây.
    """
    try:
        import soundfile as sf
        info = sf.info(BytesIO(file_bytes))
        return float(info.duration), None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


@st.cache_data(show_spinner=False, max_entries=3, ttl=1800)
def _extract_cached(file_sha256: str, _audio_bytes: bytes) -> Tuple[Dict[str, Any], str]:
    """
    Trích xuất 17 đặc trưng âm học (cached theo SHA-256 nội dung file).

    Parameters
    ----------
    file_sha256 : str
        Hash SHA-256 của file — dùng làm cache key (không dùng filename).
    _audio_bytes : bytes
        Nội dung raw của file âm thanh (tiền tố '_' để Streamlit bỏ qua hashing lặp lại).

    Returns
    -------
    (features_dict, error_message)
        error_message rỗng nếu thành công.
    """
    t0 = time.perf_counter()
    print("[Upload] Đang giải mã và trích xuất audio...", flush=True)
    try:
        from src.features.audio_extractor import extract_features_for_inference

        features = extract_features_for_inference(BytesIO(_audio_bytes), include_waveform=True)
        features["_dsp_latency_sec"] = time.perf_counter() - t0
        print(f"[Upload] Audio xong sau {features['_dsp_latency_sec']:.2f}s", flush=True)
        return features, ""
    except ValueError as exc:
        return {}, str(exc)
    except Exception as exc:  # noqa: BLE001
        return {}, f"Lỗi không mong đợi khi trích xuất âm thanh: {exc}"


@st.cache_data(show_spinner=False)
def _load_lyrics_medians() -> Dict[str, Dict[str, float]]:
    """
    Nạp dữ liệu trung vị ca từ theo 3 thị trường từ data/processed/lyrics_market_medians.json (TV3 bàn giao).
    Trả về dict dạng: {"VN": {...}, "US": {...}, "KR": {...}}.
    """
    path = ROOT_DIR / "data" / "processed" / "lyrics_market_medians.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    result: Dict[str, Dict[str, float]] = {}
    for market, v in raw.get("markets", {}).items():
        result[market] = dict(v.get("medians", {}))
    return result


@st.cache_data(show_spinner=False)
def _load_train_medians() -> Dict[str, float]:
    """Nạp train_medians từ models/feature_metadata.json (dùng làm global fallback cấp 2)."""
    path = ROOT_DIR / "models" / "feature_metadata.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta.get("train_medians", {})


def _build_prediction_payload(
    features: Dict[str, Any],
    market: str,
    lyrics_medians: Dict[str, float],
    train_medians: Dict[str, float],
    user_lyrics_features: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Ghép vector đặc trưng chuẩn bị cho DualModelPredictor:
    - 17 đặc trưng âm học thô từ audio_extractor
    - 6 biến ca từ: ưu tiên trích xuất từ ca từ người dùng nhập; chỉ biến nào thiếu mới dùng trung vị thị trường
    - release_year mặc định 2024
    - market theo lựa chọn người dùng (VN/US/KR)
    """
    lyrics_cols = [
        "word_count", "repetition_ratio", "avg_word_length",
        "line_count", "chorus_repetition_count", "punctuation_ratio",
    ]
    payload = {key: value for key, value in features.items() if not key.startswith("_")}
    lyrics_provenance: Dict[str, Dict[str, Any]] = {}

    for col in lyrics_cols:
        val = None
        source = "Trung vị thị trường"
        if user_lyrics_features and user_lyrics_features.get(col) is not None:
            try:
                candidate = float(user_lyrics_features[col])
                if np.isfinite(candidate):
                    val = candidate
                    source = "Trích xuất từ ca từ"
            except (ValueError, TypeError):
                val = None

        if val is None:
            val = lyrics_medians.get(col) if lyrics_medians else None
            if val is None:
                val = train_medians.get(col, 0.0)
                source = "Trung vị tập huấn luyện" if col in train_medians else "Mặc định"
            else:
                source = "Trung vị thị trường"

        payload[col] = float(val)
        lyrics_provenance[col] = {
            "value": float(val),
            "source": source,
        }

    payload["_lyrics_provenance"] = lyrics_provenance
    payload["market"] = str(market).upper()
    payload["release_year"] = 2024.0
    return payload


def _predict_dual_models(payload: Dict[str, Any]) -> Tuple[Optional[Dict[str, float]], str]:
    """
    Gọi DualModelPredictor từ src.utils.pipeline_inference để dự đoán song hành:
    - OLS Model 3 (tuân thủ Audio Contract v2)
    - Random Forest Model 4 (benchmark phi tuyến)
    - Ensemble Prediction

    Returns
    -------
    (result_dict, error_message)
    """
    try:
        from src.utils.pipeline_inference import get_predictor

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            predictor = get_predictor()

        res = predictor.predict(payload, allow_fallback=False)
        return res, ""
    except Exception as exc:  # noqa: BLE001
        return None, f"Lỗi dự đoán từ pipeline inference: {exc}"


# ── UI Helpers ───────────────────────────────────────────────────────────────

def _fmt_feature_val(key: str, val: Any) -> str:
    """Format giá trị số thuần túy của đặc trưng (không kèm đơn vị)."""
    if key == "mode":
        return "Major 🎵" if float(val) >= 0.5 else "Minor 🎶"
    meta = FEATURE_META.get(key, {})
    fmt = meta.get("fmt", ".4f")
    if fmt:
        try:
            return f"{float(val):{fmt}}"
        except (ValueError, TypeError):
            return str(val)
    return str(val)


def _fmt_feature(key: str, val: Any) -> str:
    """Format giá trị đặc trưng kèm đơn vị chuẩn xác."""
    val_str = _fmt_feature_val(key, val)
    if key == "mode":
        return val_str
    unit = FEATURE_META.get(key, {}).get("unit", "")
    return f"{val_str} {unit}".strip()


def _render_feature_table(features: Dict[str, Any]) -> None:
    """Hiển thị 17 đặc trưng theo 3 nhóm cấu trúc âm học."""
    for group_name, keys in FEATURE_GROUPS.items():
        with st.expander(group_name, expanded=True):
            rows = []
            for k in keys:
                meta = FEATURE_META.get(k, {})
                val = features.get(k)
                if val is None:
                    continue
                rows.append({
                    "Đặc trưng": meta.get("label", k),
                    "Giá trị": _fmt_feature_val(k, val),
                    "Đơn vị": meta.get("unit", ""),
                })
            if rows:
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True, hide_index=True)


def _render_highlight_metrics(features: Dict[str, Any]) -> None:
    """Hiển thị 4 metric tiêu điểm hàng đầu."""
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🎵 Tempo", f"{features.get('tempo', 0):.1f} BPM")
    col2.metric("⚡ Energy (RMS²)", f"{features.get('energy', 0):.4f}")
    col3.metric("📡 Spectral Centroid", f"{features.get('spectral_centroid', 0):.0f} Hz")
    key_root = features.get("key", "")
    mode_name = "Major 🎵" if float(features.get("mode", 1)) >= 0.5 else "Minor 🎶"
    full_key = f"{key_root} {mode_name}".strip() if key_root else mode_name
    col4.metric("🎹 Điệu tính & Giọng", full_key)


def _render_prediction_result(
    pred: Dict[str, float],
    market_label: str,
    market_code: str = "VN",
) -> None:
    st.subheader(f"Kết quả dự đoán · {market_label}")
    render_market_position(st, pred["rf_prediction"], market_code, "upload_market_position", explain=True)
    render_prediction_context(st)
    with st.expander("Xem điểm của hai mô hình"):
        st.caption("Năm phát hành giả định: 2024.")
        ols, rf, ensemble = st.columns(3)
        ols.metric("OLS", f"{pred['ols_prediction']:.1f}")
        rf.metric("Random Forest", f"{pred['rf_prediction']:.1f}")
        ensemble.metric("Trung bình hai mô hình", f"{pred['ensemble_prediction']:.1f}")


# ── Main render function ──────────────────────────────────────────────────────

def render() -> None:
    """
    Giao diện chính Tab 3 — Upload Âm Thanh & Trích Xuất Librosa Thời Gian Thực.
    Được import và gọi từ app/main.py.
    """
    st.header("🎙️ Tải Lên Âm Thanh & Trích Xuất Librosa Thời Gian Thực")
    st.caption(
        "Trích xuất trực tiếp 17 đặc trưng âm thanh số (DSP) bằng Librosa và dự đoán thành tích "
        "bằng kiến trúc mô hình song hành (OLS Model 3 & Random Forest Model 4)."
    )

    left_col, right_col = st.columns([1, 2], gap="large")

    with left_col:
        with st.form("tab3_upload_form", enter_to_submit=False, border=False):
            st.subheader("📂 Chọn file âm thanh")
            uploaded = st.file_uploader(
                label="Kéo thả hoặc chọn file nhạc",
                type=SUPPORTED_FORMATS,
                help=f"Hỗ trợ: {', '.join(f'.{e}' for e in SUPPORTED_FORMATS)} | Dung lượng tối đa {MAX_FILE_SIZE_MB} MB",
                key="tab3_uploader",
                max_upload_size=MAX_FILE_SIZE_MB,
            )

            st.subheader("🌏 Thị trường đánh giá")
            market_label = st.selectbox(
                "Chọn thị trường mục tiêu",
                options=list(MARKET_CODES.keys()),
                key="tab3_market_select",
                help="Thị trường quyết định vector dummy, hệ số độ dốc tương tác và bộ trung vị ca từ.",
            )
            market_code = MARKET_CODES.get(market_label, "VN")

            st.subheader("📝 Ca Từ / Lyrics (Tùy chọn)")
            user_lyrics_text = st.text_area(
                label="Dán lời bài hát nếu có:",
                placeholder="Dán lời bài hát tại đây.",
                height=140,
                key="tab3_lyrics_input",
                help="Để trống nếu chưa có lời. App sẽ dùng trung vị ca từ của thị trường đã chọn.",
            )

            predict_btn = st.form_submit_button(
                "Phân tích và dự đoán", key="tab3_predict_btn", type="primary", width="stretch"
            )

        with st.expander("ℹ️ Tiêu chuẩn Kỹ thuật Phân hệ", expanded=False):
            st.markdown(
                f"""
                - **Giới hạn an toàn:** Tối đa {MAX_FILE_SIZE_MB} MB, giải mã tối đa 360 giây (6 phút).
                - **Librosa DSP Engine:** Trích xuất 17 đặc trưng âm thanh số (DSP) thời gian thực, tự động giải phóng bộ đệm sau xử lý.
                - **Audio Contract v2:** OLS chỉ dùng 14 biến liên tục + mode (cấm `loudness_proxy` và `spectral_bandwidth`).
                - **Fallback Ca từ:** Đồng bộ trung vị thị trường từ `lyrics_market_medians.json` (TV3).
                - **Hiệu năng:** Caching `@st.cache_data` qua mã băm SHA-256 nội dung file.
                """
            )

    with right_col:
        if predict_btn:
            print(f"[Upload] Đã nhận yêu cầu ({market_code}), có tệp: {uploaded is not None}.", flush=True)
        if uploaded is None:
            st.session_state.pop("tab3_predictions", None)
            st.session_state.pop("tab3_current_file_sha", None)
            st.session_state.pop("tab3_lyrics_extraction", None)
            if predict_btn:
                st.error("Hãy chọn tệp âm thanh trước khi bấm Phân tích và dự đoán.")
            else:
                st.info("Chọn tệp âm thanh rồi bấm **Phân tích và dự đoán**.")
            return

        file_bytes = uploaded.getvalue()
        file_size_mb = len(file_bytes) / (1024 * 1024)
        if not file_bytes:
            st.error("Tệp âm thanh rỗng. Hãy chọn một tệp khác.")
            return

        if len(file_bytes) > MAX_FILE_SIZE_BYTES:
            st.error(
                f"❌ File vượt quá giới hạn an toàn {MAX_FILE_SIZE_MB} MB "
                f"(dung lượng tệp: {file_size_mb:.1f} MB). "
                "Vui lòng cắt ngắn hoặc nén tệp trước khi tải lên."
            )
            return

        # Pre-check thời lượng âm thanh bằng metadata header trước khi decode Librosa
        est_duration, _ = _inspect_audio_duration(file_bytes)
        if est_duration is not None:
            if est_duration > 360.0:
                st.warning(
                    f"⏱️ **Lưu ý thời lượng:** Tệp gốc có thời lượng {est_duration:.1f} giây (> 6 phút). "
                    "Hệ thống sẽ chuẩn hóa và giải mã 360 giây đầu tiên (6 phút) theo tiêu chuẩn bảo vệ tài nguyên."
                )
            elif est_duration < 10.0:
                st.warning(
                    f"⚠️ **Cảnh báo thời lượng:** Tệp ngắn ({est_duration:.1f}s < 10s). "
                    "Các chỉ số phân tích cấu trúc lặp (Chroma & MFCC RQA) có thể trở về 0 do không đủ độ dài lặp."
                )

        st.success(f"✅ Đã nạp tệp thành công: **{uploaded.name}** ({file_size_mb:.2f} MB)")

        file_sha = _sha256(file_bytes)
        cache_key = f"tab3_features_{file_sha}"

        if cache_key not in st.session_state:
            with st.spinner("🔬 Đang phân tích tín hiệu âm thanh Librosa... (lần đầu có thêm bước khởi tạo)"):
                features, err = _extract_cached(file_sha, file_bytes)
            st.session_state[cache_key] = (features, err)
        else:
            features, err = st.session_state[cache_key]

        if err:
            if "im lặng" in err.lower() or "silent" in err.lower():
                st.warning(f"⚠️ Phát hiện âm thanh bất thường (quá im lặng):\n\n{err}")
            elif "15mb" in err.lower() or "size" in err.lower():
                st.error(f"❌ {err}")
            else:
                st.error(f"❌ Không thể phân tích tệp âm thanh:\n\n{err}")
            return

        if not features:
            st.error("❌ Quá trình trích xuất không trả về kết quả.")
            return

        user_lyrics_features = None
        lyrics_text = user_lyrics_text.strip()
        has_user_lyrics = bool(lyrics_text)
        lyrics_hash = hashlib.sha256(lyrics_text.encode("utf-8")).hexdigest() if has_user_lyrics else "nomedia"
        lyrics_key = (file_sha, lyrics_hash)
        previous_lyrics = st.session_state.get("tab3_lyrics_extraction")
        same_lyrics = previous_lyrics is not None and previous_lyrics[0] == lyrics_key

        if has_user_lyrics:
            try:
                if same_lyrics:
                    extracted_lyrics = previous_lyrics[1]
                else:
                    with st.spinner("Đang trích xuất đặc trưng ca từ..."):
                        start = time.perf_counter()
                        print("[Upload] Đang trích xuất ca từ...", flush=True)
                        from src.features.lyrics_processor import process_lyrics_text

                        extracted_lyrics = process_lyrics_text(file_sha, lyrics_text)
                        print(f"[Upload] Ca từ xong sau {time.perf_counter() - start:.2f}s", flush=True)
                    st.session_state["tab3_lyrics_extraction"] = (lyrics_key, extracted_lyrics)
                if extracted_lyrics.get("extraction_status") == "success":
                    user_lyrics_features = extracted_lyrics
                else:
                    st.warning(f"⚠️ Trích xuất ca từ không thành công ({extracted_lyrics.get('error_reason', 'Không nhận diện được từ')}). Hệ thống tự động sử dụng trung vị thị trường.")
            except Exception as e:
                print(f"[Upload] Lỗi trích xuất ca từ ({type(e).__name__}): {e}", flush=True)
                st.warning(f"⚠️ Không thể trích xuất đặc trưng ca từ: {e}. Hệ thống tự động sử dụng trung vị thị trường.")
        elif not has_user_lyrics:
            st.info(
                f"ℹ️ **Cơ chế Fallback Ca từ:** Tệp âm thanh chưa kèm lời ca. "
                f"Hệ thống tự động nạp trung vị 6 đặc trưng ca từ của **{market_label}** từ `lyrics_market_medians.json` "
                "để tạo vector đầu vào hoàn chỉnh cho mô hình."
            )

        # Đồng bộ hóa bộ nhớ đệm dự đoán theo tệp, thị trường và ca từ
        if "tab3_predictions" not in st.session_state:
            st.session_state["tab3_predictions"] = {}

        # Làm mới kết quả nếu người dùng thay đổi tệp âm thanh
        if st.session_state.get("tab3_current_file_sha") != file_sha:
            st.session_state["tab3_current_file_sha"] = file_sha
            st.session_state["tab3_predictions"] = {}

        pred_key = f"{file_sha}_{market_code}_{lyrics_hash}"

        pred_err = ""
        if predict_btn:
            st.session_state["tab3_predictions"].pop(pred_key, None)
            with st.spinner("⚙️ Đang chuẩn bị đầu vào và chạy OLS & Random Forest..."):
                try:
                    market_medians = _load_lyrics_medians().get(market_code, {})
                    train_medians = _load_train_medians()
                    if not market_medians:
                        st.warning(
                            f"⚠️ Thiếu bảng trung vị {market_code}. "
                            "Hệ thống dùng trung vị tập huấn luyện cho các biến ca từ còn thiếu."
                        )
                    payload = _build_prediction_payload(
                        features, market_code, market_medians, train_medians,
                        user_lyrics_features=user_lyrics_features,
                    )
                except (OSError, ValueError, TypeError, AttributeError) as exc:
                    pred_err = f"Lỗi chuẩn bị đầu vào dự đoán: {exc}"
                else:
                    start = time.perf_counter()
                    print("[Upload] Đang chạy OLS và Random Forest...", flush=True)
                    pred_res, pred_err = _predict_dual_models(payload)
                    if pred_res is not None and not pred_err:
                        st.session_state["tab3_predictions"][pred_key] = (pred_res, payload["_lyrics_provenance"])
                        print(f"[Upload] Dự đoán xong sau {time.perf_counter() - start:.2f}s", flush=True)

            if pred_err:
                print(f"[Upload] {pred_err}", flush=True)
                st.error(pred_err)

        # Hiển thị kết quả tương ứng chính xác với thị trường hiện tại
        cached_prediction = st.session_state["tab3_predictions"].get(pred_key)
        if cached_prediction is not None:
            current_pred, provenance = cached_prediction
            _render_prediction_result(current_pred, market_label, market_code=market_code)
            extracted_count = sum(item["source"] == "Trích xuất từ ca từ" for item in provenance.values())
            if 0 < extracted_count < len(provenance):
                st.caption(f"{len(provenance) - extracted_count} biến ca từ dùng giá trị thay thế.")
        elif not pred_err:
            st.info(f"💡 Bấm **Phân tích và dự đoán** ở cột bên trái để tính điểm tại **{market_label}**.")

        st.markdown("---")
        st.subheader("Âm học")
        audio_ext = uploaded.name.split(".")[-1].lower() if "." in uploaded.name else "mp3"
        audio_mime = "audio/mpeg" if audio_ext == "mp3" else f"audio/{audio_ext}"
        st.audio(file_bytes, format=audio_mime)

        waveform = features.get("_waveform")
        if waveform:
            figure = go.Figure()
            figure.add_scatter(x=waveform["time"], y=waveform["min"], line={"width": 0}, hoverinfo="skip")
            figure.add_scatter(
                x=waveform["time"], y=waveform["max"], line={"width": 0},
                fill="tonexty", fillcolor="#367be8", hoverinfo="skip",
            )
            figure.update_layout(
                template="plotly_white", height=155, showlegend=False,
                xaxis_title="Thời gian (giây)", yaxis={"visible": False},
                margin={"l": 0, "r": 0, "t": 8, "b": 0},
            )
            st.plotly_chart(figure, key="upload_waveform", config={"displayModeBar": False})

        duration = features.get("duration_analyzed_seconds", 0)
        key_name = features.get("key", "N/A")
        if est_duration is not None and est_duration > 360.0:
            duration_desc = f"**{duration:.1f} giây** (trích đoạn 6 phút đầu từ tệp gốc {est_duration:.1f}s)"
        else:
            duration_desc = f"**{duration:.1f} giây**"

        lat = features.get("_dsp_latency_sec", 0.0)
        telemetry = f" | ⚡ Độ trễ DSP: **{lat:.2f}s**" if lat > 0 else ""

        st.caption(
            f"⏱️ Thời lượng giải mã: {duration_desc} | "
            f"🎼 Điệu tính ước lượng: **{key_name}** {'Major 🎵' if float(features.get('mode', 1)) >= 0.5 else 'Minor 🎶'}"
            f"{telemetry}"
        )

        _render_highlight_metrics(features)

        with st.expander("Xem 17 đặc trưng âm học"):
            _render_feature_table(features)

        if user_lyrics_features is not None or cached_prediction is not None:
            st.subheader("Ca từ")
        if user_lyrics_features is not None:
            with st.expander("📝 Phân Tích & Trích Xuất Đặc Trưng Ca Từ (Lyrics NLP)", expanded=False):
                wc = user_lyrics_features.get("word_count")
                lc = user_lyrics_features.get("line_count")
                rep = user_lyrics_features.get("repetition_ratio")
                awl = user_lyrics_features.get("avg_word_length")
                ch = user_lyrics_features.get("chorus_repetition_count")
                pq = user_lyrics_features.get("punctuation_ratio")

                lc1, lc2, lc3 = st.columns(3)
                lc1.metric("Số từ (Word Count)", f"{int(wc)}" if wc is not None else "N/A")
                lc2.metric("Số dòng (Line Count)", f"{int(lc)}" if lc is not None else "N/A")
                lc3.metric("Tỷ lệ lặp (Repetition)", f"{rep:.3f}" if rep is not None else "N/A")

                ch_desc = f"{int(ch)}" if ch is not None else "Không xác định (Dùng trung vị)"
                awl_desc = f"{awl:.2f} ký tự" if awl is not None else "Không xác định"
                pq_desc = f"{pq:.4f}" if pq is not None else "Không xác định"
                st.caption(
                    f"Độ dài từ TB: **{awl_desc}** | "
                    f"Lần lặp điệp khúc: **{ch_desc}** | "
                    f"Tỷ lệ dấu câu: **{pq_desc}**"
                )
        if cached_prediction is not None:
            with st.expander("Xem nguồn dữ liệu ca từ"):
                st.caption(
                    f"Trích xuất từ lời nhập: {extracted_count}/{len(provenance)} biến. "
                    f"Dùng giá trị thay thế: {len(provenance) - extracted_count} biến."
                )
                st.table(pd.DataFrame.from_dict(provenance, orient="index").rename(
                    columns={"value": "Giá trị", "source": "Nguồn"}
                ))
            render_prediction_feedback(st, "upload")


# Alias to conform with team-wide naming conventions (e.g. render_tab_diagnostics)
render_tab_upload = render
