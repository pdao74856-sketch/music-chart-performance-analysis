"""Collect a short usability response about the market percentile."""

import sqlite3
from contextlib import closing
from pathlib import Path

FEEDBACK_PATH = Path(__file__).resolve().parents[2] / "data" / "feedback.sqlite3"
ANSWERS = {
    "relative_position": "Vị trí của điểm dự đoán so với các điểm trong mẫu tham chiếu",
    "hit_probability": "Xác suất bài hát sẽ trở thành hit",
    "unsure": "Tôi chưa rõ",
}


def save_feedback(surface: str, answer: str) -> None:
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(FEEDBACK_PATH)) as connection, connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS prediction_feedback (
                submitted_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                surface TEXT NOT NULL,
                answer TEXT NOT NULL
            )
        """)
        connection.execute(
            "INSERT INTO prediction_feedback (surface, answer) VALUES (?, ?)",
            (surface, answer),
        )


def render_prediction_feedback(st, surface: str) -> None:
    submitted_key = f"percentile_feedback_{surface}_submitted"
    with st.expander("Góp ý về cách hiển thị Top X%"):
        if st.session_state.get(submitted_key):
            st.success("Đã ghi nhận phản hồi. Cảm ơn bạn!")
            st.caption("Top X% cho biết vị trí của điểm dự đoán trong mẫu tham chiếu của thị trường.")
            return

        answer = st.radio(
            "Theo bạn, Top X% đang đo điều gì?",
            options=tuple(ANSWERS),
            format_func=ANSWERS.__getitem__,
            index=None,
            key=f"percentile_feedback_{surface}_answer",
        )
        if st.button("Gửi phản hồi", key=f"percentile_feedback_{surface}_send", disabled=answer is None):
            try:
                save_feedback(surface, answer)
            except (OSError, sqlite3.Error):
                st.error("Chưa lưu được phản hồi. Hãy thử lại.")
                return
            st.session_state[submitted_key] = True
            st.rerun()
