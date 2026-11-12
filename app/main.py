"""
app/main.py
Main Streamlit Application Framework
Project: What Makes a Hit Song? (Comparing Chart Performance: VN, US, KR)
Owner: Member 4 (Data Pipeline, Supabase, Benchmark & Deploy Lead)
"""

import os
import sys
from pathlib import Path
import streamlit as st

# Setup Root Directory
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.supabase_client import get_supabase_credentials

from app.components.tab_diagnostics import render_tab_diagnostics
from app.components.tab_showcase import render as render_tab_showcase
from app.components.tab_simulator import render as render_tab_simulator
from app.components.tab_upload import render as render_tab_upload

# Page Configuration
st.set_page_config(
    page_title="What Makes a Hit Song? | Multimodal Analysis",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header Section
st.markdown('<div class="main-header">🎵 What Makes a Hit Song?</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Hệ thống Phân tích & Giả lập Hiệu suất Bảng Xếp Hạng Đa Phương Thức (Việt Nam — Hoa Kỳ — Hàn Quốc, 2021–2026)</div>',
    unsafe_allow_html=True
)

# Sidebar Information & Status
with st.sidebar:
    st.title("Thông Tin Dự Án")
    st.markdown("**Học phần:** ADY201m — AI, Data Science with Python & SQL")
    st.markdown("**Đề tài:** Multimodal Hit Song Analysis")
    st.markdown("**Quy mô mẫu:** 2,098 bản ghi phân tích sạch (1,912 bài hát độc bản qua 23 quý)")

    st.markdown("---")
    st.subheader("Trạng Thái Kết Nối")
    url, key = get_supabase_credentials()
    if url and key:
        st.info("Đã cấu hình Supabase. Showcase có thể dùng cache local nếu truy vấn thất bại.")
    else:
        st.info("🟡 Chế độ Offline Local Cache\n(Đầy đủ 2,098 bản ghi sạch)")

    st.markdown("---")
    st.caption("© 2026 Nhóm Nghiên Cứu ADY201m — All Rights Reserved.")

# Main Navigation Tabs (Checkpoint 3 Scaffolding & Integration Placeholders)
tabs = st.tabs([
    "🎵 1. Thư Viện Showcase",
    "🎛️ 2. Giả Lập Hit Song",
    "🎙️ 3. Upload Âm Thanh",
    "📊 4. Chẩn Đoán & Benchmark"
])

with tabs[0]:
    render_tab_showcase()

with tabs[1]:
    render_tab_simulator()

with tabs[2]:
    render_tab_upload()

with tabs[3]:
    render_tab_diagnostics()
