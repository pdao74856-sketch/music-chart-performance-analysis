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
    st.image("https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?w=400&auto=format&fit=crop&q=80", use_container_width=True)
    st.title("Thông Tin Dự Án")
    st.markdown("**Học phần:** ADY201m — Applied Data Analytics")
    st.markdown("**Đề tài:** Multimodal Hit Song Analysis")
    st.markdown("**Quy mô mẫu:** 2,098 bản ghi phân tích sạch (1,912 bài hát độc bản qua 23 quý)")

    st.markdown("---")
    st.subheader("Trạng Thái Kết Nối")
    url, key = get_supabase_credentials()
    if url and key:
        st.success(f"🟢 Supabase Connected\n({url[:22]}...)")
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
    st.header("🎵 Thư Viện Bài Hát Tiêu Biểu & Radar Chart Âm Học")
    st.info("🚧 **Đang ở giai đoạn hoàn thiện Checkpoint 2.** Phân hệ này sẽ được **Member 2** tích hợp và hoàn thiện tại **Checkpoint 3** (Showcase 15 bài mẫu Test Set & Radar Chart đa chiều).")

with tabs[1]:
    st.header("🎛️ Giả Lập Sáng Tác Hit (What-If Simulator)")
    st.info("🚧 **Đang ở giai đoạn hoàn thiện Checkpoint 2.** Phân hệ này sẽ được **Member 3** tích hợp và hoàn thiện tại **Checkpoint 3** (Thanh trượt đặc trưng ca từ NLP & âm học Librosa).")

with tabs[2]:
    st.header("🎙️ Tải Lên Âm Thanh & Trích Xuất Librosa Thời Gian Thực")
    st.info("🚧 **Đang ở giai đoạn hoàn thiện Checkpoint 2.** Phân hệ này sẽ được **Member 2** tích hợp và hoàn thiện tại **Checkpoint 3** (Tải lên file MP3/WAV, trích xuất 17 đặc trưng Librosa với caching @st.cache_data và rào chắn an toàn).")

with tabs[3]:
    st.header("📊 Model Diagnostics & Comparative Benchmark")
    st.info("🚧 **Đang ở giai đoạn hoàn thiện Checkpoint 2.** Phân hệ này sẽ được **Member 1 & Member 4** tích hợp và hoàn thiện tại **Checkpoint 3** (Bảng đối chuẩn OLS Model 3 vs Random Forest Model 4).")
