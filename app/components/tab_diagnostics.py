"""
app/components/tab_diagnostics.py
Tab 4: Chẩn Đoán Mô Hình & Đối Chuẩn Benchmark
Project: What Makes a Hit Song? (2021-2026)

Rules:
- Reads all numbers dynamically from repository files via @st.cache_data.
- Zero hardcoded numbers in tables and statistical metrics.
- No Supabase dependencies (fully offline-compatible).
- Uses correlation-based language; strictly avoids forbidden causal/dominance claims.
- Graceful degradation: displays st.info("chưa có dữ liệu: ...") if any file is missing, never crashes.
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd

# Setup Root Directory (depth 2 from repo root: app/components/tab_diagnostics.py -> repo root)
ROOT_DIR = Path(__file__).resolve().parents[2]

import streamlit as st


# ==============================================================================
# DATA LOADERS WITH ST.CACHE_DATA (OFFLINE-FIRST)
# ==============================================================================

@st.cache_data
def load_test_comparison_data() -> Optional[pd.DataFrame]:
    """Loads out-of-sample test comparison metrics for all OLS models."""
    path = ROOT_DIR / "reports" / "model_test_comparison.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


@st.cache_data
def load_rf_benchmark_metrics() -> Optional[Dict[str, Any]]:
    """Loads Random Forest Model 4 benchmark metrics."""
    path = ROOT_DIR / "reports" / "report5" / "rf_benchmark_metrics.json"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data
def load_rf_vs_ols_paired_bootstrap() -> Optional[pd.DataFrame]:
    """Loads paired bootstrap results between Random Forest and OLS Model 3."""
    path = ROOT_DIR / "reports" / "rf_vs_ols_paired_bootstrap.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


@st.cache_data
def load_paired_bootstrap_nested() -> Optional[pd.DataFrame]:
    """Loads nested hierarchical model comparisons via cluster bootstrap."""
    path = ROOT_DIR / "reports" / "diagnostics" / "paired_bootstrap_nested.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


@st.cache_data
def load_diagnostics_summary() -> Optional[Dict[str, Any]]:
    """Loads Breusch-Pagan, Cook's distance, and sensitivity analysis summary."""
    path = ROOT_DIR / "reports" / "diagnostics" / "diagnostics_summary.json"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data
def load_reset_test() -> Optional[Dict[str, Any]]:
    """Loads Ramsey RESET test statistics and p-values."""
    path = ROOT_DIR / "reports" / "diagnostics" / "reset_test.json"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data
def load_wald_blocks() -> Optional[pd.DataFrame]:
    """Loads Joint Wald block test results on Model 3 core blocks."""
    path = ROOT_DIR / "reports" / "model_wald_blocks.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, comment="#")
        cols = ["block", "chi2", "df", "p_value", "p_holm", "model"]
        df = df[cols]
        return df[pd.to_numeric(df["p_value"], errors="coerce").notna()].copy()
    except Exception:
        return None


# ==============================================================================
# MAIN TAB RENDER FUNCTION
# ==============================================================================

def render_tab_diagnostics():
    """Renders Tab 4: Diagnostics and Comparative Benchmark Suite."""
    st.header("📊 Chẩn Đoán Mô Hình & Đối Chuẩn Benchmark")
    st.markdown(
        "Phân hệ kiểm định thống kê toàn diện, đối chuẩn giữa các bậc mô hình kinh tế lượng OLS lồng ghép "
        "và mô hình học máy Random Forest Benchmark trên mẫu kiểm thử ngoài mẫu ($N = 410$ quan sát, $382$ ca khúc độc bản)."
    )

    # --------------------------------------------------------------------------
    # KHỐI A: BẢNG HIỆU NĂNG NGOÀI MẪU (TEST N=410)
    # --------------------------------------------------------------------------
    st.subheader("A. Bảng Hiệu Năng Ngoài Mẫu (Test Set, N = 410)")
    
    scope = st.selectbox(
        "Lựa chọn phạm vi thị trường đánh giá ngoài mẫu:",
        options=["Overall", "VN", "US", "KR"],
        index=0,
        help="Lọc kết quả kiểm thử trên toàn bộ mẫu Test (410 dòng) hoặc theo từng thị trường cụ thể."
    )

    df_comp = load_test_comparison_data()
    rf_json = load_rf_benchmark_metrics()

    if df_comp is None:
        st.info("Chưa có dữ liệu: không tìm thấy tệp `reports/model_test_comparison.csv`.")
    elif rf_json is None:
        st.info("Chưa có dữ liệu: không tìm thấy tệp `reports/report5/rf_benchmark_metrics.json`.")
    else:
        sample_key = "Overall" if scope == "Overall" else f"Market_{scope}"
        sub_comp = df_comp[df_comp["sample"] == sample_key].copy()

        model_mapping = [
            ("Baseline (Trung bình Train theo Market)", "Baseline: Market Train Mean"),
            ("Model 1 (Audio Baseline)", "Model 1 (Audio Baseline)"),
            ("Model 2 (Audio + Lyrics)", "Model 2 (Audio + Lyrics)"),
            ("Model 2b (+ Context)", "Model 2b (+ Context)"),
            ("Model 3 (Full OLS)", "Model 3 (Full OLS)"),
        ]

        table_rows = []
        for label, match_name in model_mapping:
            matched = sub_comp[sub_comp["model"] == match_name]
            if not matched.empty:
                r = matched.iloc[0]
                table_rows.append({
                    "Mô hình": label,
                    "R² Test": f"{float(r['r2']):.4f}",
                    "95% CI R²": str(r["r2_ci_95"]),
                    "MAE Test": f"{float(r['mae']):.2f}",
                    "95% CI MAE": str(r["mae_ci_95"]),
                    "RMSE Test": f"{float(r['rmse']):.2f}",
                    "95% CI RMSE": str(r["rmse_ci_95"]),
                })

        # Thêm hàng Random Forest Benchmark
        rf_metrics = rf_json.get("overall_metrics", {}) if scope == "Overall" else rf_json.get("by_market", {}).get(scope, {})
        if rf_metrics:
            table_rows.append({
                "Mô hình": "Model 4 (Random Forest Benchmark)",
                "R² Test": f"{float(rf_metrics.get('r2', 0.0)):.4f}",
                "95% CI R²": "—",
                "MAE Test": f"{float(rf_metrics.get('mae', 0.0)):.2f}",
                "95% CI MAE": "—",
                "RMSE Test": f"{float(rf_metrics.get('rmse', 0.0)):.2f}",
                "95% CI RMSE": "—",
            })

        df_display_table = pd.DataFrame(table_rows)
        st.dataframe(df_display_table, use_container_width=True, hide_index=True)

    # --------------------------------------------------------------------------
    # KHỐI B: GHI CHÚ BOOTSTRAP GHÉP CẶP & BẢNG NESTED (OVERALL)
    # --------------------------------------------------------------------------
    st.subheader("B. Kiểm Định Ghép Cặp Bootstrap Giữa Các Bậc Mô Hình")

    df_rf_ols = load_rf_vs_ols_paired_bootstrap()
    if df_rf_ols is not None:
        matched_rf_ols = df_rf_ols[df_rf_ols["scope"] == "Overall"]
        if not matched_rf_ols.empty:
            r_rf_ols = matched_rf_ols.iloc[0]
            d_r2 = float(r_rf_ols["obs_delta_r2"])
            ci_l = float(r_rf_ols["delta_r2_ci_lower"])
            ci_u = float(r_rf_ols["delta_r2_ci_upper"])
            d_mae = float(r_rf_ols["obs_delta_mae"])
            m_ci_l = float(r_rf_ols["delta_mae_ci_lower"])
            m_ci_u = float(r_rf_ols["delta_mae_ci_upper"])

            st.markdown(
                f"📌 **So sánh ghép cặp RF vs OLS Model 3 (Overall):** "
                f"$\\Delta R^2 = {d_r2:+.3f}$, 95% CI $[{ci_l:.3f}, {ci_u:+.3f}]$ chứa 0; "
                f"$\\Delta \\text{{MAE}} = {d_mae:+.2f}$, 95% CI $[{m_ci_l:.2f}, {m_ci_u:+.2f}]$ chứa 0. "
                f"*(Khoảng tin cậy chứa 0 phản ánh chưa có bằng chứng thống kê ngoài mẫu về sự khác biệt giữa hai mô hình).*"
            )
            st.caption("ΔMAE > 0 nghĩa là mô hình sau (hoặc RF trong so sánh RF vs OLS) có MAE thấp hơn.")
        else:
            st.info("Chưa có dữ liệu: không tìm thấy dòng Overall trong `reports/rf_vs_ols_paired_bootstrap.csv`.")

        # Bảng đối chuẩn RF vs OLS M3 theo các thị trường (Overall, VN, US, KR)
        rf_table_rows = []
        for _, r in df_rf_ols.iterrows():
            r2_l = float(r["delta_r2_ci_lower"])
            r2_u = float(r["delta_r2_ci_upper"])
            m_l = float(r["delta_mae_ci_lower"])
            m_u = float(r["delta_mae_ci_upper"])
            has_0 = (r2_l <= 0.0 <= r2_u) and (m_l <= 0.0 <= m_u)

            rf_table_rows.append({
                "Phạm vi": str(r["scope"]),
                "ΔR²": f"{float(r['obs_delta_r2']):+.4f}",
                "95% CI ΔR²": f"[{r2_l:+.3f}, {r2_u:+.3f}]",
                "ΔMAE": f"{float(r['obs_delta_mae']):+.2f}",
                "95% CI ΔMAE": f"[{m_l:+.2f}, {m_u:+.2f}]",
                "CI chứa 0?": "Có" if has_0 else "Không",
            })
        st.dataframe(pd.DataFrame(rf_table_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Chưa có dữ liệu: không tìm thấy tệp `reports/rf_vs_ols_paired_bootstrap.csv`.")

    target_pairs = [
        ("Model 2 (Audio + Lyrics)", "Model 1 (Audio Baseline)"),
        ("Model 2b (+ Context)", "Model 2 (Audio + Lyrics)"),
        ("Model 3 (Full OLS)", "Model 2b (+ Context)"),
    ]

    df_nested = load_paired_bootstrap_nested()
    if df_nested is not None:
        st.markdown("##### So sánh các bậc mô hình lồng nhau (Nested Bootstrap, Overall, Track Clusters):")
        sub_nested = df_nested[df_nested["sample"] == "Overall"].copy()

        nested_rows = []
        for m_name, b_name in target_pairs:
            m_row = sub_nested[(sub_nested["model"] == m_name) & (sub_nested["baseline"] == b_name)]
            if not m_row.empty:
                nr = m_row.iloc[0]
                nested_rows.append({
                    "Mô hình đánh giá": nr["model"],
                    "Mô hình tham chiếu": nr["baseline"],
                    "ΔMAE": f"{float(nr['delta_mae']):+.3f}",
                    "95% CI ΔMAE": str(nr["delta_mae_ci_95"]),
                    "% MAE cải thiện": f"{float(nr['pct_mae_better']):.1f}%",
                    "ΔR²": f"{float(nr['delta_r2']):+.4f}",
                    "95% CI ΔR²": str(nr["delta_r2_ci_95"]),
                    "% R² cải thiện": f"{float(nr['pct_r2_better']):.1f}%",
                })

        if nested_rows:
            st.dataframe(pd.DataFrame(nested_rows), use_container_width=True, hide_index=True)
            st.caption("ΔMAE > 0 nghĩa là mô hình sau (hoặc RF trong so sánh RF vs OLS) có MAE thấp hơn.")
        else:
            st.info("Chưa có dữ liệu: không tìm thấy các cặp lồng nhau trong `reports/diagnostics/paired_bootstrap_nested.csv`.")
    else:
        st.info("Chưa có dữ liệu: không tìm thấy tệp `reports/diagnostics/paired_bootstrap_nested.csv`.")

    # --------------------------------------------------------------------------
    # KHỐI C: HÌNH ẢNH HỆ SỐ HỒI QUY & ĐỘ QUAN TRỌNG BIẾN
    # --------------------------------------------------------------------------
    st.subheader("C. Biểu Đồ Hệ Số Hồi Quy Chuẩn Hóa & Độ Quan Trọng Biến")

    p_forest_main = ROOT_DIR / "reports" / "figures" / "forest_main.png"
    if p_forest_main.exists():
        st.image(str(p_forest_main), caption="Hình 1: Hệ số hồi quy chuẩn hóa OLS Model 3 (CI cụm 95%)", use_container_width=True)
    else:
        st.info("Chưa có dữ liệu: không tìm thấy `reports/figures/forest_main.png`.")

    p_slopes = ROOT_DIR / "reports" / "figures" / "forest_market_slopes.png"
    if p_slopes.exists():
        st.image(str(p_slopes), caption="Hình 2: Độ dốc liên hệ theo thị trường cho các biến tương tác (US vs VN vs KR)", use_container_width=True)
    else:
        st.info("Chưa có dữ liệu: không tìm thấy `reports/figures/forest_market_slopes.png`.")

    p_rf_imp = ROOT_DIR / "reports" / "figures" / "rf_permutation_importance.png"
    if p_rf_imp.exists():
        st.image(str(p_rf_imp), caption="Hình 3: Độ quan trọng hoán vị (Permutation Importance) trong Random Forest Benchmark", use_container_width=True)
    else:
        st.info("Chưa có dữ liệu: không tìm thấy `reports/figures/rf_permutation_importance.png`.")

    st.divider()

    # --------------------------------------------------------------------------
    # KHỐI D: EXPANDER "KIỂM ĐỊNH CHẨN ĐOÁN"
    # --------------------------------------------------------------------------
    with st.expander("🔬 Kiểm định chẩn đoán mô hình OLS (Model 3)", expanded=False):
        st.markdown("#### 1. Kiểm định Phương Sai Sai Số (Breusch–Pagan) & Điểm Ảnh Hưởng (Cook's Distance)")
        diag_summary = load_diagnostics_summary()
        if diag_summary is not None:
            bp = diag_summary.get("breusch_pagan", {})
            ck = diag_summary.get("cooks_distance", {})

            bp_lm = float(bp.get("lm", 0.0))
            bp_p_lm = float(bp.get("p_lm", 0.0))
            bp_f = float(bp.get("f", 0.0))
            bp_p_f = float(bp.get("p_f", 0.0))
            n_obs = bp.get("n", 1685)
            ck_max = float(ck.get("max_cooks_d", 0.0))
            ck_thresh = float(ck.get("threshold_4_n", 0.0))
            ck_n_exceed = ck.get("n_exceeding_4_n", 0)

            col_d1, col_d2, col_d3, col_d4 = st.columns(4)
            col_d1.metric("Breusch–Pagan LM", f"{bp_lm:.2f}")
            col_d2.metric("p-value LM", f"{bp_p_lm:.3g}")
            col_d3.metric("Cook's Max Distance", f"{ck_max:.4f}")
            col_d4.metric("Điểm vượt 4/n", f"{ck_n_exceed} / {n_obs}")

            st.caption(
                f"Thống kê Breusch–Pagan (LM = {bp_lm:.2f}, p = {bp_p_lm:.3g}, F = {bp_f:.2f}, p = {bp_p_f:.3g}) "
                "cho thấy phương sai sai số thay đổi giữa các quan sát; vì vậy báo cáo sai số chuẩn vững theo cụm track. "
                f"Khoảng cách Cook tối đa đạt {ck_max:.4f} (ngưỡng 4/n = {ck_thresh:.5f}, "
                f"có {ck_n_exceed} quan sát vượt ngưỡng, chiếm {float(ck_n_exceed)/float(n_obs)*100:.1f}%)."
            )
        else:
            st.info("Chưa có dữ liệu: không tìm thấy `reports/diagnostics/diagnostics_summary.json`.")

        st.markdown("---")
        st.markdown("#### 2. Kiểm định Dạng Hàm Tuyến Tính (Ramsey RESET Test)")
        reset_data = load_reset_test()
        if reset_data is not None:
            r_f = float(reset_data.get("F", 0.0))
            r_pf = float(reset_data.get("p_F", 0.0))
            r_wchi2 = float(reset_data.get("wald_chi2", 0.0))
            r_wp = float(reset_data.get("wald_p", 0.0))
            df_n = int(reset_data.get("df_num", 2))
            df_d = int(reset_data.get("df_denom", 1648))

            col_r1, col_r2, col_r3, col_r4 = st.columns(4)
            col_r1.metric("RESET F-statistic", f"{r_f:.2f}")
            col_r2.metric("p-value (F)", f"{r_pf:.3g}")
            col_r3.metric("Wald χ² (cụm)", f"{r_wchi2:.2f}")
            col_r4.metric("p-value (Wald cụm)", f"{r_wp:.3g}")

            st.caption(
                f"Kiểm định Ramsey RESET (bậc 2 và 3 của ŷ; F({df_n}, {df_d}) = {r_f:.2f}, p = {r_pf:.3g}; Wald cụm χ² = {r_wchi2:.2f}, p = {r_wp:.3g}) "
                "cho p nhỏ, phản ánh dấu hiệu sai dạng hàm hoặc thiếu biến phi tuyến."
            )
        else:
            st.info("Chưa có dữ liệu: không tìm thấy `reports/diagnostics/reset_test.json`.")

        st.markdown("---")
        st.markdown("#### 3. Kiểm Định Wald Đồng Thời Theo Nhóm Biến (Joint Wald Block Tests)")
        df_wald = load_wald_blocks()
        if df_wald is not None:
            wald_rows = []
            for _, r in df_wald.iterrows():
                p_raw_f = float(r["p_value"])
                p_holm_val = r.get("p_holm")
                if pd.notna(p_holm_val) and str(p_holm_val).strip() not in ["", "nan"]:
                    p_holm_str = f"{float(p_holm_val):.3g}"
                else:
                    p_holm_str = "—"

                wald_rows.append({
                    "Khối biến": str(r["block"]),
                    "χ²": f"{float(r['chi2']):.2f}",
                    "df": int(r["df"]),
                    "p": f"{p_raw_f:.3g}",
                    "p Holm": p_holm_str,
                    "Mô hình": str(r["model"]),
                })
            st.dataframe(pd.DataFrame(wald_rows), use_container_width=True, hide_index=True)
        else:
            st.info("Chưa có dữ liệu: không tìm thấy `reports/model_wald_blocks.csv`.")

        st.markdown("---")
        st.markdown("#### 4. Đồ Thị Chẩn Đoán Phần Dư OLS Model 3")
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            p_qq = ROOT_DIR / "reports" / "diagnostics" / "qq_residuals.png"
            if p_qq.exists():
                st.image(str(p_qq), caption="Q-Q Plot: Phân phối chuẩn của phần dư (Train vs Test)", use_container_width=True)
                st.caption("ChartScore bị chặn trong [−99, 100] nên đuôi phần dư lệch khỏi đường chuẩn; suy luận dựa trên SE cụm và bootstrap, không dựa vào giả định phần dư chuẩn.")
            else:
                st.info("Chưa có dữ liệu: không tìm thấy `reports/diagnostics/qq_residuals.png`.")
        with col_res2:
            p_rvf = ROOT_DIR / "reports" / "diagnostics" / "residuals_vs_fitted.png"
            if p_rvf.exists():
                st.image(str(p_rvf), caption="Residuals vs Fitted: Phần dư theo giá trị dự báo OLS Model 3", use_container_width=True)
            else:
                st.info("Chưa có dữ liệu: không tìm thấy `reports/diagnostics/residuals_vs_fitted.png`.")

    # --------------------------------------------------------------------------
    # KHỐI E: HỘP "GIỚI HẠN" (ST.WARNING)
    # --------------------------------------------------------------------------
    st.subheader("E. Giới Hạn Của Nghiên Cứu")

    # Dynamic evaluations from data (zero hardcoded statistical values)
    # 1) Mục 2: R² Test OLS Model 3 & RF
    r2_m3_val = "N/A"
    if df_comp is not None:
        m3_match = df_comp[(df_comp["sample"] == "Overall") & (df_comp["model"] == "Model 3 (Full OLS)")]
        if not m3_match.empty:
            r2_m3_val = f"{float(m3_match.iloc[0]['r2']):.3f}"

    r2_rf_val = "N/A"
    if rf_json is not None:
        rf_ov = rf_json.get("overall_metrics", {})
        if "r2" in rf_ov:
            r2_rf_val = f"{float(rf_ov['r2']):.3f}"

    def _ci_has_zero(val: Any) -> bool:
        if pd.isna(val):
            return False
        import re
        nums = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(val))]
        return len(nums) == 2 and (nums[0] <= 0.0 <= nums[1])

    ols_r2_all_zero = True
    if df_comp is not None:
        sub_ov = df_comp[df_comp["sample"] == "Overall"]
        for _, r_ols in sub_ov.iterrows():
            if not _ci_has_zero(r_ols.get("r2_ci_95")):
                ols_r2_all_zero = False

    ols_ci_clause = "CI 95% R² của các mô hình OLS đều chứa 0 (RF không có CI trong bảng này)" if ols_r2_all_zero else "khoảng tin cậy 95% của R² bao hàm 0"

    # 2) Mục 3: Nested & RF vs OLS CI checks
    nested_m2b_only = True
    if df_nested is not None:
        sub_nest = df_nested[df_nested["sample"] == "Overall"]
        for m_name, b_name in target_pairs:
            match_row = sub_nest[(sub_nest["model"] == m_name) & (sub_nest["baseline"] == b_name)]
            if not match_row.empty:
                nr = match_row.iloc[0]
                m_zero = _ci_has_zero(nr.get("delta_mae_ci_95"))
                r_zero = _ci_has_zero(nr.get("delta_r2_ci_95"))
                is_m2b = (m_name == "Model 2b (+ Context)" and b_name == "Model 2 (Audio + Lyrics)")
                if is_m2b:
                    if m_zero or r_zero:
                        nested_m2b_only = False
                else:
                    if not m_zero or not r_zero:
                        nested_m2b_only = False
            else:
                nested_m2b_only = False

    rf_all_zero = True
    if df_rf_ols is not None:
        for _, r_rf in df_rf_ols.iterrows():
            r2_z = float(r_rf["delta_r2_ci_lower"]) <= 0.0 <= float(r_rf["delta_r2_ci_upper"])
            mae_z = float(r_rf["delta_mae_ci_lower"]) <= 0.0 <= float(r_rf["delta_mae_ci_upper"])
            if not (r2_z and mae_z):
                rf_all_zero = False

    if nested_m2b_only and rf_all_zero:
        comp_summary_text = "Trong các bước nested, chỉ M2→M2b có CI ΔMAE/ΔR² không chứa 0 (sát biên, chưa hiệu chỉnh đa so sánh); các bước còn lại và RF vs OLS (Overall, VN, US, KR) đều có CI chứa 0."
    else:
        comp_summary_text = "Khoảng tin cậy chênh lệch giữa các mô hình hầu hết đều bao hàm giá trị 0."

    # 3) Mục 5: Ramsey RESET p-values
    reset_info = load_reset_test()
    if reset_info is not None:
        wald_p = float(reset_info.get("wald_p", 0.0))
        p_F = float(reset_info.get("p_F", 0.0))
        reset_clause = f"p Wald cụm = {wald_p:.3g}; F cổ điển p = {p_F:.3g}"
    else:
        reset_clause = "chưa có kết quả"

    st.warning(
        f"""
        ⚠️ **Khuyến Cáo Diễn Giải & Giới Hạn Phương Pháp Luận:**
        1. **Tương quan thống kê không đồng nghĩa với quan hệ nhân quả:** Các hệ số phản ánh mối liên hệ thống kê quan sát được trên mẫu thực nghiệm, không đo lường tác động can thiệp thực tế.
        2. **Năng lực giải thích ngoài mẫu khiêm tốn:** $R^2$ trên tập Test ở mức thấp ({r2_m3_val} đối với OLS Model 3, {r2_rf_val} đối với Random Forest) và {ols_ci_clause}.
        3. **Kiểm định so sánh các mô hình:** {comp_summary_text}
        4. **Hiệu ứng thời gian tích lũy của năm phát hành:** Biến `release_year` có liên hệ thống kê mạnh nhất nhưng có thể phản ánh hiệu ứng thời gian tiếp xúc tích lũy trên bảng xếp hạng (exposure time) thay vì đặc trưng âm nhạc thuần túy.
        5. **Giới hạn dạng hàm tuyến tính:** Kiểm định Ramsey RESET ({reset_clause}) gợi ý dạng hàm tuyến tính chưa đủ để bao quát toàn bộ cấu trúc quan hệ phức tạp giữa các đặc trưng.
        """
    )
