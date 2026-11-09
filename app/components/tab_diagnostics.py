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

import io
import json
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import streamlit as st

# Setup Root Directory (depth 2 from repo root: app/components/tab_diagnostics.py -> repo root)
ROOT_DIR = Path(__file__).resolve().parents[2]


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
    """Loads Joint Wald block test results on Model 3 core blocks (first section only)."""
    path = ROOT_DIR / "reports" / "model_wald_blocks.csv"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        section1_lines = []
        found_first_header = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#"):
                if found_first_header:
                    break
                continue
            if stripped:
                found_first_header = True
            section1_lines.append(line)

        if not section1_lines:
            return None

        df = pd.read_csv(io.StringIO("".join(section1_lines)))
        cols = ["block", "chi2", "df", "p_value", "p_holm", "model"]
        if not all(col in df.columns for col in cols):
            return None
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
                    "R² Test": f"{float(r['r2']):.4f}" if pd.notna(r.get("r2")) else "—",
                    "95% CI R²": str(r["r2_ci_95"]) if pd.notna(r.get("r2_ci_95")) else "—",
                    "MAE Test": f"{float(r['mae']):.2f}" if pd.notna(r.get("mae")) else "—",
                    "95% CI MAE": str(r["mae_ci_95"]) if pd.notna(r.get("mae_ci_95")) else "—",
                    "RMSE Test": f"{float(r['rmse']):.2f}" if pd.notna(r.get("rmse")) else "—",
                    "95% CI RMSE": str(r["rmse_ci_95"]) if pd.notna(r.get("rmse_ci_95")) else "—",
                })

        # Thêm hàng Random Forest Benchmark (không hardcode 0.0 nếu thiếu trường)
        rf_metrics = rf_json.get("overall_metrics", {}) if scope == "Overall" else rf_json.get("by_market", {}).get(scope, {})
        if rf_metrics:
            r2_val = rf_metrics.get("r2")
            mae_val = rf_metrics.get("mae")
            rmse_val = rf_metrics.get("rmse")
            table_rows.append({
                "Mô hình": "Model 4 (Random Forest Benchmark)",
                "R² Test": f"{float(r2_val):.4f}" if r2_val is not None else "—",
                "95% CI R²": "—",
                "MAE Test": f"{float(mae_val):.2f}" if mae_val is not None else "—",
                "95% CI MAE": "—",
                "RMSE Test": f"{float(rmse_val):.2f}" if rmse_val is not None else "—",
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

            # Đánh giá động tính chất bao hàm 0 của khoảng tin cậy
            r2_has_zero = (ci_l <= 0.0 <= ci_u)
            mae_has_zero = (m_ci_l <= 0.0 <= m_ci_u)

            r2_ci_label = "chứa 0" if r2_has_zero else "không chứa 0"
            mae_ci_label = "chứa 0" if mae_has_zero else "không chứa 0"

            if r2_has_zero and mae_has_zero:
                interp_text = "Khoảng tin cậy chứa 0 phản ánh chưa có bằng chứng thống kê ngoài mẫu về sự khác biệt hiệu năng giữa hai mô hình."
            elif (not r2_has_zero) and (not mae_has_zero):
                interp_text = "Khoảng tin cậy không chứa 0 phản ánh sự khác biệt có ý nghĩa thống kê ngoài mẫu giữa hai mô hình."
            else:
                interp_text = "Kết quả khoảng tin cậy cho thấy sự khác biệt ngoài mẫu không đồng nhất giữa tiêu chuẩn R² và MAE."

            st.markdown(
                f"📌 **So sánh ghép cặp RF vs OLS Model 3 (Overall):** "
                f"$\\Delta R^2 = {d_r2:+.3f}$, 95% CI $[{ci_l:.3f}, {ci_u:+.3f}]$ {r2_ci_label}; "
                f"$\\Delta \\text{{MAE}} = {d_mae:+.2f}$, 95% CI $[{m_ci_l:.2f}, {m_ci_u:+.2f}]$ {mae_ci_label}. "
                f"*({interp_text})*"
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
        st.caption("ΔMAE > 0 nghĩa là mô hình sau (RF) có MAE thấp hơn mô hình trước (OLS Model 3).")
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
            st.caption("ΔMAE > 0 nghĩa là mô hình sau có MAE thấp hơn mô hình tham chiếu.")
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
            bp = diag_summary.get("breusch_pagan") or {}
            ck = diag_summary.get("cooks_distance") or {}

            bp_lm = bp.get("lm")
            bp_p_lm = bp.get("p_lm")
            bp_f = bp.get("f")
            bp_p_f = bp.get("p_f")
            n_obs = bp.get("n")

            ck_max = ck.get("max_cooks_d")
            ck_thresh = ck.get("threshold_4_n")
            ck_n_exceed = ck.get("n_exceeding_4_n")

            col_d1, col_d2, col_d3, col_d4 = st.columns(4)
            col_d1.metric("Breusch–Pagan LM", f"{float(bp_lm):.2f}" if bp_lm is not None else "—")
            col_d2.metric("p-value LM", f"{float(bp_p_lm):.3g}" if bp_p_lm is not None else "—")
            col_d3.metric("Cook's Max Distance", f"{float(ck_max):.4f}" if ck_max is not None else "—")
            if ck_n_exceed is not None and n_obs is not None:
                col_d4.metric("Điểm vượt 4/n", f"{ck_n_exceed} / {n_obs}")
            else:
                col_d4.metric("Điểm vượt 4/n", "—")

            if all(v is not None for v in [bp_lm, bp_p_lm, bp_f, bp_p_f, ck_max, ck_thresh, ck_n_exceed, n_obs]):
                pct_exceed = float(ck_n_exceed) / float(n_obs) * 100
                st.caption(
                    f"Thống kê Breusch–Pagan (LM = {float(bp_lm):.2f}, p = {float(bp_p_lm):.3g}, F = {float(bp_f):.2f}, p = {float(bp_p_f):.3g}) "
                    "cho thấy phương sai sai số thay đổi giữa các quan sát; vì vậy báo cáo sai số chuẩn vững theo cụm track (Cameron & Miller, 2015). "
                    f"Khoảng cách Cook tối đa đạt {float(ck_max):.4f} (ngưỡng 4/n = {float(ck_thresh):.5f}, "
                    f"có {ck_n_exceed} quan sát vượt ngưỡng, chiếm {pct_exceed:.1f}%)."
                )
            else:
                st.info("Chưa có đầy đủ các chỉ số Breusch–Pagan hoặc khoảng cách Cook trong tệp chẩn đoán.")
        else:
            st.info("Chưa có dữ liệu: không tìm thấy `reports/diagnostics/diagnostics_summary.json`.")

        st.markdown("---")
        st.markdown("#### 2. Kiểm định Dạng Hàm Tuyến Tính (Ramsey RESET Test)")
        reset_data = load_reset_test()
        if reset_data is not None:
            r_f = reset_data.get("F")
            r_pf = reset_data.get("p_F")
            r_wchi2 = reset_data.get("wald_chi2")
            r_wp = reset_data.get("wald_p")
            df_n = reset_data.get("df_num")
            df_d = reset_data.get("df_denom")

            col_r1, col_r2, col_r3, col_r4 = st.columns(4)
            col_r1.metric("RESET F-statistic", f"{float(r_f):.2f}" if r_f is not None else "—")
            col_r2.metric("p-value (F)", f"{float(r_pf):.3g}" if r_pf is not None else "—")
            col_r3.metric("Wald χ² (cụm)", f"{float(r_wchi2):.2f}" if r_wchi2 is not None else "—")
            col_r4.metric("p-value (Wald cụm)", f"{float(r_wp):.3g}" if r_wp is not None else "—")

            if all(v is not None for v in [r_f, r_pf, r_wchi2, r_wp, df_n, df_d]):
                st.caption(
                    f"Kiểm định Ramsey RESET (bậc 2 và 3 của ŷ; F({int(df_n)}, {int(df_d)}) = {float(r_f):.2f}, p = {float(r_pf):.3g}; Wald cụm χ² = {float(r_wchi2):.2f}, p = {float(r_wp):.3g}) "
                    "cho p nhỏ (bác bỏ giả thuyết dạng hàm tuyến tính chuẩn xác), phản ánh dấu hiệu sai dạng hàm hoặc thiếu các hiệu ứng phi tuyến."
                )
            else:
                st.info("Chưa có đầy đủ các chỉ số kiểm định Ramsey RESET trong tệp dữ liệu.")
        else:
            st.info("Chưa có dữ liệu: không tìm thấy `reports/diagnostics/reset_test.json`.")

        st.markdown("---")
        st.markdown("#### 3. Kiểm Định Wald Đồng Thời Theo Nhóm Biến (Joint Wald Block Tests)")
        df_wald = load_wald_blocks()
        if df_wald is not None and not df_wald.empty:
            wald_rows = []
            for _, r in df_wald.iterrows():
                p_raw = r.get("p_value")
                chi2_val = r.get("chi2")
                df_val = r.get("df")
                p_holm_val = r.get("p_holm")

                chi2_str = f"{float(chi2_val):.2f}" if pd.notna(chi2_val) else "—"
                df_str = f"{int(df_val)}" if pd.notna(df_val) else "—"
                p_str = f"{float(p_raw):.3g}" if pd.notna(p_raw) else "—"
                if pd.notna(p_holm_val) and str(p_holm_val).strip() not in ["", "nan"]:
                    p_holm_str = f"{float(p_holm_val):.3g}"
                else:
                    p_holm_str = "—"

                wald_rows.append({
                    "Khối biến": str(r["block"]),
                    "χ²": chi2_str,
                    "df": df_str,
                    "p": p_str,
                    "p Holm": p_holm_str,
                    "Mô hình": str(r["model"]),
                })
            st.dataframe(pd.DataFrame(wald_rows), use_container_width=True, hide_index=True)
            st.caption("Kiểm định Wald đồng thời (Cluster-Robust SE theo track) đánh giá ý nghĩa thống kê chung của từng khối đặc trưng trong Model 3 và các mô hình cơ sở.")
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
        if "r2" in rf_ov and rf_ov["r2"] is not None:
            r2_rf_val = f"{float(rf_ov['r2']):.3f}"

    def _ci_has_zero(val: Any) -> Optional[bool]:
        """
        Ternary logic:
        - True: CI có đủ 2 cận hợp lệ và bao hàm 0.
        - False: CI có đủ 2 cận hợp lệ và KHÔNG bao hàm 0.
        - None: Thiếu dữ liệu, null, hoặc không thể trích xuất đúng 2 cận.
        """
        if pd.isna(val) or val is None:
            return None
        import re
        nums = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(val))]
        if len(nums) != 2:
            return None
        return nums[0] <= 0.0 <= nums[1]

    if df_comp is None:
        ols_ci_clause = "chưa có dữ liệu so sánh ngoài mẫu để đánh giá khoảng tin cậy của R²"
    else:
        sub_ov = df_comp[df_comp["sample"] == "Overall"]
        r2_cis = [_ci_has_zero(r_ols.get("r2_ci_95")) for _, r_ols in sub_ov.iterrows()]
        if not r2_cis or any(c is None for c in r2_cis):
            ols_ci_clause = "chưa đủ cơ sở dữ liệu xác định tính bao hàm 0 của khoảng tin cậy R²"
        elif all(c is True for c in r2_cis):
            ols_ci_clause = "CI 95% R² của các mô hình OLS đều chứa 0 (RF không có CI trong bảng này)"
        else:
            ols_ci_clause = "khoảng tin cậy 95% của R² không đồng nhất bao hàm 0"

    # 2) Mục 3: Nested & RF vs OLS CI checks (hoàn toàn tự động tổng hợp từ dữ liệu)
    if df_nested is None or df_rf_ols is None:
        comp_summary_text = "Chưa đủ cơ sở dữ liệu để so sánh ghép cặp giữa các bậc mô hình."
    else:
        non_zero_nested = []
        zero_nested = []
        has_missing_data = False

        sub_nest = df_nested[df_nested["sample"] == "Overall"]
        pair_short_names = {
            ("Model 2 (Audio + Lyrics)", "Model 1 (Audio Baseline)"): "M1→M2",
            ("Model 2b (+ Context)", "Model 2 (Audio + Lyrics)"): "M2→M2b",
            ("Model 3 (Full OLS)", "Model 2b (+ Context)"): "M2b→M3",
        }
        for m_name, b_name in target_pairs:
            match_row = sub_nest[(sub_nest["model"] == m_name) & (sub_nest["baseline"] == b_name)]
            if match_row.empty:
                has_missing_data = True
                continue
            nr = match_row.iloc[0]
            m_zero = _ci_has_zero(nr.get("delta_mae_ci_95"))
            r_zero = _ci_has_zero(nr.get("delta_r2_ci_95"))
            if m_zero is None or r_zero is None:
                has_missing_data = True
                continue
            p_label = pair_short_names.get((m_name, b_name), f"{b_name}→{m_name}")
            if (m_zero is False) or (r_zero is False):
                non_zero_nested.append(p_label)
            elif (m_zero is True) and (r_zero is True):
                zero_nested.append(p_label)

        rf_zero_scopes = []
        rf_non_zero_scopes = []
        for _, r_rf in df_rf_ols.iterrows():
            sc = str(r_rf.get("scope", ""))
            r2_l = r_rf.get("delta_r2_ci_lower")
            r2_u = r_rf.get("delta_r2_ci_upper")
            mae_l = r_rf.get("delta_mae_ci_lower")
            mae_u = r_rf.get("delta_mae_ci_upper")
            if any(pd.isna(x) or x is None for x in [r2_l, r2_u, mae_l, mae_u]):
                has_missing_data = True
                continue
            try:
                r2_z = float(r2_l) <= 0.0 <= float(r2_u)
                mae_z = float(mae_l) <= 0.0 <= float(mae_u)
            except (ValueError, TypeError):
                has_missing_data = True
                continue
            if r2_z and mae_z:
                rf_zero_scopes.append(sc)
            else:
                rf_non_zero_scopes.append(sc)

        if has_missing_data:
            comp_summary_text = "Dữ liệu kiểm định bootstrap lồng nhau hoặc đối chuẩn RF chưa đầy đủ để đưa ra kết luận toàn diện."
        elif non_zero_nested and not rf_non_zero_scopes:
            comp_summary_text = (
                f"Trong các bước nested, chỉ {' và '.join(non_zero_nested)} có CI ΔMAE/ΔR² không chứa 0 "
                f"(sát biên, chưa hiệu chỉnh đa so sánh); các bước còn lại ({', '.join(zero_nested)}) "
                f"và RF vs OLS ({', '.join(rf_zero_scopes)}) đều có CI chứa 0."
            )
        elif not non_zero_nested and not rf_non_zero_scopes:
            comp_summary_text = (
                f"Toàn bộ các bước so sánh lồng nhau ({', '.join(zero_nested)}) và đối chuẩn RF vs OLS ({', '.join(rf_zero_scopes)}) "
                "đều có khoảng tin cậy 95% bao hàm giá trị 0."
            )
        else:
            comp_summary_text = (
                f"Các so sánh có khoảng tin cậy không chứa 0 gồm: {', '.join(non_zero_nested + rf_non_zero_scopes)}; "
                f"các so sánh còn lại đều có khoảng tin cậy chứa 0."
            )

    # 3) Mục 5: Ramsey RESET p-values
    reset_info = load_reset_test()
    if reset_info is not None and reset_info.get("wald_p") is not None and reset_info.get("p_F") is not None:
        wald_p = float(reset_info["wald_p"])
        p_F = float(reset_info["p_F"])
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
