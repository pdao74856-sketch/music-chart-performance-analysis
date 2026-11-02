"""
Script to generate the comprehensive Vietnamese Word Document (.docx)
for Member 2: Báo Cáo Khám Phá Dữ Liệu Âm Học & Phân Tích Chuyên Sâu EDA.
Embeds high-resolution figures, detailed statistical tables, feature selection rationales,
and rigorous acoustic analyses in Vietnamese.
"""

import os
import sys
import io

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def create_vietnamese_report():
    doc = Document()

    for section in doc.sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)

    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Calibri'
    normal_style.font.size = Pt(11)
    normal_style.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
    normal_style.paragraph_format.line_spacing = 1.2
    normal_style.paragraph_format.space_after = Pt(6)

    # Tiêu đề tài liệu
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(18)
    p_title.paragraph_format.space_after = Pt(4)
    run_title = p_title.add_run("BÁO CÁO PHÂN TÍCH KHÁM PHÁ DỮ LIỆU ÂM HỌC CHUYÊN SÂU\n(ACOUSTIC FEATURES EXPLORATORY DATA ANALYSIS)")
    run_title.bold = True
    run_title.font.size = Pt(18)
    run_title.font.color.rgb = RGBColor(0x1D, 0x35, 0x57)

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(16)
    run_sub = p_sub.add_run("Đề tài: What Makes a Hit Song? (2021–2026)\nĐối sánh 3 Thị trường: Việt Nam (VN), Hoa Kỳ (US), và Hàn Quốc (KR)\nHọc phần: ADY201m – Phân tích Dữ liệu Ứng dụng")
    run_sub.font.size = Pt(11.5)
    run_sub.italic = True
    run_sub.font.color.rgb = RGBColor(0x45, 0x7B, 0x9D)

    # Bảng thông tin Metadata
    meta_table = doc.add_table(rows=4, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_data = [
        ("Thành viên phụ trách:", "Member 2 – Trích xuất Đặc trưng Âm học & Phân tích Tín hiệu"),
        ("Nguồn dữ liệu chân thực:", "Supabase Database (public.modeling_dataset với is_sampled = True)"),
        ("Quy mô mẫu phân tích:", "2,860 lượt xếp hạng | 1,912 bài hát Spotify độc bản | Độ hoàn thiện: 100.0%"),
        ("Mục tiêu học thuật:", "Phân tích đặc trưng âm học phục vụ xây dựng mô hình Hồi quy (Regression)")
    ]
    for i, (k, v) in enumerate(meta_data):
        row = meta_table.rows[i]
        c0, c1 = row.cells[0], row.cells[1]
        c0.width = Inches(2.2)
        c1.width = Inches(4.3)
        c0.paragraphs[0].add_run(k).bold = True
        c0.paragraphs[0].runs[0].font.size = Pt(9.5)
        c1.paragraphs[0].add_run(v)
        c1.paragraphs[0].runs[0].font.size = Pt(9.5)
        set_cell_background(c0, "F1FAEE")
        set_cell_background(c1, "F8F9FA")
        set_cell_margins(c0, 60, 60, 100, 100)
        set_cell_margins(c1, 60, 60, 100, 100)

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # --- PHẦN 1 ---
    h1 = doc.add_heading(level=1)
    run_h1 = h1.add_run("1. Cơ Sở Lý Thuyết: Phương Pháp Trích Xuất & Lý Do Lựa Chọn Đặc Trưng")
    run_h1.font.color.rgb = RGBColor(0x1D, 0x35, 0x57)
    run_h1.bold = True

    doc.add_paragraph(
        "Tín hiệu âm thanh dạng sóng (raw audio waveform) ở tần số 44.1 kHz chứa hơn 2.64 triệu điểm mẫu biên độ mỗi phút. "
        "Do đó, dữ liệu sóng thô không thể đưa trực tiếp vào các mô hình học máy do vấn đề bùng nổ số chiều và nhiễu thời gian. "
        "Nhóm nghiên cứu đã chuẩn hóa tín hiệu âm thanh (44.1 kHz, mono) và phân tích trên toàn bộ thời lượng bài hát (Full Audio Track, trung bình ~207.1 giây / 3.45 phút/bài). "
        "Việc phân tích toàn bộ ca khúc cho phép nắm bắt trọn vẹn sự phát triển cấu trúc tác phẩm từ Intro, Verse, Chorus, Bridge đến Outro, "
        "đồng thời giúp các thuật toán lặp lại cấu trúc (RQA) đo lường chính xác tính chu kỳ và các đoạn điệp khúc hook lặp lại xuyên suốt toàn bài."
    )

    doc.add_heading(level=2).add_run("Bảng Từ Điển Đặc Trưng Âm Học & Lý Do Lựa Chọn (Feature Rationale)")
    table = doc.add_table(rows=1, cols=4)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["Tên Biến & Miền Âm Học", "Công Thức & Phương Pháp Xử Lý", "Đơn Vị & Miền Giá Trị", "Ý Nghĩa Cảm Thụ & Lý Do Lựa Chọn"]
    hdr_cells = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].paragraphs[0].add_run(h).bold = True
        hdr_cells[i].paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        hdr_cells[i].paragraphs[0].runs[0].font.size = Pt(10)
        set_cell_background(hdr_cells[i], "1D3557")
        set_cell_margins(hdr_cells[i], 80, 80, 100, 100)

    features_vn = [
        ("tempo_bpm\n(Nhịp điệu)", 
         "Dynamic Programming Tempogram Peak Tracking", 
         "BPM\n[58 - 235]", 
         "TẠI SAO CHỌN: Tốc độ nhịp điệu cơ bản. Quyết định sự đồng bộ vận động sinh học của cơ thể (nhún nhảy, gõ chân). Pop toàn cầu tuân thủ cấu trúc song đỉnh tối ưu."),
        
        ("beat_interval_variability\n(Nhịp điệu)", 
         "Hệ số biến thiên khoảng cách phách: CV = σ_Δt / μ_Δt", 
         "Tỷ số vô hướng\n[0.007 - 0.06]", 
         "TẠI SAO CHỌN: Đo lường độ ổn định cơ học của nhịp điệu. Cho phép phát hiện nhạc được thu âm căn chuẩn trên lưới nhịp máy tính (Quantized DAW) hay chơi mộc tự nhiên có sai số con người."),
        
        ("onset_rate\n(Mật độ nốt)", 
         "Spectral Flux Peak Picking trên các khung thời gian", 
         "Sự kiện / Giây\n[25 - 345]", 
         "TẠI SAO CHỌN: Đo mật độ nốt nhạc xuất hiện trong 1 giây. Phân biệt rõ rệt giữa bản Ballad chậm rãi với các bản K-Pop hay EDM dồn dập (hi-hat 16th-note, synth arpeggio dày đặc)."),
        
        ("rms_energy\n(Năng lượng)", 
         "Căn bậc hai trung bình bình phương biên độ: √(1/N ∑ x[n]²)", 
         "Biên độ tuyến tính\n[0.02 - 0.57]", 
         "TẠI SAO CHỌN: Đại diện cho năng lượng vật lý thực sự của âm thanh. Phản ánh mức độ nén âm (compression), độ lớn 'bốc' của bản phối khi phát trên tai nghe thương mại."),
        
        ("dynamic_range\n(Động học)", 
         "Chênh lệch phân vị năng lượng: P95(RMS_dB) - P05(RMS_dB)", 
         "Decibel (dB)\n[2.2 - 193.5]", 
         "TẠI SAO CHỌN: Đo khoảng cách giữa đoạn êm dịu nhất và đoạn cao trào bùng nổ nhất. Phản ánh 'Cuộc chiến độ lớn' (Loudness War) trong công nghiệp âm nhạc."),
        
        ("spectral_centroid\n(Âm sắc)", 
         "Trọng tâm khối lượng phổ tần số: (∑ f·|X(f)|) / (∑ |X(f)|)", 
         "Hertz (Hz)\n[544 - 3,628]", 
         "TẠI SAO CHỌN: Tương ứng trực tiếp với cảm nhận về 'Độ sáng' (Brightness) của âm sắc. Giá trị cao thể hiện âm thanh sắc nét, nhiều âm treble kim loại và giọng hát bay bổng."),
        
        ("spectral_bandwidth\n(Âm sắc)", 
         "Độ lệch chuẩn phổ quanh trọng tâm: √(∑ (f - C)²·|X(f)| / ∑ |X(f)|)", 
         "Hertz (Hz)\n[908 - 3,291]", 
         "TẠI SAO CHỌN: Đo độ dày và mức độ bao phủ phổ âm thanh. Phân biệt bản phối mộc đơn âm với bản phối thương mại đa lớp lấp đầy toàn bộ phổ nghe được."),
        
        ("audio_key & audio_mode\n(Hòa âm)", 
         "Krumhansl-Schmuckler (K-S) Key-Finding qua hệ số tương quan Pearson cực đại", 
         "Key: 12 Cao độ\nMode: Trưởng / Thứ", 
         "TẠI SAO CHỌN: Nốt chủ âm và thể thức điệu thức. Giọng Trưởng (Major) khơi gợi cảm xúc phấn chấn, lạc quan; giọng Thứ (Minor) khơi gợi sự trầm buồn, suy tư và kịch tính."),
        
        ("chord_change_rate\n(Hòa âm)", 
         "Beat-Synchronous Triad Template Matching qua khoảng cách Cosine", 
         "Hợp âm / Phút\n[12 - 87]", 
         "TẠI SAO CHỌN: Vận tốc biến đổi hòa thanh (Harmonic Rhythm). Phân biệt ca khúc chỉ dùng vòng 4 hợp âm lặp đơn giản với ca khúc sử dụng hợp âm chuyển tiếp dày đặc."),
        
        ("mfcc_repetition_determinism\n(Cấu trúc lặp)", 
         "Recurrence Quantification Analysis (RQA) trên không gian MFCC", 
         "Tỷ số [0.0 - 0.72]", 
         "TẠI SAO CHỌN: Thay thế độ tự tương đồng cũ bị bão hòa. Đo lường mức độ lặp lại cơ học của âm sắc (ví dụ: đoạn loop beat trống lặp từ đầu đến cuối trong nhạc Hip-hop/Trap)."),
        
        ("chroma_repetition_determinism\n(Cấu trúc lặp)", 
         "RQA trên không gian Chroma chuẩn hóa Cosine (độ dài dòng tối thiểu = 2)", 
         "Tỷ số [0.0 - 0.93]", 
         "TẠI SAO CHỌN: Đo lường tính chu kỳ hòa âm. Phản ánh cấu trúc đối xứng giữa các phân đoạn phiên khúc và điệp khúc trong bài hát thương mại."),
        
        ("chroma_longest_pattern_duration\n(Cấu trúc lặp)", 
         "Độ dài đường chéo dài nhất trong ma trận RQA quy đổi ra số giây", 
         "Giây (s)\n[0.0 - 72.0]", 
         "TẠI SAO CHỌN: Độ dài của đoạn lặp hòa âm dài nhất. Bắt trúng độ dài của một câu nhạc điệp khúc (Hook/Chorus) chuẩn mực 8 ô nhịp.")
    ]

    for row_idx, data in enumerate(features_vn):
        row = table.add_row()
        bg_col = "FFFFFF" if row_idx % 2 == 0 else "F8F9FA"
        for col_idx, text in enumerate(data):
            cell = row.cells[col_idx]
            cell.paragraphs[0].add_run(text)
            cell.paragraphs[0].runs[0].font.size = Pt(8.5)
            set_cell_background(cell, bg_col)
            set_cell_margins(cell, 60, 60, 80, 80)

    for row in table.rows:
        row.cells[0].width = Inches(1.5)
        row.cells[1].width = Inches(1.8)
        row.cells[2].width = Inches(1.2)
        row.cells[3].width = Inches(2.5)

    doc.add_page_break()

    # --- PHẦN 2 ---
    h2 = doc.add_heading(level=1)
    run_h2 = h2.add_run("2. Phân Tích Khám Phá Dữ Liệu (EDA) Chuyên Sâu Từng Biểu Đồ")
    run_h2.font.color.rgb = RGBColor(0x1D, 0x35, 0x57)
    run_h2.bold = True

    # FIG 1
    doc.add_heading(level=2).add_run("2.1. Hình 1: Phân Phối Nhịp Điệu và Năng Lượng Âm Học (Rhythm & Energy Dynamics)")
    fig1_path = "reports/report4/figures/member2/fig1_rhythm_energy_distribution.png"
    if os.path.exists(fig1_path):
        doc.add_picture(fig1_path, width=Inches(6.2))
        p_c = doc.add_paragraph()
        p_c.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_c = p_c.add_run("Hình 1: Phân phối Nhịp điệu và Năng lượng Âm học theo Thị trường (Violin & Boxplots, 300 DPI)")
        r_c.italic = True
        r_c.font.size = Pt(9.5)
        r_c.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    p1_shape = doc.add_paragraph()
    p1_shape.add_run("A. Mô tả Hình dạng Trực quan của Biểu đồ:\n").bold = True
    p1_shape.add_run(
        "• Bố cục tổng thể: Biểu đồ được chia thành 4 ô con (2 hàng × 2 cột), sử dụng bảng màu đại diện: Đỏ Coral (Việt Nam - VN), Xanh thép (Hoa Kỳ - US), và Xanh ngọc (Hàn Quốc - KR).\n"
        "• Ô (A) Tốc độ Nhịp điệu (Tempo BPM - Violin plot): Hình dáng của thân violin ở cả 3 nước đều phình to rõ rệt ở hai vị trí (phân phối song đỉnh - bimodal). Điểm phình thứ nhất nằm quanh vùng 90–105 BPM (tiêu biểu cho Ballad và Hip-hop chậm), và điểm phình thứ hai lớn hơn nằm ở vùng 120–130 BPM (chuẩn mực của Dance-Pop sôi động). Các đường nét đứt biểu thị trung vị bên trong violin nằm ngang nhau ở mức ~117.5–123.0 BPM.\n"
        "• Ô (B) Mật độ Nốt nhạc (Onset Rate - Boxplot): Khung hộp chữ nhật (IQR) của Hoa Kỳ (US) nằm cao nhất (từ 160 đến 235 onsets/giây), trung vị đạt 198.8 onsets/s. Hộp của Hàn Quốc bám sát phía sau (trung vị 193.7), và Việt Nam nằm thấp nhất (trung vị 189.4).\n"
        "• Ô (C) Năng lượng Sóng âm (RMS Energy - Violin plot) & Ô (D) Khoảng động học (Dynamic Range - Boxplot): Thân violin của Hàn Quốc phình to ở vùng biên độ cao (0.24–0.30), còn Việt Nam phình to ở dải thấp hơn (0.18–0.24). Ngược lại ở ô D, Việt Nam có mức động học trung bình cao nhất (15.51 dB) so với US (14.53 dB) và KR (14.48 dB)."
    )

    p1_stat = doc.add_paragraph()
    p1_stat.add_run("B. Ý nghĩa Âm học & Kiểm định Thống kê Thực nghiệm:\n").bold = True
    p1_stat.add_run(
        "1. Tính Đồng nhất của Tempo (Tempo Invariance): Kiểm định phi tham số Kruskal-Wallis cho ra giá trị H = 2.87, p = 0.238 > 0.05. Kết quả này khẳng định: KHÔNG có sự khác biệt có ý nghĩa thống kê về trung vị Tempo giữa 3 quốc gia (Mean: US 122.3, KR 120.6, VN 120.4 BPM). Về mặt sinh lý học, tốc độ nhịp điệu bài hát bị chi phối bởi cơ chế 'motor entrainment' (sự đồng bộ vận động cơ thể người), nhắm vào dải tần số bước chân và nhịp tim con người.\n"
        "2. Mật độ Nốt Dày Đặc của US và K-Pop (Onset Rate): Hoa Kỳ (US) ghi nhận mật độ nốt trung bình cao nhất (Mean = 199.1, Median = 198.8 onsets/s), theo sát bởi Hàn Quốc (Mean = 192.5, Median = 193.7) và Việt Nam (Mean = 188.2, Median = 189.4), khác biệt có ý nghĩa thống kê cao (H = 22.35, p < 0.001). Điều này phản ánh nhịp điệu ca từ và tiết tấu bộ gõ trong nhạc Pop/Hip-hop Mỹ và K-Pop có mật độ nốt dày hơn nhạc trữ tình Việt Nam.\n"
        "3. Mức độ Nén Năng lượng K-Pop (RMS Energy): RMS Energy gia tăng tịnh tiến có ý nghĩa cực mạnh (H = 103.88, p < 0.001): KR (Mean = 0.254, Median = 0.261) > US (Mean = 0.238, Median = 0.248) > VN (Mean = 0.224, Median = 0.225). K-Pop áp dụng kỹ thuật mastering compression tối đa hóa độ lớn (Loudness War) để tạo âm lượng to, bốc trên loa điện thoại và tai nghe di động."
    )

    p1_imp = doc.add_paragraph()
    p1_imp.add_run("C. Tác động đến Bài toán Hồi quy (Member 4):\n").bold = True
    p1_imp.add_run(
        "Tempo_bpm không có lực phân tách thị trường, nhưng Onset Rate và RMS Energy là hai ứng viên sáng giá mang trọng số quan trọng trong mô hình dự đoán điểm số thành tích (ChartScore)."
    )

    # FIG 2
    doc.add_heading(level=2).add_run("2.2. Hình 2: Đặc Trưng Phổ Tần Số và Độ Ổn Định Nhịp Điệu (Spectral & Stability)")
    fig2_path = "reports/report4/figures/member2/fig2_spectral_stability_distribution.png"
    if os.path.exists(fig2_path):
        doc.add_picture(fig2_path, width=Inches(6.2))
        p_c = doc.add_paragraph()
        p_c.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_c = p_c.add_run("Hình 2: Phân tích Phổ Tần số và Độ Ổn định Nhịp điệu theo Thị trường (300 DPI)")
        r_c.italic = True
        r_c.font.size = Pt(9.5)
        r_c.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    p2_shape = doc.add_paragraph()
    p2_shape.add_run("A. Mô tả Hình dạng Trực quan của Biểu đồ:\n").bold = True
    p2_shape.add_run(
        "• Bố cục: Gồm 3 bảng xếp ngang (1 hàng × 3 cột).\n"
        "• Ô (A) Độ Biến thiên Phách (Beat Variability - Boxplot): Cả 3 hộp chữ nhật của 3 thị trường bị nén dẹt sát đáy trục tung (dưới mức 0.02), râu râu whiskers rất ngắn và có rải rác các điểm chấm ngoại lai kéo dài lên phía trên.\n"
        "• Ô (B) Trọng tâm Phổ (Spectral Centroid - Violin plot): Phần thân violin phình rộng nhất của Mỹ và Hàn Quốc nằm ở vùng cao quanh 2,300 Hz, trong khi phần bụng violin của Việt Nam lại nằm tụt thấp hơn ở khoảng 2,100 Hz.\n"
        "• Ô (C) Độ rộng Phổ (Spectral Bandwidth - Violin plot): Cả 3 quốc gia có hình dáng giọt nước đồng đều, phình rộng ở khoảng 2,400–2,500 Hz."
    )

    p2_stat = doc.add_paragraph()
    p2_stat.add_run("B. Ý nghĩa Âm học & Kiểm định Thống kê Thực nghiệm:\n").bold = True
    p2_stat.add_run(
        "1. Bằng chứng Định lượng về Máy tính DAW: Hệ số biến thiên khoảng cách phách (Beat CV) cực kỳ thấp (CV < 0.02 trên toàn bộ 3 thị trường, H = 1.41, p = 0.495 > 0.05). Điều này cung cấp bằng chứng thực nghiệm vững chắc rằng: 100% các bài hit lọt vào bảng xếp hạng đều được sản xuất trên Digital Audio Workstations (DAW) với nhịp bấm máy (Quantized Metronome Grid). Sự dao động tự nhiên của con người (human rubato) đã hoàn toàn bị triệt tiêu trong pop hiện đại.\n"
        "2. Độ Sáng Âm Thanh (Spectral Centroid): Hàn Quốc (KR) có phổ sáng nhất (Mean = 2,335.3 Hz, Median = 2,359.6 Hz), cao hơn hẳn Hoa Kỳ (Mean = 2,214.9 Hz) và Việt Nam (Mean = 2,166.0 Hz) với ý nghĩa thống kê vượt trội (H = 80.28, p < 0.001). Tương tự, độ rộng phổ (Bandwidth) của KR cũng rộng nhất (Mean = 2,507.4 Hz, H = 68.73, p < 0.001). Về mặt xử lý tín hiệu, K-Pop sử dụng rất nhiều nhạc cụ điện tử dải cao và áp dụng kỹ thuật kích sáng treble để tạo độ đanh, sắc sảo. Nhạc Việt Nam giữ trọng tâm dải trung âm ấm áp hơn."
    )

    p2_imp = doc.add_paragraph()
    p2_imp.add_run("C. Tác động đến Bài toán Hồi quy (Member 4):\n").bold = True
    p2_imp.add_run(
        "Biến Beat CV có độ phương sai rất thấp nên ít mang tính phân loại, trong khi Spectral Centroid đóng vai trò thước đo quan trọng phản ánh chất lượng bản phối tiêu chuẩn quốc tế."
    )

    doc.add_page_break()

    # FIG 3
    doc.add_heading(level=2).add_run("2.3. Hình 3: Cấu Trúc Hòa Âm, Điệu Thức và Tốc Độ Đổi Hợp Âm (Tonal & Harmony)")
    fig3_path = "reports/report4/figures/member2/fig3_tonal_harmony_distribution.png"
    if os.path.exists(fig3_path):
        doc.add_picture(fig3_path, width=Inches(6.2))
        p_c = doc.add_paragraph()
        p_c.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_c = p_c.add_run("Hình 3: Cấu trúc Hòa âm, Điệu thức và Tốc độ Đổi Hợp âm theo Thị trường (300 DPI)")
        r_c.italic = True
        r_c.font.size = Pt(9.5)
        r_c.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    p3_shape = doc.add_paragraph()
    p3_shape.add_run("A. Mô tả Hình dạng Trực quan của Biểu đồ:\n").bold = True
    p3_shape.add_run(
        "• Bố cục: Gồm 3 bảng phân tích hòa âm (1 hàng × 3 cột).\n"
        "• Ô (A) Tỷ lệ Phân bố 12 Cao độ (Key % - Biểu đồ cột nhóm): 12 nốt nhạc xếp dọc trục hoành từ C đến B. Các cột màu đỏ (VN), xanh thép (US), xanh ngọc (KR) cùng dựng đứng cao vút tại các nốt C, C#, D, E (đạt 10–13% mỗi nốt), và cùng chạm đáy thấp nhất ở nốt B và A# (<4%). Chiều cao của 3 cột màu tại từng nốt gần như bằng nhau chằn chặn.\n"
        "• Ô (B) Cơ cấu Điệu thức (Major vs Minor % - Cột chồng 100%): Màu xanh đậm (Major - Giọng Trưởng) chiếm ưu thế ở cả 3 cột, đặc biệt cột của Việt Nam vươn cao nhất đạt 64.1%, tiếp theo là Hàn Quốc (58.6%) và Hoa Kỳ (56.4%). Màu cam đất (Minor - Giọng Thứ) chiếm phần còn lại.\n"
        "• Ô (C) Tốc độ Đổi Hợp âm (Chord Change Rate - Boxplot): Khung hộp chữ nhật của Hàn Quốc nằm nhỉnh hơn hẳn hai khung hộp của Mỹ và Việt Nam, với trung vị đạt 43.6 và giá trị trung bình 44.8 hợp âm/phút."
    )

    p3_stat = doc.add_paragraph()
    p3_stat.add_run("B. Ý nghĩa Âm học & Kiểm định Thống kê Thực nghiệm:\n").bold = True
    p3_stat.add_run(
        "1. Độc lập về Chủ âm (Key Independence): Kiểm định Chi bình phương độc lập xác nhận: χ² = 20.82, df = 22, p = 0.532 > 0.05. Sự lựa chọn nốt chủ âm HOÀN TOÀN ĐỘC LẬP với thị trường quốc gia. Điều này được lý giải bởi 'Công thái học nhạc cụ' (Instrument Ergonomics): các phím đàn piano và dây đàn guitar được thiết kế tự nhiên thuận tay nhất ở các nốt C, G, D, E. Nhạc sĩ sáng tác pop tại Việt Nam, Mỹ hay Hàn Quốc đều bấm các phím thuận này trước tiên.\n"
        "2. Điệu thức (Mode Distribution): Kiểm định Chi bình phương cho thấy sự khác biệt có ý nghĩa thống kê cao: χ² = 11.70, df = 2, p = 0.0029 < 0.01. Việt Nam chuộng giọng Trưởng (Major) cao nhất với 64.1% (so với KR: 58.6% và US: 56.4%), phản ánh thị hiếu đại chúng chuộng cảm xúc trong sáng, tích cực và dễ tiếp nhận.\n"
        "3. Tốc độ Hòa thanh K-Pop (Chord Change Rate): Hàn Quốc đổi hợp âm trung bình 44.8 hợp âm/phút, nhanh hơn có ý nghĩa thống kê so với US (43.8) và VN (42.1) (H = 19.50, p = 5.83 × 10⁻⁵ < 0.001). K-Pop thường sử dụng các vòng hợp âm ngắn 2 phách và chèn dày đặc các hợp âm chuyển tiếp (passing chords) để duy trì cảm giác căng thẳng và biến ảo trong ca khúc."
    )

    p3_imp = doc.add_paragraph()
    p3_imp.add_run("C. Tác động đến Bài toán Hồi quy (Member 4):\n").bold = True
    p3_imp.add_run(
        "Cần mã hóa nhị phân cho `audio_mode` (Major=1, Minor=0) và One-Hot Encoding cho `audio_key`. Biến `chord_change_rate` là đặc trưng hòa âm động quan trọng nhất để phân loại thị trường."
    )

    # FIG 4
    doc.add_heading(level=2).add_run("2.4. Hình 4: Tính Chu Kỳ và Cấu Trúc Lặp Lại qua RQA (Repetition Dynamics)")
    fig4_path = "reports/report4/figures/member2/fig4_repetition_rqa_distribution.png"
    if os.path.exists(fig4_path):
        doc.add_picture(fig4_path, width=Inches(6.2))
        p_c = doc.add_paragraph()
        p_c.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_c = p_c.add_run("Hình 4: Phân tích Tính Chu kỳ và Cấu trúc Lặp lại qua RQA (300 DPI)")
        r_c.italic = True
        r_c.font.size = Pt(9.5)
        r_c.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    p4_shape = doc.add_paragraph()
    p4_shape.add_run("A. Mô tả Hình dạng Trực quan của Biểu đồ:\n").bold = True
    p4_shape.add_run(
        "• Bố cục: Gồm 3 bảng đo lường mức độ lặp lại (1 hàng × 3 cột).\n"
        "• Ô (A) Lặp lại Hòa âm (Chroma DET - Violin plot): Phân phối dạng hình nón thuôn dài, phình nhẹ ở vùng thấp 0.08–0.15 ở cả 3 nước.\n"
        "• Ô (B) Lặp lại Âm sắc (MFCC DET - Violin plot): Cả 3 quốc gia đều có hình dáng violin tương tự nhau, tập trung dày đặc quanh mức 0.20–0.25 (trung vị ~0.21, mean ~0.22–0.23).\n"
        "• Ô (C) Độ dài Vòng Lặp Cực Đại (Chroma Longest Pattern Duration - Boxplot): Khung hộp chữ nhật co cụm thấp ở mức trung vị 10.0–11.0 giây, nhưng râu phía trên kéo dài vút lên tận 40–60 giây với hàng loạt điểm dị biệt (outliers) phía trên, thể hiện sự lệch phải cực nặng (Extreme Right Skewness)."
    )

    p4_stat = doc.add_paragraph()
    p4_stat.add_run("B. Ý nghĩa Âm học & Kiểm định Thống kê Thực nghiệm:\n").bold = True
    p4_stat.add_run(
        "1. Sự Đồng Nhất về Lặp Lại Âm Sắc (MFCC Repetition): Cả 3 thị trường đều có chỉ số MFCC DET đồng đều quanh mức 0.22–0.23 (KR: 0.230, VN: 0.227, US: 0.223). Kiểm định H = 2.95, p = 0.229 > 0.05 khẳng định KHÔNG có sự khác biệt có ý nghĩa thống kê giữa các quốc gia. Điều này chứng minh âm nhạc thương mại toàn cầu đều dựa trên triết lý sản xuất sử dụng sample loop và kết cấu lặp lại cố định làm nền móng.\n"
        "2. Hằng Số Cấu Trúc Điệp Khúc 8 Ô Nhịp (The 8-Bar Hook Rule): Độ dài đoạn lặp cực đại đạt trung vị 10.0–11.0 giây trên cả 3 nước (Mean: VN 14.16s, KR 13.82s, US 12.09s, H = 22.61, p = 1.23 × 10⁻⁵ < 0.001). Con số 10–12 giây tương ứng hoàn hảo với độ dài của một câu nhạc điệp khúc (Hook/Chorus) tiêu chuẩn 8 ô nhịp (8 bars) ở tempo 120 BPM. Cấu trúc bài hát pop thương mại toàn cầu bị giới hạn nghiêm ngặt bởi dung lượng trí nhớ âm thanh ngắn hạn (auditory short-term memory) của não người."
    )

    p4_imp = doc.add_paragraph()
    p4_imp.add_run("C. Tác động đến Bài toán Hồi quy (Member 4):\n").bold = True
    p4_imp.add_run(
        "Biến `chroma_longest_pattern_duration` có độ lệch phải rất cao (skewness > 2.5). BẮT BUỘC Member 4 phải áp dụng phép biến đổi log1p trước khi đưa vào mô hình hồi quy tuyến tính."
    )

    doc.add_page_break()

    # FIG 5
    doc.add_heading(level=2).add_run("2.5. Hình 5: Dấu Ấn Âm Học Đa Chiều Radar (Acoustic Market Fingerprints)")
    fig5_path = "reports/report4/figures/member2/fig5_audio_fingerprint_radar.png"
    if os.path.exists(fig5_path):
        doc.add_picture(fig5_path, width=Inches(5.8))
        p_c = doc.add_paragraph()
        p_c.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_c = p_c.add_run("Hình 5: So sánh Đa chiều Dấu ấn Âm học Trung bình (Polar Radar Chart, 300 DPI)")
        r_c.italic = True
        r_c.font.size = Pt(9.5)
        r_c.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    p5_shape = doc.add_paragraph()
    p5_shape.add_run("A. Mô tả Hình dạng Trực quan của Biểu đồ:\n").bold = True
    p5_shape.add_run(
        "• Bố cục: Biểu đồ mạng nhện hình học cực (Polar Radar Chart) gồm 8 trục tọa độ tỏa ra từ tâm (tương ứng với 8 đặc trưng đã được chuẩn hóa Min-Max về khoảng [0, 1]).\n"
        "• Đa giác màu xanh ngọc (Hàn Quốc - KR): Có diện tích bao phủ lớn nhất, kéo dãn phình to áp đảo ở 6/8 đỉnh: Năng lượng RMS (0.426), Trọng tâm phổ (0.581), Độ rộng phổ (0.671), Tốc độ đổi hợp âm (0.446), Lặp lại hòa âm (0.131), và Lặp lại âm sắc (0.321).\n"
        "• Đa giác màu xanh thép (Hoa Kỳ - US): Kéo nhọn vươn xa nhất ở 2 trục: Mật độ nốt (Onset Rate: 0.545) và Nhịp độ (Tempo BPM: 0.361).\n"
        "• Đa giác màu đỏ (Việt Nam - VN): Có diện tích thu nhỏ hơn ở các dải năng lượng và độ sáng, nhưng giữ vị trí cân bằng và dẫn đầu tuyệt đối về Khoảng động học (Dynamic Range: 15.51 dB)."
    )

    p5_stat = doc.add_paragraph()
    p5_stat.add_run("B. Tổng Hợp Hồ Sơ Âm Học Của 3 Thị Trường:\n").bold = True
    p5_stat.add_run(
        "1. Hàn Quốc (K-Pop) - Động Năng Tột Bậc (Kinetic Intensity): Bản phối nhiều lớp, mật độ năng lượng nén tối đa (RMS), độ sáng và độ rộng dải phổ áp đảo, biến đổi hòa thanh dồn dập nhằm kích thích thính giác mạnh mẽ.\n"
        "2. Hoa Kỳ (US-UK) - Tiết Tấu Nhanh & Mật Độ Nốt Dày (Rhythmic Drive & Fast Onset): Dẫn đầu về mật độ nốt gõ (Onset Rate) và Tempo, tôn vinh tiết tấu bộ gõ và nhịp điệu ca từ (vocal delivery).\n"
        "3. Việt Nam (V-Pop) - Khoảng Động Rộng & Trữ Tình (Dynamic Headroom & Melodic Breathing): Bản phối giữ khoảng thở tự nhiên, dải động lớn nhất (ít bị nén bẹp), tôn trọng không gian để người nghe cảm thụ ca từ và giai điệu."
    )

    # --- PHẦN 3 ---
    h3 = doc.add_heading(level=1)
    run_h3 = h3.add_run("3. Khuyến Nghị Kỹ Thuật Bàn Giao Cho Member 4 (Mô Hình Hồi Quy)")
    run_h3.font.color.rgb = RGBColor(0x1D, 0x35, 0x57)
    run_h3.bold = True

    recs_vn = [
        ("Khử Đa Cộng Tuyến (Multicollinearity):", "Cặp biến loudness_proxy và rms_energy có tương quan Pearson r > 0.95. Bắt buộc Member 4 phải loại bỏ loudness_proxy và giữ rms_energy làm biến vật lý chuẩn. Tương tự, giữ beat_interval_variability và loại bỏ tempo_stability."),
        ("Biến Đổi Dữ Liệu Lệch (Log Transformations):", "Các đặc trưng thời lượng RQA (chroma_longest_pattern_duration) bị lệch phải rất mạnh. Bắt buộc áp dụng log1p trước khi đưa vào các mô hình hồi quy tuyến tính (Ridge/Lasso)."),
        ("Chiến Lược Mã Hóa Biến Phân Loại:", "Mã hóa nhị phân audio_mode (Major = 1, Minor = 0). Áp dụng One-Hot Encoding cho audio_key."),
        ("Lựa Chọn Thuật Toán Hồi Quy:", "Do tồn tại các ngưỡng bão hòa phi tuyến (năng lượng và độ lặp lại tăng quá cao không làm tăng thêm thứ hạng), các mô hình Tree-based Ensemble (LightGBM Regressor, XGBoost, Random Forest) sẽ cho kết quả vượt trội so với Hồi quy Tuyến tính OLS thuần túy.")
    ]

    for title, text in recs_vn:
        p_r = doc.add_paragraph()
        p_r.paragraph_format.left_indent = Inches(0.2)
        p_r.add_run(f"• {title} ").bold = True
        p_r.add_run(text)

    output_path = "reports/report4/Bao_Cao_Chuyen_Sau_EDA_Dac_Trung_Am_Hoc_Member2.docx"
    try:
        doc.save(output_path)
        print(f"Đã tạo thành công file Word tiếng Việt tại: {output_path}")
    except PermissionError:
        output_path_alt = "reports/report4/Bao_Cao_Chuyen_Sau_EDA_Dac_Trung_Am_Hoc_Member2_CapNhat.docx"
        doc.save(output_path_alt)
        print(f"File gốc đang mở trong Word, đã lưu bản cập nhật tại: {output_path_alt}")

if __name__ == "__main__":
    create_vietnamese_report()
