"""
Script to generate the comprehensive, professional Word document (.docx)
for Member 2: Audio Feature Engineering & Exploratory Data Analysis Report.
Embeds high-resolution figures, detailed statistical tables, feature selection rationales,
and publication-grade acoustic analyses.
"""

import os
import sys
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def create_report_document():
    doc = Document()

    # Page setup - Margins 1 inch
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)

    # Styles
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Calibri'
    normal_style.font.size = Pt(11)
    normal_style.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
    normal_style.paragraph_format.line_spacing = 1.2
    normal_style.paragraph_format.space_after = Pt(6)

    # Document Title
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(18)
    p_title.paragraph_format.space_after = Pt(4)
    run_title = p_title.add_run("ACADEMIC EXPLORATORY DATA ANALYSIS REPORT\nAUDIO FEATURE EXTRACTION & ACOUSTIC PROFILING")
    run_title.bold = True
    run_title.font.size = Pt(20)
    run_title.font.color.rgb = RGBColor(0x1D, 0x35, 0x57)

    # Subtitle
    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(18)
    run_sub = p_sub.add_run("Cross-Market Analysis: Vietnam (VN), United States (US), and South Korea (KR) [2021–2026]\nModule: ADY201m – Applied Data Analytics | Project: What Makes a Hit Song?")
    run_sub.font.size = Pt(12)
    run_sub.italic = True
    run_sub.font.color.rgb = RGBColor(0x45, 0x7B, 0x9D)

    # Metadata Box
    meta_table = doc.add_table(rows=4, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_data = [
        ("Author / Primary Responsibility:", "Member 2 – Audio Feature Engineering & Acoustic Analytics"),
        ("Data Source of Truth:", "Supabase Database (public.modeling_dataset WHERE is_sampled = True)"),
        ("Sample Size & Scope:", "2,860 Chart Entries | 1,912 Unique Spotify Tracks | 100% Extraction Completeness"),
        ("Date of Publication:", "October 2026 (Academic Submission)")
    ]
    for i, (k, v) in enumerate(meta_data):
        row = meta_table.rows[i]
        c0, c1 = row.cells[0], row.cells[1]
        c0.width = Inches(2.5)
        c1.width = Inches(4.0)
        c0.paragraphs[0].add_run(k).bold = True
        c0.paragraphs[0].runs[0].font.size = Pt(9.5)
        c1.paragraphs[0].add_run(v)
        c1.paragraphs[0].runs[0].font.size = Pt(9.5)
        set_cell_background(c0, "F1FAEE")
        set_cell_background(c1, "F8F9FA")
        set_cell_margins(c0, 60, 60, 100, 100)
        set_cell_margins(c1, 60, 60, 100, 100)

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # --- SECTION 1 ---
    h1 = doc.add_heading(level=1)
    run_h1 = h1.add_run("1. Theoretical Rationale: Feature Selection & Acoustic Meanings")
    run_h1.font.color.rgb = RGBColor(0x1D, 0x35, 0x57)
    run_h1.bold = True

    p = doc.add_paragraph(
        "Commercial music analysis requires moving beyond raw waveforms into psychoacoustically grounded descriptors. "
        "Sound signals cannot be fed directly into predictive modeling due to extreme dimensionality and temporal variance. "
        "Therefore, we engineered 18 discrete, standardized acoustic features across four foundational signal processing domains: "
        "(1) Rhythm & Temporal Structure, (2) Signal Energy & Dynamics, (3) Spectral Timbre, and (4) Harmony & Structural Recurrence. "
        "All features were derived from full audio tracks (mean duration ~207.1 seconds / 3.45 minutes) at 44.1 kHz mono, "
        "capturing complete structural trajectories from intro to outro and enabling holistic recurrence pattern discovery across entire songs."
    )

    # Detailed Table of Features
    doc.add_heading(level=2).add_run("Comprehensive Acoustic Feature Dictionary & Selection Rationale")
    table = doc.add_table(rows=1, cols=4)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["Feature Name & Domain", "Signal Formula & Method", "Physical Unit & Range", "Psychoacoustic Meaning & Research Rationale"]
    hdr_cells = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].paragraphs[0].add_run(h).bold = True
        hdr_cells[i].paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        hdr_cells[i].paragraphs[0].runs[0].font.size = Pt(10)
        set_cell_background(hdr_cells[i], "1D3557")
        set_cell_margins(hdr_cells[i], 80, 80, 100, 100)

    features_data = [
        ("tempo_bpm\n(Rhythm)", 
         "Dynamic Programming Tempogram Peak Tracking", 
         "BPM\n[58 - 235]", 
         "WHY CHOSEN: Fundamental pace of rhythmic entrainment. Governs physiological motor response (dancing, heart rate matching). Bimodal pop standard."),
        
        ("beat_interval_variability\n(Rhythm)", 
         "Coefficient of Variation of Inter-Beat Intervals: CV = σ_Δt / μ_Δt", 
         "Dimensionless Ratio\n[0.007 - 0.06]", 
         "WHY CHOSEN: Quantifies mechanical timing stability vs organic human drift. Critical for detecting modern DAW quantized grid production vs acoustic live recordings."),
        
        ("onset_rate\n(Rhythm / Density)", 
         "Spectral Flux Peak Picking over STFT frames", 
         "Events / Second\n[25 - 345]", 
         "WHY CHOSEN: Note attack density. Distinguishes sparse ballad vocal delivery from kinetic, hyper-active synth and percussion arrangements (e.g., K-Pop 16th-note patterns)."),
        
        ("rms_energy\n(Energy)", 
         "Root-Mean-Square Amplitude: √(1/N ∑ x[n]²)", 
         "Linear Amplitude\n[0.02 - 0.57]", 
         "WHY CHOSEN: Primary measure of physical acoustic power. Reflects mastering compression, perceived 'loudness punch', and commercial mixing fullness."),
        
        ("dynamic_range\n(Dynamics)", 
         "Percentile Difference: P95(RMS_dB) - P05(RMS_dB)", 
         "Decibels (dB)\n[2.2 - 193.5]", 
         "WHY CHOSEN: Measures dynamic contrast between verse lull and chorus explosion. Essential for auditing the 'loudness war' vs dynamic breathing space."),
        
        ("spectral_centroid\n(Timbre)", 
         "Center of Spectral Mass: (∑ f·|X(f)|) / (∑ |X(f)|)", 
         "Hertz (Hz)\n[544 - 3,628]", 
         "WHY CHOSEN: Corresponds directly to perceived sound 'brightness' and sharp presence. High centroids reflect crisp treble excitation, metallic hi-hats, and synth leads."),
        
        ("spectral_bandwidth\n(Timbre)", 
         "Standard Deviation around Centroid: √(∑ (f - C)²·|X(f)| / ∑ |X(f)|)", 
         "Hertz (Hz)\n[908 - 3,291]", 
         "WHY CHOSEN: Measures timbral spread across the auditory frequency spectrum. Differentiates monochromatic acoustic sounds from multi-layered, wide commercial mixes."),
        
        ("audio_key & audio_mode\n(Harmony)", 
         "Krumhansl-Schmuckler (K-S) Key-Finding via Max Pearson r with Pitch Class Profiles", 
         "Key: 12 Pitch Classes\nMode: Major / Minor", 
         "WHY CHOSEN: Core tonal center and emotional valence. Major keys project uplifting, euphoric pop hooks; minor keys project melancholy, tension, and introspection."),
        
        ("chord_change_rate\n(Harmony)", 
         "Beat-Synchronous Triad Template Matching via Cosine Distance", 
         "Chords / Minute\n[12 - 87]", 
         "WHY CHOSEN: Harmonic rhythm velocity. Measures how frequently chord functions modulate, separating simple 4-bar repetitive loops from complex passing-chord arrangements."),
        
        ("mfcc_repetition_determinism\n(Structure / RQA)", 
         "Recurrence Quantification Analysis (RQA) on MFCC space: DET = (∑ l·P(l)) / (∑ l·P(l))", 
         "Bounded Ratio\n[0.0 - 0.72]", 
         "WHY CHOSEN: Replaces saturated legacy self-similarity. Measures deterministic timbral looping (e.g., persistent drum loop or sample groove repeating end-to-end)."),
        
        ("chroma_repetition_determinism\n(Structure / RQA)", 
         "RQA on Cosine-normalized Chroma space (l_min = 2)", 
         "Bounded Ratio\n[0.0 - 0.93]", 
         "WHY CHOSEN: Quantifies harmonic structural periodicity. Measures verse-chorus symmetry and the degree of chord progression recycling throughout the track."),
        
        ("chroma_longest_pattern_duration\n(Structure / RQA)", 
         "Max diagonal line length in Chroma Recurrence Matrix converted to seconds", 
         "Seconds\n[0.0 - 72.0]", 
         "WHY CHOSEN: Length of the single longest repeating harmonic phrase. Directly captures standard 8-bar musical hook / chorus duration.")
    ]

    for row_idx, data in enumerate(features_data):
        row = table.add_row()
        bg_col = "FFFFFF" if row_idx % 2 == 0 else "F8F9FA"
        for col_idx, text in enumerate(data):
            cell = row.cells[col_idx]
            cell.paragraphs[0].add_run(text)
            cell.paragraphs[0].runs[0].font.size = Pt(8.5)
            set_cell_background(cell, bg_col)
            set_cell_margins(cell, 60, 60, 80, 80)

    # Set column widths
    for row in table.rows:
        row.cells[0].width = Inches(1.5)
        row.cells[1].width = Inches(1.8)
        row.cells[2].width = Inches(1.2)
        row.cells[3].width = Inches(2.5)

    doc.add_page_break()

    # --- SECTION 2 ---
    h2 = doc.add_heading(level=1)
    run_h2 = h2.add_run("2. In-Depth Exploratory Data Analysis & Empirical Findings")
    run_h2.font.color.rgb = RGBColor(0x1D, 0x35, 0x57)
    run_h2.bold = True

    doc.add_paragraph(
        "This section presents the core empirical findings derived from 2,860 chart entries across Vietnam (VN, N=885), "
        "the United States (US, N=955), and South Korea (KR, N=1,020). Statistical inferences utilize non-parametric Kruskal-Wallis H-tests "
        "and Chi-Square tests of independence to avoid assuming normality, adhering strictly to objective acoustic and production interpretations."
    )

    # SUBSECTION: FIGURE 1
    doc.add_heading(level=2).add_run("2.1. Rhythm and Signal Energy Dynamics (Figure 1)")
    fig1_img = "reports/report4/figures/member2/fig1_rhythm_energy_distribution.png"
    if os.path.exists(fig1_img):
        doc.add_picture(fig1_img, width=Inches(6.2))
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run_cap = p_cap.add_run("Figure 1: Distribution of Rhythm and Acoustic Energy across Markets (Violin & Boxplots, 300 DPI)")
        run_cap.italic = True
        run_cap.font.size = Pt(9.5)
        run_cap.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    p1_desc = doc.add_paragraph()
    p1_desc.add_run("Empirical Observations & Statistical Tests:\n").bold = True
    p1_desc.add_run(
        "1. Tempo Invariance (Panel A): A distinct bimodal distribution is observed across all three countries, peaking at 90–105 BPM and 120–130 BPM. "
        "A Kruskal-Wallis H-test reveals NO statistically significant difference in tempo across markets (H = 2.87, p = 0.238 > 0.05; Mean: US 122.3, KR 120.6, VN 120.4 BPM; Medians: US 123.0, KR 117.5, VN 117.5 BPM).\n"
        "2. Kinetic Note Density (Panel B): The United States exhibits the highest Onset Rate (median = 198.8, mean = 199.1 onsets/sec), followed by South Korea (median = 193.7, mean = 192.5) and Vietnam (median = 189.4, mean = 188.2), with extreme statistical significance (H = 22.35, p = 1.40 × 10⁻⁵ < 0.001).\n"
        "3. RMS Energy & Dynamic Range (Panels C & D): Signal power exhibits massive divergence (H = 103.88, p = 2.77 × 10⁻²³ < 0.001): KR is highest (mean = 0.254, median = 0.261), followed by US (mean = 0.238, median = 0.248) and VN (mean = 0.224, median = 0.225). Conversely, Vietnam leads in Dynamic Range (mean = 15.51 dB vs US: 14.53 dB and KR: 14.48 dB; H = 18.12, p = 1.16 × 10⁻⁴ < 0.001)."
    )

    p1_imp = doc.add_paragraph()
    p1_imp.add_run("Acoustic Interpretation & Engineering Implications:\n").bold = True
    p1_imp.add_run(
        "The tempo invariance is grounded in biological motor entrainment: pop music globally targets a synchronized sweet spot for human heart rate and somatic movement. "
        "Conversely, onset rate and RMS divergence reflect stark production techniques: US and K-Pop feature dense 16th-note drum subdivisions and aggressive dynamic mastering compression (optimizing for high perceptual loudness on mobile earphones). "
        "Vietnamese tracks preserve acoustic headroom and vocal space with sparser arrangements and greater dynamic range. "
        "For predictive modeling, Onset Rate and RMS Energy represent premier candidates for market-level feature importance."
    )

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # SUBSECTION: FIGURE 2
    doc.add_heading(level=2).add_run("2.2. Spectral Brightness and Temporal Stability (Figure 2)")
    fig2_img = "reports/report4/figures/member2/fig2_spectral_stability_distribution.png"
    if os.path.exists(fig2_img):
        doc.add_picture(fig2_img, width=Inches(6.2))
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run_cap = p_cap.add_run("Figure 2: Spectral Characteristics and Beat Interval Stability by Market (300 DPI)")
        run_cap.italic = True
        run_cap.font.size = Pt(9.5)
        run_cap.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    p2_desc = doc.add_paragraph()
    p2_desc.add_run("Empirical Observations & Statistical Tests:\n").bold = True
    p2_desc.add_run(
        "1. Extreme Beat Regularity (Panel A): The coefficient of variation for beat intervals (CV) is compressed uniformly near zero across all three markets (KR median = 0.0172, VN = 0.0173, US = 0.0175; H = 1.41, p = 0.495 > 0.05).\n"
        "2. Spectral Centroid Brightness (Panel B): South Korea displays the brightest spectral center of mass (median = 2,359.6 Hz, mean = 2,335.3 Hz), significantly outranking the US (median = 2,211.1 Hz, mean = 2,214.9 Hz) and Vietnam (median = 2,150.3 Hz, mean = 2,166.0 Hz) (H = 80.28, p = 3.70 × 10⁻¹⁸ < 0.001).\n"
        "3. Spectral Bandwidth Fullness (Panel C): South Korea similarly exhibits the widest spectral spread (mean = 2,507.4 Hz), followed by the US (2,430.3 Hz) and Vietnam (2,409.7 Hz) (H = 68.73, p < 0.001)."
    )

    p2_imp = doc.add_paragraph()
    p2_imp.add_run("Acoustic Interpretation & Engineering Implications:\n").bold = True
    p2_imp.add_run(
        "The near-zero beat variability (< 0.02) provides quantitative proof that modern charting tracks are universally produced on Digital Audio Workstations (DAWs) snapped to quantized metronomic grids. "
        "Human timing micro-drifts (rubato) are virtually eliminated from commercial pop. "
        "The significantly higher spectral centroid and bandwidth in K-Pop indicate intentional high-frequency excitation: heavy synthesis layers, bright vocal formant boosts, and boosted high-shelf EQ (8–12 kHz). "
        "Vietnamese tracks maintain a warmer, lower midrange acoustic envelope centered on lyrical intelligibility."
    )

    doc.add_page_break()

    # SUBSECTION: FIGURE 3
    doc.add_heading(level=2).add_run("2.3. Tonal Center, Modality, and Harmonic Velocity (Figure 3)")
    fig3_img = "reports/report4/figures/member2/fig3_tonal_harmony_distribution.png"
    if os.path.exists(fig3_img):
        doc.add_picture(fig3_img, width=Inches(6.2))
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run_cap = p_cap.add_run("Figure 3: Key Distribution, Mode Ratios, and Chord Change Rates across Markets (300 DPI)")
        run_cap.italic = True
        run_cap.font.size = Pt(9.5)
        run_cap.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    p3_desc = doc.add_paragraph()
    p3_desc.add_run("Empirical Observations & Statistical Tests:\n").bold = True
    p3_desc.add_run(
        "1. Universal Key Preference (Panel A): Pitch classes C, C#, D, and E consistently dominate all markets (~10–13% each), while B and A# are least frequent (<4%). "
        "A Chi-Square Test of Independence yields χ² = 20.82 (df = 22, p = 0.532 > 0.05), confirming that musical key choice is statistically independent of market region.\n"
        "2. Major Modality Preference (Panel B): Vietnam displays the highest proportion of Major keys at 64.1%, significantly higher than South Korea (58.6%) and the United States (56.4%) (Chi-Square χ² = 11.70, df = 2, p = 0.0029 < 0.01).\n"
        "3. Chord Transition Velocity (Panel C): South Korea displays a significantly faster harmonic rhythm (mean = 44.8 chords/min, median = 43.6) compared to the US (mean = 43.8, median = 43.4) and Vietnam (mean = 42.1, median = 41.5) (H = 19.50, p = 5.83 × 10⁻⁵ < 0.001)."
    )

    p3_imp = doc.add_paragraph()
    p3_imp.add_run("Acoustic Interpretation & Engineering Implications:\n").bold = True
    p3_imp.add_run(
        "Key selection is governed by instrument ergonomics (natural key centers of piano keyboards, guitar fretboards, and DAW MIDI defaults), rather than regional culture. "
        "Vietnam's significant preference for Major modality reflects commercial listener affinity for bright, uplifting, accessible melodic themes. "
        "K-Pop's accelerated chord change rate aligns with its compositional identity: frequent two-beat substitutions and passing chords designed to maintain auditory excitement."
    )

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # SUBSECTION: FIGURE 4
    doc.add_heading(level=2).add_run("2.4. Structural Recurrence & Repetition Determinism (Figure 4)")
    fig4_img = "reports/report4/figures/member2/fig4_repetition_rqa_distribution.png"
    if os.path.exists(fig4_img):
        doc.add_picture(fig4_img, width=Inches(6.2))
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run_cap = p_cap.add_run("Figure 4: Recurrence Quantification Analysis (RQA) of Harmonic and Timbral Loops (300 DPI)")
        run_cap.italic = True
        run_cap.font.size = Pt(9.5)
        run_cap.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    p4_desc = doc.add_paragraph()
    p4_desc.add_run("Empirical Observations & Statistical Tests:\n").bold = True
    p4_desc.add_run(
        "1. Harmonic Repetition (Panel A): Chroma DET displays a uniform, tapered distribution centered at 0.08–0.15 across all three markets.\n"
        "2. Timbral Repetition Uniformity (Panel B): MFCC Recurrence Determinism hovers consistently around 0.22–0.23 across markets (KR mean = 0.230, VN = 0.227, US = 0.223, medians ~0.21; H = 2.95, p = 0.229 > 0.05), showing no statistically significant cross-market divergence.\n"
        "3. Universal 8-Bar Hook Duration (Panel C): The distribution of the longest recurring pattern duration clusters at a median of 10.0 to 11.0 seconds across all regions (KR: 11.0s, VN: 11.0s, US: 10.0s; Means: VN 14.16s, KR 13.82s, US 12.09s; H = 22.61, p = 1.23 × 10⁻⁵ < 0.001) with extreme right-skewness."
    )

    p4_imp = doc.add_paragraph()
    p4_imp.add_run("Acoustic Interpretation & Engineering Implications:\n").bold = True
    p4_imp.add_run(
        "The cross-market stability of MFCC determinism highlights a universal production philosophy: modern commercial pop across all continents relies on standardized loop beds and sample repetition. "
        "The 10–11 second median hook length precisely matches an 8-measure musical phrase at 120 BPM, bounded by the limits of human working memory for melodic recall."
    )

    doc.add_page_break()

    # SUBSECTION: FIGURE 5
    doc.add_heading(level=2).add_run("2.5. Holistic Market Fingerprints: Polar Radar Synthesis (Figure 5)")
    fig5_img = "reports/report4/figures/member2/fig5_audio_fingerprint_radar.png"
    if os.path.exists(fig5_img):
        doc.add_picture(fig5_img, width=Inches(5.8))
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run_cap = p_cap.add_run("Figure 5: Normalized Multidimensional Acoustic Fingerprints (Polar Radar Chart, 300 DPI)")
        run_cap.italic = True
        run_cap.font.size = Pt(9.5)
        run_cap.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    p5_desc = doc.add_paragraph()
    p5_desc.add_run("Holistic Profiling Synthesis:\n").bold = True
    p5_desc.add_run(
        "By normalizing all continuous metrics onto a uniform [0, 1] scale, Figure 5 encapsulates the overarching acoustic signatures:\n"
        "• South Korea (Teal Polygon): Defined by Maximum Kinetic Intensity. Leads on 6 of 8 axes: RMS Energy (0.426), Spectral Centroid (0.581), Spectral Bandwidth (0.671), Chord Change Rate (0.446), Chroma DET (0.131), and MFCC DET (0.321).\n"
        "• United States (Blue Polygon): Defined by Rhythmic Drive & Vocal Speed. Leads on 2 axes: Onset Rate (0.545) and Tempo BPM (0.361).\n"
        "• Vietnam (Red Polygon): Defined by Dynamic Headroom & Melodic Space. Dominates Dynamic Range (15.51 dB), preserving acoustic space for emotionally expressive vocal phrasing."
    )

    # --- SECTION 3 ---
    h3 = doc.add_heading(level=1)
    run_h3 = h3.add_run("3. Concrete Handover Recommendations for Member 4 (Modeling)")
    run_h3.font.color.rgb = RGBColor(0x1D, 0x35, 0x57)
    run_h3.bold = True

    recs = [
        ("Multicollinearity Elimination:", "loudness_proxy and rms_energy exhibit Pearson r > 0.95. Member 4 must drop loudness_proxy and retain rms_energy as the uncorrupted physical signal metric. Similarly, keep beat_interval_variability and drop tempo_stability."),
        ("Logarithmic Transformations:", "RQA durational features (chroma_longest_pattern_duration and mfcc_longest_pattern_duration) exhibit heavy right skewness (kurtosis > 5.0). Member 4 must apply log1p transformations before fitting regularized linear models (Ridge/Lasso)."),
        ("Categorical Encoding Strategy:", "Encode audio_mode as binary (Major = 1, Minor = 0). For audio_key, standard One-Hot Encoding is recommended, or optionally mapping onto Circle-of-Fifths 2D Cartesian coordinates (cos(2πk/12), sin(2πk/12)) to preserve harmonic proximity."),
        ("Algorithm Suitability:", "Due to non-linear saturation plateaus identified in energy and repetition, non-linear tree ensembles (LightGBM Regressor, XGBoost, Random Forest) will yield superior R² and RMSE over OLS regression.")
    ]

    for title, text in recs:
        p_r = doc.add_paragraph()
        p_r.paragraph_format.left_indent = Inches(0.2)
        p_r.add_run(f"• {title} ").bold = True
        p_r.add_run(text)

    # Save document
    output_docx_path = "reports/report4/Member2_Audio_Features_EDA_Report.docx"
    try:
        doc.save(output_docx_path)
        print(f"Successfully generated comprehensive Word report at: {output_docx_path}")
    except PermissionError:
        output_docx_alt = "reports/report4/Member2_Audio_Features_EDA_Report_Updated.docx"
        doc.save(output_docx_alt)
        print(f"File locked, saved updated version at: {output_docx_alt}")

if __name__ == "__main__":
    create_report_document()
