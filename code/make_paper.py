import os
import pandas as pd, numpy as np
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, 'outputs')
PAPER = os.path.join(BASE, 'paper')
os.makedirs(PAPER, exist_ok=True)
dfc = pd.read_csv(os.path.join(OUT, 'clusters.csv'))
per = pd.read_csv(os.path.join(OUT, 'per_state_forecast.csv'))
mp = pd.read_csv(os.path.join(BASE, 'data', 'circle_state_map.csv'))
cl = dfc.groupby('kmeans')
names = {0: 'C0 Micro-UT / very low count', 1: 'C1 Hill, NE and small States',
         2: 'C2 Mainstream mid-volume', 3: 'C3 High-volume fast-riser'}

doc = Document()
sec = doc.sections[0]
sec.top_margin = Inches(0.8); sec.bottom_margin = Inches(0.8)
sec.left_margin = Inches(0.9); sec.right_margin = Inches(0.9)
st = doc.styles['Normal']; st.font.name = 'Calibri'; st.font.size = Pt(10.5)
st.paragraph_format.space_after = Pt(4); st.paragraph_format.line_spacing = 1.08

def P(t, b=False, i=False, size=10.5, align=None, space=4):
    p = doc.add_paragraph()
    r = p.add_run(t); r.bold = b; r.italic = i; r.font.size = Pt(size)
    p.paragraph_format.space_after = Pt(space)
    if align == 'c': p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    return p

def H(t, size=13):
    p = doc.add_heading(t, level=1)
    for r in p.runs:
        r.font.size = Pt(size); r.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)
    p.paragraph_format.space_before = Pt(10); p.paragraph_format.space_after = Pt(3)
    return p

def shade(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear'); shd.set(qn('w:color'), 'auto'); shd.set(qn('w:fill'), hexcolor)
    tcPr.append(shd)

def box(title, lines, fill='EAF2FB', title_color=RGBColor(0x1F, 0x3A, 0x5F)):
    t = doc.add_table(rows=1, cols=1)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    c = t.rows[0].cells[0]
    shade(c, fill)
    c.width = Inches(6.4)
    p0 = c.paragraphs[0]
    r = p0.add_run(title); r.bold = True; r.font.size = Pt(10.5); r.font.color.rgb = title_color
    p0.paragraph_format.space_after = Pt(3)
    for ln in lines:
        p = c.add_paragraph()
        r = p.add_run(ln); r.font.size = Pt(9.5)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.left_indent = Inches(0.08)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)

def caption(t):
    p = doc.add_paragraph(); r = p.add_run(t); r.italic = True; r.font.size = Pt(9)
    p.paragraph_format.space_after = Pt(2)

def table(headers, rows, bold_first_col=False, bold_win_col=None):
    t = doc.add_table(rows=1, cols=len(headers)); t.style = 'Table Grid'
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, h in enumerate(headers):
        c = t.rows[0].cells[j]; c.text = ''
        r = c.paragraphs[0].add_run(h); r.bold = True; r.font.size = Pt(9)
        c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    for row in rows:
        cells = t.add_row().cells
        for j, v in enumerate(row):
            cells[j].text = ''
            r = cells[j].paragraphs[0].add_run(str(v)); r.font.size = Pt(9)
            if bold_first_col and j == 0: r.bold = True
            if bold_win_col is not None and j == bold_win_col: r.bold = True
            cells[j].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t

# ---------------- TITLE BLOCK ----------------
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run('Reports, Not Crimes: Forecasting and Grouping Indian States for Cyber-Police Planning, 2018–2022')
r.bold = True; r.font.size = Pt(15); r.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)
p.paragraph_format.space_after = Pt(4)
for line in ['Abhirup Chakraborty', 'Department of Computer Science | Roll No. 24020570039',
             'Machine Learning (B.Sc. (H) Computer Science, Semester VI) | Problem 51',
             'Article type: Short Communication — focused note for cyber-police planning']:
    P(line, size=9, align='c', space=1)

# ---------------- ABSTRACT ----------------
H('Abstract')
P('Problem. State cyber-crime registrations in India are rising fast, but five years of data make forecasting difficult. '
  'Methods. Using NCRB State/UT registrations for 2018–2022 (36 entities), we test one-year-ahead linear, polynomial and ridge trend models '
  'against a persistence baseline, relate national totals to TRAI internet/tele-density and NPCI/RBI UPI penetration, and cluster States '
  'by their growth profiles using k-means and Ward hierarchical clustering. '
  'Results. Persistence is hardest to beat (RMSE 1,360; MAE 581; R² 0.861); ridge improves on trend models (RMSE 1,855; R² 0.742) but not on persistence. '
  'National registrations track internet subscribers (r = 0.981) and UPI volume (r = 0.987), while headline tele-density falls (91.45% to 84.56%) '
  'as a denominator artefact. A State-level merge with TRAI service-area tele-density adds little once State effects and trend are included '
  '(ridge RMSE 1,833 against 1,855) and the 2022 cross-section is essentially flat (r = −0.03). '
  'Four State clusters emerge, with a high-volume fast-riser group led by Telangana, Karnataka, Uttar Pradesh and Maharashtra. '
  'Implication. Plan by cluster, forecast with explicit uncertainty, and treat counts as reports rather than incidence.')
P('Keywords: cyber-crime; NCRB; forecasting; ridge regression; k-means; hierarchical clustering; tele-density; UPI; SDG 9; SDG 16', i=True, size=9.5)

box('What we found in five lines', [
    '1. Cyber-crime registrations grew ~25% a year; 2022 alone added 12,900 cases.',
    '2. "Last year\'s count" beat every trend model — the series is shock-driven, not smooth.',
    '3. Internet and UPI penetration track registrations almost one-to-one at the national level.',
    '4. States fall into four growth groups, not a single ranking.',
    '5. High counts may mean good reporting; low counts often mean the opposite.'])

# ---------------- 1 INTRODUCTION ----------------
H('1. Introduction')
P('In 2018, a district cyber cell in India might have handled a handful of online-fraud complaints a week. By 2022, the national count had risen '
  'from 27,248 to 65,893 — a 25% annual climb that outpaced almost every other crime category. Behind each registration is a person who lost money '
  'to a UPI scam, a teenager being blackmailed over a private photo, or a woman facing online harassment. The policing question is not whether '
  'cyber-crime is rising, but where next year\'s load will land — and whether States with similar growth can plan together. This note offers a '
  'deliberately simple, auditable answer.')
P('We ask three linked questions: (1) can next year\'s registrations be predicted from the previous four? (2) is the rise associated with digital '
  'penetration? (3) do States fall into distinct growth types that a planner can act on? The work speaks to SDG 16 (peace, justice and strong '
  'institutions) and SDG 9 (industry, innovation and infrastructure).')
P('Prior research frames our cautions. Official statistics systematically undercount crime, and the gap between recorded and actual offending is '
  'widest where reporting infrastructure is weakest (Biderman and Reiss, 1967; Skogan, 1977); the global picture is similarly reporting-dependent '
  '(UNODC, 2013). Indian cyber-crime scholarship documents the same problem alongside the gendered harms of online abuse (Jaishankar, 2011; '
  'Halder and Jaishankar, 2011). Correlations computed on aggregated data also cannot be transferred to individuals without committing the '
  'ecological fallacy (Robinson, 1950).')

# ---------------- 2 DATA ----------------
H('2. Data and its limits')
P('NCRB State/UT cyber-crime registrations 2018–2022 were obtained from the OGD Platform India (GODL-India) and contain 28 States, 8 Union '
  'Territories and three aggregate rows. Aggregates are excluded from modelling. Ladakh (created in 2019; no 2018–2019 values) is excluded from '
  'supervised fits but reported descriptively, leaving 35 entities and 140 training plus 35 test observations. Counts are extremely skewed '
  '(2022: Telangana 15,297 versus Lakshadweep 1), so models are fitted on log(1 + cases) and evaluated on counts. Zero counts are retained because '
  'they are informative — Sikkim recorded none in 2020–2021 and then 26 in 2022.')
P('Penetration data come from TRAI Yearly Performance Indicators and the Oct–Dec 2022 quarterly report (December-end tele-density 91.45, 88.56, '
  '86.38, 85.91 and 84.56%; internet subscribers 604.21, 718.74, 795.18, 829.30 and 865.90 million) and from NPCI/RBI UPI volumes '
  '(0.92, 5.34, 12.52, 22.33 and 45.96 billion transactions).')
P('For the State-level merge requested in the brief, TRAI Table 1.5 (service-area-wise tele-density) was transcribed for 2018–2022 and mapped to '
  'States using the licence-service-area definitions: Andhra Pradesh includes Telangana; Bihar includes Jharkhand; Madhya Pradesh includes '
  'Chhattisgarh; Maharashtra includes Goa; Uttar Pradesh includes Uttarakhand; West Bengal includes Sikkim; the North-East circle covers Arunachal '
  'Pradesh, Manipur, Meghalaya, Mizoram, Nagaland and Tripura; Chandigarh is in Punjab, Puducherry and the Andaman and Nicobar Islands in Tamil '
  'Nadu, Lakshadweep in Kerala, and Dadra and Nagar Haveli and Daman and Diu in Gujarat. The mapping is documented in data/circle_state_map.csv. '
  'Because the circle values aggregate neighbouring States, merged tele-density is an approximation and is used as a predictor, not as a causal '
  'variable. The 19-circle table and its provenance are in data/teledensity_circle.csv; the merged panel is outputs/panel_with_teledensity.csv.')

# ---------------- 3 METHOD ----------------
H('3. Method in plain language')
P('Training uses 2018–2021 and testing uses 2022, so no model sees the future. Four forecasters are compared. Persistence predicts 2022 as 2021. '
  'Pooled ordinary least squares fits log(1 + cases) with a separate intercept for each State and one shared time slope, so States differ in level '
  'but grow at a common rate. Ridge regression applies the same structure with an L2 penalty that shrinks noisy small-State estimates (Hoerl and '
  'Kennard, 1970); the penalty is chosen on a 2021 holdout from {0.1, 1, 10, 100, 1000}. A quadratic trend with ridge allows curvature. Accuracy '
  'is reported as RMSE (sensitive to large misses), MAE (typical miss) and R² (improvement over the mean). Persistence is included as a formal '
  'comparator because simple benchmarks routinely beat complex models on short series (Makridakis et al., 2020; Hyndman and Athanasopoulos, 2021).')
P('A further specification adds the merged tele-density as a predictor: log(1 + cases) = alpha_s + beta t + gamma tele-density_s,t + e. '
  'The coefficient gamma is read as the change in log cases per additional connection per 100 people, holding State and trend fixed. Because '
  'tele-density is circle-aggregated for several States, this model is a robustness check rather than a causal estimate.')
P('For profiling, each State is represented by its five-year log-count vector, standardised year by year, and clustered with k-means (MacQueen, '
  '1967; 20 restarts) and Ward hierarchical clustering (Ward, 1963). Silhouette scores (Rousseeuw, 1987) are computed for k = 2 to 6. We retain '
  'k = 4 for planning usefulness while reporting that k = 2 maximises separation. All code is in run_experiments.py with a fixed seed of 42.')

# ---------------- 4 RESULTS ----------------
H('4. Results')
P('4.1 Forecasting. Table 1 answers the planner\'s question directly. Every learned trend model loses to persistence. Ridge is the best learned model '
  'but still 36% worse in RMSE; the quadratic overfits reversals such as Assam\'s fall from 4,846 to 1,733 and Karnataka\'s rebound from 8,136 to 12,556. '
  'The shared OLS slope is 0.193 log units per year, roughly 21% annual growth.')
P('Table 1. Which model should a planner trust for 2022? (35 entities, counts; best value in bold.)', b=True, size=9)
table(['Model', 'RMSE', 'MAE', 'R²'],
      [['Persistence (2022 = 2021)', '1,359.8', '580.8', '0.861'],
       ['State mean 2018–21', '1,994.0', '730.0', '0.702'],
       ['Pooled OLS (State effects + trend)', '1,909.6', '717.8', '0.727'],
       ['Ridge, alpha = 0.1', '1,854.7', '646.5', '0.742'],
       ['Quadratic + Ridge, alpha = 1.0', '2,569.5', '967.0', '0.505']],
      bold_first_col=True)

P('Two States illustrate why a single national growth rate is misleading. Assam registered 4,846 cases in 2021 and 1,733 in 2022 — a collapse no trend '
  'model anticipated. Karnataka moved the other way, from 8,136 to 12,556, past its 2019 peak. Fig. 1 ranks these surprises. Telangana\'s actual count '
  'of 15,297 against a ridge forecast of 5,409 is the largest single miss, and Uttar Pradesh is over-predicted by about 3,300.')
caption('Fig. 1. Where the forecast was wrong. Bars show actual 2022 registrations minus the ridge prediction; green means under-prediction, red means over-prediction.')
doc.add_picture(os.path.join(OUT, 'fig5_surprise.png'), width=Inches(5.4))

P('4.2 Penetration. Nationally, log registrations correlate at r = 0.981 with log internet subscribers and r = 0.987 with log UPI volume. '
  'Headline tele-density moves the other way (r = −0.980) because subscriber counts were cleaned up and population projections revised; it is not a '
  'protective factor. These are ecological associations: a more connected, more digital-payment-heavy India registers more cyber-crime, but the data '
  'cannot say anything about an individual user\'s risk.')
caption('Fig. 2. All-India cyber-crime registrations against internet subscribers and UPI volume, 2018–2022.')
doc.add_picture(os.path.join(OUT, 'fig1_allindia.png'), width=Inches(5.2))

P('4.3 Growth profiles. Silhouette scores peak at k = 2 (0.607) and decline thereafter; we retain k = 4 (k-means 0.432, Ward 0.464) because four '
  'operational groups are more useful to planners. Table 2 names them. The high-volume fast-riser group carries most of the national caseload. '
  'Ward places Assam, Andhra Pradesh, Odisha and Rajasthan with the mid-volume group, which is a useful warning that cluster membership is sensitive '
  'to single-year reversals.')
P('Table 2. Four growth personalities (k-means, k = 4).', b=True, size=9)
rows = []
for k in range(4):
    sub = dfc[dfc.kmeans == k]
    mean22 = int(per[per.state.isin(sub.state)]['actual2022'].mean())
    rows.append([names[k], str(len(sub)), str(mean22), ', '.join(sub.state.tolist())])
table(['Cluster', 'n', 'Mean 2022 cases', 'States'], rows, bold_first_col=True)
caption('Fig. 3. Growth trajectories by cluster. Thin lines are individual States; thick lines are cluster means.')
doc.add_picture(os.path.join(OUT, 'fig3_clusters.png'), width=Inches(5.2))
caption('Fig. 4. Ridge forecast versus observed 2022 counts. Points below the diagonal are under-predictions.')
doc.add_picture(os.path.join(OUT, 'fig2_obs_pred.png'), width=Inches(3.6))

P('4.4 State-level tele-density merge. Table 3 adds the merged tele-density to the pooled specification. The penalty is small: the best learned '
  'model with tele-density (ridge, RMSE 1,833) improves on the trend-only ridge (1,855) by about 1% but remains far behind persistence. The '
  'coefficient is −0.004 log points per additional connection per 100 people, effectively zero at the scale of observed variation. Correlations '
  'confirm the null: pooled r = −0.12, within-State r = −0.23, and the 2022 cross-section r = −0.03 (Fig. 5). The strong national association '
  'in Fig. 2 is therefore a shared time trend, not a State-level relationship — a useful caution when a single national series is used to justify '
  'local policy.')
P('Table 3. Does State tele-density improve the forecast? (2022 holdout.)', b=True, size=9)
table(['Model', 'RMSE', 'MAE', 'R²', 'coef. tele-density'],
      [['Persistence', '1,359.8', '580.8', '0.861', '—'],
       ['OLS + tele-density', '1,893.1', '712.4', '0.731', '−0.0039'],
       ['OLS + log tele-density', '1,846.2', '689.1', '0.745', '−1.3455'],
       ['Ridge + tele-density', '1,833.0', '640.4', '0.748', '−0.0077']],
      bold_first_col=True)
caption('Fig. 5. State-level cross-section: 2022 log registrations against December 2022 tele-density. The relationship is essentially flat (r = −0.03).')
doc.add_picture(os.path.join(OUT, 'fig6_teledensity.png'), width=Inches(4.8))

# ---------------- 5 PLANNING ----------------
H('5. What this means for cyber-police planning')
box('For a district cyber cell, next quarter', [
    '• Use last year\'s count plus 20–25% as the planning baseline, not a single "predicted" number.',
    '• If your State is in C3 (Telangana, Karnataka, Uttar Pradesh, Maharashtra and peers), prepare surge capacity for fraud and payment disputes.',
    '• If you are in C0 or C1, low registrations probably reflect reporting gaps — invest in reporting channels before prediction.',
    '• Re-run this analysis every year; cluster membership and shocks change.',
    '• Publish forecasts with uncertainty. A smooth line through a shock-driven series creates false confidence.'],
    fill='EAF7EE', title_color=RGBColor(0x1E, 0x6B, 0x3A))

# ---------------- 6 CAUTION ----------------
H('6. A caution we take seriously')
box('What this note does not say', [
    'These are registrations, not crimes. A State can look safer because victims do not report, or worse because its police record more diligently.',
    'Our clusters are operational groupings for resource planning. They are not rankings of State safety, governance or morality, and must not be read as such.',
    'Correlation with internet and UPI growth is not proof of causation; both may respond to the same underlying digital expansion.',
    'Low counts in small States may reflect absence of reporting infrastructure rather than absence of harm.'],
    fill='FDF2E3', title_color=RGBColor(0x8A, 0x4B, 0x08))

# ---------------- 7 CONCLUSION ----------------
H('7. Conclusion')
P('Five years of NCRB data support profiling more than precise point forecasting. Registrations grow at roughly 21% a year on a log scale and track '
  'digital penetration almost one-to-one at the national level, but the State-level tele-density merge shows that this association does not survive '
  'once State effects and trend are included. Four growth clusters offer a practical basis for resource allocation. The clearest next steps are to '
  'extend the series with crime-head detail and State-level internet/UPI data, and to re-train annually rather than treat any single forecast as fixed.')

# ---------------- DECLARATIONS ----------------
H('Declarations')
P('Data availability. NCRB State/UT cyber-crime registrations 2018–2022 via the OGD Platform India (GODL-India); the working copy is datafile.xls. '
  'TRAI Yearly Performance Indicators 2019 (25 November 2020), 2020 (15 September 2021), 2021 (8 July 2022) and the Oct–Dec 2022 quarterly report '
  '(31 May 2023) provide tele-density and internet figures, including Table 1.5 service-area tele-density transcribed in data/teledensity_circle.csv '
  'with the State mapping in data/circle_state_map.csv. UPI volumes follow NPCI/RBI financial-year series. Cleaned tables and figures are in outputs/; '
  'run_experiments.py regenerates all results with seed 42.', size=9.5)
P('Conflict of interest. None. Funding. None.', size=9.5)
P('Declaration of generative AI and AI-assisted technologies. During the preparation of this work, the author used OpenCode, an AI coding '
  'assistant, to help write and debug the analysis code and to suggest editorial improvements to the manuscript. All data extraction, model '
  'fitting, metric computation and figure generation were executed and verified by the author. The author reviewed and edited all AI-assisted '
  'output and takes full responsibility for the content of this publication.', size=9.5)

H('References')
for r in [
    'Biderman, A.D. and Reiss, A.J. (1967). On exploring the "dark figure" of crime. The Annals of the American Academy of Political and Social Science, 374(1), 1–15.',
    'Halder, D. and Jaishankar, K. (2011). Cyber Crime and the Victimization of Women: Laws, Rights and Regulations. IGI Global, Hershey, PA.',
    'Hastie, T., Tibshirani, R. and Friedman, J. (2009). The Elements of Statistical Learning, 2nd edn. Springer, New York.',
    'Hoerl, A.E. and Kennard, R.W. (1970). Ridge regression: biased estimation for nonorthogonal problems. Technometrics, 12(1), 55–67.',
    'Hyndman, R.J. and Athanasopoulos, G. (2021). Forecasting: Principles and Practice, 3rd edn. OTexts, Melbourne.',
    'Jaishankar, K. (ed.) (2011). Cyber Criminology: Exploring Internet Crimes and Criminal Behavior. CRC Press, Boca Raton, FL.',
    'MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability, 1, 281–297.',
    'Makridakis, S., Spiliotis, E. and Assimakopoulos, V. (2020). The M4 Competition: 100,000 time series and 61 forecasting methods. International Journal of Forecasting, 36(1), 54–74.',
    'National Crime Records Bureau (2022). Crime in India 2022. Ministry of Home Affairs, Government of India, New Delhi.',
    'National Payments Corporation of India (2023). UPI Product Statistics. NPCI, Mumbai.',
    'Open Government Data Platform India (2023). State/UT-wise number of cases registered under cyber crimes, 2018–2022. NCRB, New Delhi.',
    'Pedregosa, F. et al. (2011). Scikit-learn: machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830.',
    'Reserve Bank of India (2023). Annual Report 2022–23. RBI, Mumbai.',
    'Robinson, W.S. (1950). Ecological correlations and the behavior of individuals. American Sociological Review, 15(3), 351–357.',
    'Rousseeuw, P.J. (1987). Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. Journal of Computational and Applied Mathematics, 20, 53–65.',
    'Skogan, W.G. (1977). Dimensions of the dark figure of unreported crime. Crime & Delinquency, 23(1), 41–50.',
    'Telecom Regulatory Authority of India (2020–2023). The Indian Telecom Services Performance Indicators (yearly 2019, 2020, 2021; quarterly Oct–Dec 2022). TRAI, New Delhi.',
    'United Nations Office on Drugs and Crime (2013). Comprehensive Study on Cybercrime. UNODC, Vienna.',
    'Ward, J.H. (1963). Hierarchical grouping to optimize an objective function. Journal of the American Statistical Association, 58(301), 236–244.']:
    P(r, size=9)

# ---------------- APPENDIX ----------------
doc.add_page_break()
H('Appendix A. Per-State forecast detail')
P('Full table in outputs/per_state_forecast.csv. Largest misses: Telangana actual 15,297 against persistence 10,303 and ridge 5,409; Assam 1,733 against '
  '4,846 and 4,475; Karnataka 12,556 against 8,136 and 13,008, where ridge was closest.', size=9.5)
P('Table A1. Cluster silhouette by k.', b=True, size=9)
table(['k', 'k-means silhouette', 'Ward silhouette', 'Inertia'],
      [[2, '0.607', '0.607', '45.0'], [3, '0.477', '0.489', '26.0'], [4, '0.432', '0.464', '17.1'],
       [5, '0.412', '0.437', '12.9'], [6, '0.389', '0.423', '10.2']])
caption('Fig. A1. Ward dendrogram of State growth profiles.')
doc.add_picture(os.path.join(OUT, 'fig4_dendrogram.png'), width=Inches(6.0))

H('Appendix B. Tele-density merge: circle-to-State mapping')
P('TRAI publishes tele-density by licence service area, not by State. The mapping below is used for the State-level merge in Section 4.4; aggregated '
  'circles are flagged in the note column.', size=9.5)
table(['State/UT', 'TRAI circle', 'Note'],
      [[r['state'], r['circle'], ('' if pd.isna(r['note']) else r['note'])] for _, r in mp.iterrows()])

doc.save(os.path.join(PAPER, 'Short_Communication_51_CyberCrime_Abhirup_24020570039.docx'))
print('saved')
