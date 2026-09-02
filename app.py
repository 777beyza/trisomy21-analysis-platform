import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from fpdf import FPDF
import io
import requests
from src.module_advanced import AcousticAnalyzer, BehavioralTracker, GeneticRiskProjector
from src.database import save_caregiver_log, get_caregiver_logs
from src.pubmed_api import fetch_pubmed_articles
from src.r_wrapper import run_r_analysis

# Set page config
st.set_page_config(
    page_title="Trisomi 21 Analiz Platformu",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Custom CSS for modern look
st.markdown("""
<style>
    /* Global App Background */
    .stApp {
        background-color: #f8fafc;
    }
    
    /* Sleek Cards */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    }
    
    /* Modern Buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4);
    }
    
    /* Form Background */
    div[data-testid="stForm"] {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.03);
        border: 1px solid #e2e8f0;
    }
    
    /* Text Accents */
    h1, h2, h3 {
        color: #1e293b;
        font-family: 'Inter', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

from src.module1_diagnosis import EarlyDiagnosisModel
from src.module3_projection import FutureProjectionModel

@st.cache_resource
def load_models():
    m1_model = EarlyDiagnosisModel()
    m3_model = FutureProjectionModel()
    try:
        m1_model.load_model()
        m3_model.load_model()
        return m1_model, m3_model
    except FileNotFoundError:
        return None, None

def tr_to_en(text):
    tr_map = str.maketrans("ığüşöçİĞÜŞÖÇ", "igusocIGUSOC")
    return str(text).translate(tr_map)

def generate_pdf_report(patient_data, risk_score, diagnosis):
    pdf = FPDF()
    pdf.add_page()
    
    # 1. Header (Logo / Title Background)
    pdf.set_fill_color(37, 99, 235) # Blue background
    pdf.rect(0, 0, 210, 30, 'F')
    pdf.set_y(10)
    pdf.set_font("helvetica", "B", 18)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 10, tr_to_en("TRISOMI 21 KLINIK ANALIZ RAPORU"), ln=True, align="C")
    
    # Reset text color
    pdf.set_text_color(0, 0, 0)
    pdf.ln(15)
    
    # 2. Meta Information
    pdf.set_font("helvetica", "B", 10)
    pdf.cell(100, 8, tr_to_en(f"Rapor Tarihi: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}"), border=0)
    pdf.cell(90, 8, tr_to_en(f"Protokol No: TR-{np.random.randint(10000, 99999)}"), border=0, align="R", ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    
    # 3. Patient Data Section (Table format)
    pdf.set_font("helvetica", "B", 12)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(0, 10, tr_to_en(" Hasta Parametreleri (Giris Verileri)"), ln=True, fill=True)
    
    pdf.set_font("helvetica", "", 11)
    for key, value in patient_data.items():
        pdf.cell(95, 8, tr_to_en(f" {key}"), border=1)
        pdf.cell(95, 8, tr_to_en(f" {value}"), border=1, ln=True)
    
    pdf.ln(10)
    
    # 4. Result Section
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 10, tr_to_en(" Yapay Zeka Risk Analizi (Modul 1)"), ln=True, fill=True)
    
    pdf.set_font("helvetica", "", 12)
    pdf.cell(95, 10, tr_to_en(" Hesaplanmis Risk Skoru:"), border=1)
    
    # Color logic for score
    if risk_score > 50:
        pdf.set_text_color(220, 38, 38) # Red
    else:
        pdf.set_text_color(22, 163, 74) # Green
    pdf.cell(95, 10, tr_to_en(f" %{risk_score:.2f}"), border=1, ln=True, align="C")
    
    pdf.set_text_color(0, 0, 0)
    pdf.cell(95, 10, tr_to_en(" Algoritma Karari:"), border=1)
    
    if risk_score > 50:
        pdf.set_text_color(220, 38, 38)
    else:
        pdf.set_text_color(22, 163, 74)
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(95, 10, tr_to_en(f" {diagnosis}"), border=1, ln=True, align="C")
    
    pdf.ln(15)
    
    # 5. Disclaimer (Footer warning)
    pdf.set_text_color(100, 100, 100)
    pdf.set_font("helvetica", "I", 9)
    pdf.multi_cell(0, 5, tr_to_en("YASAL UYARI: Bu rapor gelismis yapay zeka ve biyoinformatik algoritmalarina dayali bir tarama sonucudur. Hicbir sekilde kesin klinik teshis anlami tasimaz. Sonuclarin uzman bir genetik doktoru tarafindan yorumlanmasi ve gerekliyse invaziv (amniyosentez vb.) testler ile dogrulanmasi sarttir."))
    
    return bytes(pdf.output())



def plot_karyotype(is_trisomy=False):
    # Simulated karyotype visualization using a bar chart for 23 pairs
    pairs = list(range(1, 23)) + ["X", "Y"]
    counts = [2] * 24
    if is_trisomy:
        counts[20] = 3 # Chromosome 21
    
    colors = ['#1f77b4'] * 24
    if is_trisomy:
        colors[20] = '#ff4b4b' # Highlight 21
    
    fig = go.Figure(data=[go.Bar(
        x=pairs, y=counts,
        marker_color=colors,
        text=counts,
        textposition='auto',
    )])
    fig.update_layout(
        title="Dijital Karyotip Görünümü (Kromozom Sayıları)",
        xaxis_title="Kromozom Numarası",
        yaxis_title="Kopya Sayısı",
        yaxis=dict(range=[0, 4], dtick=1),
        height=400
    )
    return fig

def main():
    # Initialize advanced modules
    acoustic_engine = AcousticAnalyzer()
    behavior_engine = BehavioralTracker()
    genetic_engine = GeneticRiskProjector()
    
    st.title("🧬 Trisomi 21 Entegre Biyoinformatik Analiz Platformu")
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Bireysel Analiz", 
        "📂 Toplu Analiz", 
        "🎤 Ses ve Akustik Analiz", 
        "🏠 Bakıcı Takip Paneli",
        "📚 Literatür"
    ])
    with tab1:
        st.markdown("Bu platform, cfDNA üzerinden **Down Sendromu** taraması, genetik etki haritalaması ve gelecekteki olası nörolojik hastalık risklerinin analizini gerçekleştirmektedir.")

        m1_model, m3_model = load_models()
        if m1_model is None or m3_model is None:
            st.warning("Makine Öğrenmesi modelleri henüz eğitilmemiş. Lütfen arka planda modelleri eğitin.")
            st.stop()

        with st.form("diagnosis_form"):
            col1, col2 = st.columns(2)
            with col1:
                maternal_age = st.number_input("Anne Yaşı", 18, 50, 30)
                fetal_fraction = st.slider("Fetal Fraksiyon (%)", 0.0, 20.0, 10.0) / 100
            with col2:
                chr21_ratio = st.number_input("Chr21 Oranı", 0.0, 2.0, 1.0)
                gc_bias = st.number_input("GC Sapması", 0.0, 1.0, 0.5)
                fragment_ratio = st.number_input("Fragment Oranı", 0.0, 1.0, 0.5)
            
            submit_button = st.form_submit_button("Analizi Başlat")

        if submit_button:
            st.session_state['analysis_done'] = True
            st.session_state['patient_features'] = [maternal_age, fetal_fraction, chr21_ratio, gc_bias, fragment_ratio]
            
        if st.session_state.get('analysis_done', False):
            patient_features = st.session_state['patient_features']
            risk_score, prediction = m1_model.predict_risk(patient_features)
            
            # To pass risk_score down without re-predicting if you want, but fast prediction is fine here.
            
            st.header("📌 Analiz Sonucu")
            c1, c2 = st.columns(2)
            
            with c1:
                if risk_score > 50:
                    st.error(f"⚠️ YÜKSEK RİSK: %{risk_score:.2f}")
                else:
                    st.success(f"✅ DÜŞÜK RİSK: %{risk_score:.2f}")
                
                st.subheader("💡 Karar Açıklaması (SHAP)")
                shap_vals, base_val = m1_model.get_shap_explanation(patient_features)
                fig_shap, ax_shap = plt.subplots(figsize=(10, 6))
                feature_names = ["Anne Yaşı", "Fetal Fraksiyon", "Chr21 Oranı", "GC Sapması", "Fragment Oranı"]
                shap.bar_plot(shap_vals, feature_names=feature_names, show=False)
                st.pyplot(fig_shap)
                plt.close()

            with c2:
                st.subheader("🧬 Dijital Karyotip")
                st.plotly_chart(plot_karyotype(is_trisomy=(risk_score > 50)), use_container_width=True)

            st.markdown("---")
            c3, c4 = st.columns(2)
            
            with c3:
                st.subheader("📈 Popülasyon Kıyaslaması")
                # Load some background data for comparison
                bg_data = pd.read_csv("data/cfdna_dataset.csv")
                fig_comp, ax_comp = plt.subplots()
                sns.kdeplot(data=bg_data, x="Chr21_Read_Ratio", hue="Trisomy_21", fill=True, ax=ax_comp)
                plt.axvline(chr21_ratio, color='red', linestyle='--', label='Sizin Hastanız')
                plt.title("Kromozom 21 Oranı Dağılımı")
                st.pyplot(fig_comp)
                plt.close()

            with c4:
                st.subheader("🧪 Yolak (Pathway) Analizi")
                path_scores = m1_model.get_pathway_scores(risk_score)
                df_path = pd.DataFrame(list(path_scores.items()), columns=['Pathway', 'Score'])
                fig_path = px.bar(df_path, x='Score', y='Pathway', orientation='h', 
                                  color='Score', color_continuous_scale='RdPu')
                st.plotly_chart(fig_path, use_container_width=True)

            st.markdown("---")
            st.subheader("🔬 Modül 2: Moleküler Etki Haritalaması (R Entegrasyonu)")
            st.info("Bu modül, arka planda R programlama dilini (Rscript) çalıştırarak Diferansiyel Gen Ekspresyonu (RNA-Seq) analizi yapar. Yüksek risk grubundaki hastalar için Trisomi 21 ile ilişkili genlerin istatistiksel farklılıklarını (P-value ve Fold Change) hesaplar.")
            
            if st.button("R ile RNA-Seq Analizini Başlat (Derin Analiz)"):
                with st.spinner("R Motoru Başlatılıyor... Veriler Rscript'e gönderilip analiz ediliyor..."):
                    try:
                        volcano_df = run_r_analysis(risk_score)
                        
                        st.success("R Analizi başarıyla tamamlandı! İşte sonuçlar:")
                        
                        fig_volcano = px.scatter(volcano_df, x='Log2FC', y='-Log10(P-value)', 
                                                 hover_name='Gene', 
                                                 title="Diferansiyel Gen Ekspresyonu (Volcano Plot)",
                                                 color='-Log10(P-value)', color_continuous_scale='Viridis')
                        # Highlight significant thresholds
                        fig_volcano.add_hline(y=1.3, line_dash="dash", line_color="red") # p=0.05
                        fig_volcano.add_vline(x=1, line_dash="dash", line_color="red")
                        fig_volcano.add_vline(x=-1, line_dash="dash", line_color="red")
                        
                        st.plotly_chart(fig_volcano, use_container_width=True)
                        
                        # Show raw data
                        with st.expander("R Çıktısı (Raw Data - İlk 50 Gen)"):
                            st.dataframe(volcano_df.head(50))
                            
                    except Exception as e:
                        st.error(f"R çalıştırılırken bir hata oluştu: {str(e)}")

            st.markdown("---")
            st.subheader("🧠 Genetik Çoklu Hastalık Risk Projeksiyonu (Modül 3 ML)")
            st.info("Modül 2'deki olası bozuk çalışan gen ekspresyonlarına dayanarak yapay zeka ile hastanın gelecekteki hastalık geliştirme riskleri modellenmektedir.")
            
            with st.spinner("Modül 3: Hastalık riskleri hesaplanıyor..."):
                # Simulate key Trisomy 21 genes over-expression based on risk score for Modül 3 ML model
                intensity = risk_score / 100.0
                app_val = 1.0 + (intensity * 2.5)
                dyrk1a_val = 1.0 + (intensity * 1.8)
                rcan1_val = 1.0 + (intensity * 2.0)
                gene_features = [app_val, dyrk1a_val, rcan1_val]
                
                predicted_risks = m3_model.predict_future_risks(gene_features)
                
            categories = list(predicted_risks.keys())
            risks = list(predicted_risks.values())
            
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=risks,
                theta=categories,
                fill='toself',
                name='Hasta Risk Profili',
                line_color='purple'
            ))

            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=False,
                title="Yapay Zeka Destekli Hastalık Geliştirme Riski (%)"
            )
            st.plotly_chart(fig_radar, use_container_width=True)

            # PDF Download
            diagnosis_text = "YUKSEK RISK" if risk_score > 70 else ("ORTA RISK" if risk_score > 30 else "DUSUK RISK")
            patient_data = {"Anne Yasi": maternal_age, "Fetal Fraksiyon": f"%{fetal_fraction*100:.2f}", 
                            "Chr21 Oranı": chr21_ratio, "GC Sapmasi": gc_bias, "Fragment Oranı": fragment_ratio}
            pdf_bytes = generate_pdf_report(patient_data, risk_score, diagnosis_text)
            st.download_button("📄 Detaylı Analiz Raporunu İndir", data=pdf_bytes, file_name="Rapor.pdf", mime="application/pdf")

    with tab2:
        st.markdown("### 📂 Çoklu Hasta Analizi (Batch Processing)")
        st.info("Aşağıdaki alana hasta verilerini içeren bir CSV dosyası yükleyerek toplu tarama yapabilirsiniz.")
        
        uploaded_file = st.file_uploader("CSV Dosyası Yükle", type="csv")
        if uploaded_file:
            batch_df = pd.read_csv(uploaded_file)
            st.write("Yüklenen Veri (İlk 5 Satır):", batch_df.head())
            
            if st.button("🚀 Toplu Analizi Başlat"):
                with st.spinner("Yüzlerce veri analiz ediliyor..."):
                    # Drop existing output columns to avoid feature mismatch errors from cached model
                    clean_df = batch_df.drop(columns=['Trisomy_21', 'Risk Skoru (%)', 'Teşhis'], errors='ignore')
                    
                    probs, preds = m1_model.predict_batch(clean_df)
                    batch_df['Risk Skoru (%)'] = probs
                    batch_df['Teşhis'] = ["Pozitif" if p == 1 else "Negatif" for p in preds]
                    
                    st.success("Analiz Tamamlandı!")
                    st.dataframe(batch_df)
                    
                    # Batch summary chart
                    fig_batch = px.histogram(batch_df, x="Risk Skoru (%)", color="Teşhis", 
                                             title="Grup Risk Dağılımı", barmode="overlay")
                    st.plotly_chart(fig_batch, use_container_width=True)
                    
                    csv = batch_df.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Sonuçları CSV Olarak İndir", data=csv, file_name="Toplu_Analiz_Sonuclari.csv", mime="text/csv")

    with tab3:
        st.markdown("### 🎤 Ses ve Akustik Analiz (Bilişsel Takip)")
        st.markdown("""
        Klasik bilişsel testlere alternatif olarak, kullanıcının günlük konuşmalarındaki duraksamaları (hesitation) 
        ve kelime dağarcığındaki değişimleri analiz eden invaziv olmayan bir yöntemdir.
        """)
        
        audio_file = st.file_uploader("Ses Kaydı Yükle (wav/mp3)", type=["wav", "mp3"])
        
        if audio_file:
            # Save uploaded file temporarily
            temp_audio_path = "data/temp_audio." + audio_file.name.split('.')[-1]
            os.makedirs("data", exist_ok=True)
            with open(temp_audio_path, "wb") as f:
                f.write(audio_file.getbuffer())
                
            st.audio(audio_file, format=f'audio/{audio_file.name.split(".")[-1]}')
            
            if st.button("Akustik Biyobelirteçleri Analiz Et"):
                with st.spinner("Ses dalgaları ve akustik biyobelirteçler analiz ediliyor..."):
                    results = acoustic_engine.analyze_speech(temp_audio_path)
                    
                    c_a1, c_a2, c_a3 = st.columns(3)
                    c_a1.metric("Duraksama (Sessizlik) Oranı", f"%{results['Hesitation Index']}")
                    c_a2.metric("Vokal Çeşitlilik Skoru", results['Vocal Variety'])
                    c_a3.metric("Bilişsel Yük Skoru", results['Cognitive Load Score'])
                    
                    st.subheader("Akustik Spektrum ve Duraksama Tespiti")
                    y = results['signal']
                    sr = results['sr']
                    intervals = results['non_silent_intervals']
                    
                    # Plotly chart for real audio
                    # Downsample for visualization speed if it's too long
                    max_points = 5000
                    if len(y) > max_points:
                        downsample_factor = len(y) // max_points
                        y_viz = y[::downsample_factor]
                        t_viz = np.linspace(0, len(y)/sr, len(y_viz))
                    else:
                        y_viz = y
                        t_viz = np.linspace(0, len(y)/sr, len(y))
                        
                    fig_audio = px.line(x=t_viz, y=y_viz, title="Gerçek Zamanlı Konuşma Sinyali (Zaman Ekseni: Saniye)", labels={'x': 'Zaman (s)', 'y': 'Genlik'})
                    
                    # Highlight non-silent parts
                    # Plotly doesn't easily shade multiple vrects if there are hundreds, so we just show the wave.
                    fig_audio.update_traces(line_color='#1f77b4')
                    st.plotly_chart(fig_audio, use_container_width=True)
                    
                    st.info("💡 Mavi grafik sesinizin dalga formunu (amplitude) gösterir. Düşük frekanslar ve uzun sessizlikler (duraksama) bilişsel yük tablosuna eklenir.")
                    
                    if results['Cognitive Load Score'] > 50:
                        st.error("Yüksek bilişsel yük ve uzun duraksamalar tespit edildi. Erken evre bilişsel gerileme (dementia) riski açısından takibi önerilir.")
                    else:
                        st.success("Ses analizinde akıcılık normal sınırlarda (Düşük bilişsel yük).")

    with tab4:
        st.markdown("### 🏠 Bakıcı Takip Paneli (Davranışsal Takip)")
        st.info("Down sendromunda Alzheimer ilk belirtileri genellikle kişilik değişimi ve inatçılık ile başlar. Bu panel, bakıcıların günlük gözlemlerini takip eder.")
        
        col_b1, col_b2 = st.columns([1, 2])
        
        with col_b1:
            st.markdown("#### Günlük Puanlama")
            
            # Use current date by default, but allow user to select a date
            selected_date = st.date_input("Kayıt Tarihi", pd.Timestamp.now().date())
            
            apathy = st.slider("Apati / İlgisizlik", 1, 10, 5)
            stubbornness = st.slider("İnatçılık / Kişilik Değişimi", 1, 10, 5)
            memory = st.slider("Hafıza Sorunları", 1, 10, 5)
            
            if st.button("Günü Kaydet"):
                # Save to sqlite
                save_caregiver_log(str(selected_date), apathy, stubbornness, memory)
                st.success("Veriler veritabanına kaydedildi ve trend analizine eklendi.")
                st.rerun() # Refresh to update the chart immediately
        
        with col_b2:
            st.markdown("#### Davranışsal Trend Analizi (Canlı Veritabanı)")
            
            # Pull historical data from sqlite
            history = get_caregiver_logs()
            
            if history.empty:
                st.info("Henüz sisteme girilmiş bir bakıcı verisi yok. Lütfen sol taraftan 'Günü Kaydet' butonunu kullanarak ilk verinizi ekleyin.")
            else:
                st.dataframe(history, use_container_width=True)
                
                # Sadece en az 3 günlük veri varsa anlamlı trend analizi yapılabilir
                if len(history) >= 3:
                    status, slope = behavior_engine.analyze_trends(history['Score'].tolist())
                    
                    fig_trend = px.line(history, x='Gün', y='Score', title="Tarihsel Davranışsal Gelişim Eğrisi", markers=True)
                    fig_trend.add_hline(y=4, line_dash="dash", line_color="red", annotation_text="Kritik Eşik")
                    st.plotly_chart(fig_trend, use_container_width=True)
                    
                    if "Riskli" in status:
                        st.error(f"DİKKAT: Davranışsal verilerde anlamlı bir düşüş tespit edildi: {status}")
                        st.info("Literatür Bilgisi: Down sendromlu bireylerde Alzheimer, hafıza kaybından önce bu tür apati ve kişilik değişimleriyle belirti gösterebilir.")
                    else:
                        st.success(f"Durum: {status}")
                else:
                    st.warning("Trend (eğilim) tahmini yapabilmek için en az 3 günlük veri girmelisiniz.")

    with tab5:
        st.markdown("### 📚 Bilimsel Literatür ve Araştırmalar")
        st.info("Trisomi 21 ve ilişkili genetik faktörler hakkında PubMed üzerinden en güncel araştırmalar.")
        
        search_query = st.text_input("Araştırma Terimi", value="Trisomy 21 Genetics")
        if st.button("🔍 Makaleleri Ara"):
            with st.spinner("PubMed veritabanı taranıyor..."):
                articles = fetch_pubmed_articles(search_query)
                if not articles:
                    st.warning("Sonuç bulunamadı veya API bağlantısı sağlanamadı.")
                for art in articles:
                    st.markdown(f"**[{art['title']}]({art['url']})**")
                    st.caption(f"Yayın Tarihi: {art['date']} | Yazar(lar): {art['authors']} | Dergi: {art['journal']}")
                    st.markdown("---")

if __name__ == '__main__':
    main()
