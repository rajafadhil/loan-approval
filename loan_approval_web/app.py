import streamlit as st
import joblib
import pandas as pd
import numpy as np

# === Konfigurasi Halaman ===
st.set_page_config(
    page_title="Simulasi Pinjaman UMKM",
    page_icon="🏦",
    layout="centered"
)

# === Judul & Penjelasan ===
st.title("🏦 Simulasi Pinjaman UMKM")
st.markdown("""
    Masukkan data usaha dan pribadi Anda untuk mendapatkan **prediksi approval pinjaman**  
    berdasarkan model AI yang telah dilatih pada data riil.
""")

# === Load Model & Preprocessor ===
@st.cache_resource
def load_model():
    model = joblib.load('best_loan_approval_model_reduced.pkl')
    scaler = joblib.load('scaler.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    return model, scaler, label_encoders

try:
    model, scaler, label_encoders = load_model()
    FEATURE_NAMES = scaler.feature_names_in_
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")
    st.stop()

# === Baca fitur yang digunakan (jika ada) ===
try:
    with open('selected_features.txt', 'r') as f:
        SELECTED_FEATURES = [line.strip() for line in f.readlines()]
except:
    # Jika tidak ada, gunakan semua fitur dari scaler
    SELECTED_FEATURES = list(FEATURE_NAMES)

# === Sidebar untuk Input ===
st.sidebar.header("📝 Masukkan Data Anda")

# --- Input Numerik ---
age = st.sidebar.number_input("Usia (tahun)", min_value=18, max_value=70, value=35)
years_employed = st.sidebar.number_input("Lama Bekerja (tahun)", min_value=0.0, max_value=50.0, value=5.0, step=0.5)
annual_income = st.sidebar.number_input("Pendapatan Tahunan (Rp)", min_value=0, value=60000000, step=1000000)
credit_score = st.sidebar.number_input("Skor Kredit (300–850)", min_value=300, max_value=850, value=650)
credit_history_years = st.sidebar.number_input("Riwayat Kredit (tahun)", min_value=0.0, max_value=50.0, value=5.0, step=0.5)
savings_assets = st.sidebar.number_input("Aset Tabungan (Rp)", min_value=0, value=10000000, step=1000000)
current_debt = st.sidebar.number_input("Utang Saat Ini (Rp)", min_value=0, value=5000000, step=1000000)
defaults_on_file = st.sidebar.number_input("Jumlah Default di File", min_value=0, max_value=10, value=0)
delinquencies_last_2yrs = st.sidebar.number_input("Keterlambatan 2 Tahun Terakhir", min_value=0, max_value=10, value=0)
derogatory_marks = st.sidebar.number_input("Tanda Negatif", min_value=0, max_value=10, value=0)
loan_amount = st.sidebar.number_input("Jumlah Pinjaman (Rp)", min_value=0, value=25000000, step=1000000)
interest_rate = st.sidebar.number_input("Suku Bunga (%/tahun)", min_value=0.0, max_value=25.0, value=8.5, step=0.1)
debt_to_income_ratio = st.sidebar.number_input("Rasio Utang/Pendapatan", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
loan_to_income_ratio = st.sidebar.number_input("Rasio Pinjaman/Pendapatan", min_value=0.0, max_value=1.0, value=0.4, step=0.01)
payment_to_income_ratio = st.sidebar.number_input("Rasio Pembayaran/Pendapatan", min_value=0.0, max_value=1.0, value=0.2, step=0.01)

# --- Input Kategorikal ---
occupation_status = st.sidebar.selectbox("Status Pekerjaan", ["Employed", "Self-Employed", "Student"])
product_type = st.sidebar.selectbox("Jenis Pinjaman", ["Personal Loan", "Business Loan", "Mortgage"])
loan_intent = st.sidebar.selectbox("Tujuan Pinjaman", ["Debt Consolidation", "Home Improvement", "Business Expansion"])

# === Prediksi Saat Tombol Diklik ===
if st.sidebar.button("🔍 Simulasikan Approval"):
    try:
        # Buat dict input
        input_data = {
            'age': age,
            'occupation_status': occupation_status,
            'years_employed': years_employed,
            'annual_income': annual_income,
            'credit_score': credit_score,
            'credit_history_years': credit_history_years,
            'savings_assets': savings_assets,
            'current_debt': current_debt,
            'defaults_on_file': defaults_on_file,
            'delinquencies_last_2yrs': delinquencies_last_2yrs,
            'derogatory_marks': derogatory_marks,
            'product_type': product_type,
            'loan_intent': loan_intent,
            'loan_amount': loan_amount,
            'interest_rate': interest_rate,
            'debt_to_income_ratio': debt_to_income_ratio,
            'loan_to_income_ratio': loan_to_income_ratio,
            'payment_to_income_ratio': payment_to_income_ratio
        }

        # Buat DataFrame
        df = pd.DataFrame([input_data])

        # Encode kategorikal
        for col, le in label_encoders.items():
            if col in df.columns:
                df[col] = le.transform(df[col])

        # Pastikan urutan kolom sesuai dengan scaler
        df = df.reindex(columns=FEATURE_NAMES, fill_value=0)

        # Scaling
        df_scaled = scaler.transform(df)
        df_scaled = pd.DataFrame(df_scaled, columns=FEATURE_NAMES)

        # Ambil hanya fitur yang digunakan model (jika ada)
        X_input = df_scaled[SELECTED_FEATURES] if set(SELECTED_FEATURES).issubset(df_scaled.columns) else df_scaled

        # Prediksi
        pred = model.predict(X_input)[0]
        proba = model.predict_proba(X_input)[0]

        # Tampilkan hasil
        st.subheader("📊 Hasil Prediksi")
        if pred == 1:
            st.success(f"✅ **Pinjaman Anda DISETAJUI!**")
        else:
            st.error(f"❌ **Pinjaman Anda DITOLAK**")

        st.metric("Keyakinan Prediksi", f"{max(proba) * 100:.1f}%")

        # Penjelasan (opsional)
        if pred == 0:
            st.warning("ℹ️ **Alasan Penolakan (Estimasi):**")
            reasons = []
            if credit_score < 600:
                reasons.append("Skor kredit terlalu rendah (<600)")
            if debt_to_income_ratio > 0.4:
                reasons.append("Rasio utang terhadap pendapatan > 40%")
            if delinquencies_last_2yrs > 0:
                reasons.append("Ada riwayat keterlambatan pembayaran")
            if not reasons:
                reasons.append("Profil risiko dinilai tinggi berdasarkan kombinasi faktor")
            st.write("; ".join(reasons))

    except Exception as e:
        st.error(f"Terjadi kesalahan: {str(e)}")

# === Footer ===
st.markdown("---")
st.caption("💡 Dibuat untuk mendukung UMKM Indonesia | Model AI berbasis data riil")