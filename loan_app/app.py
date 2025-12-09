from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

app = Flask(__name__)

# Load ML artifacts
model = joblib.load("best_loan_approval_model_reduced.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("label_encoders.pkl")

with open("selected_features.txt", "r") as f:
    selected_features = [line.strip() for line in f.readlines() if line.strip()]

SYSTEM_PROMPT = """
Anda adalah "Asisten Keuangan UMKM", sebuah chatbot profesional berbasis kecerdasan buatan
yang bertugas memberikan edukasi keuangan, menjelaskan faktor-faktor pinjaman, dan membantu
pengguna memahami proses simulasi approval dalam aplikasi "Simulasi Pinjaman UMKM".

===========================
🎯 1. Peran Utama
===========================
Anda berperan sebagai:
- Konsultan pembiayaan UMKM
- Edukator literasi keuangan tingkat dasar–menengah
- Pendamping pengajuan pinjaman berbasis logika finansial
- Penjelas fitur aplikasi Simulasi Pinjaman UMKM

Anda TIDAK berperan sebagai:
- Model prediksi ML aplikasi
- Pengambil keputusan resmi bank
- Pemberi kepastian hasil pinjaman
- Pengelola data pribadi pengguna

===========================
📌 2. Tugas Utama Chatbot
===========================
1. Menjelaskan arti setiap input dalam form simulasi:
   - Skor kredit, DTI, LTI, income, default, delinquencies, interest rate, dll.
2. Memberikan saran realistis untuk meningkatkan peluang approval pinjaman.
3. Menjelaskan konsep finansial:
   - Manajemen utang, risiko kredit, suku bunga, likuiditas, aset, rasio, dsb.
4. Menjawab pertanyaan teknis ringan terkait kredit UMKM.
5. Membantu pengguna memahami faktor apa yang biasanya mempengaruhi approval di lembaga keuangan.
6. Memberikan edukasi dengan bahasa mudah dipahami oleh pelaku UMKM.
7. Menjawab secara etis, sopan, dan tidak menggurui.

===========================
⚠ 3. Batasan & Hal yang Dilarang
===========================
Anda TIDAK boleh:
- Memberikan hasil prediksi model machine learning aplikasi.
- Memberikan angka peluang, probabilitas, atau tingkat approval.
- Mengelola atau menyimpan data pribadi pengguna.
- Memberikan instruksi ilegal atau manipulatif.
- Memberikan nasihat keuangan bersifat mengikat.
- Memberikan perhitungan eksak berdasarkan data pengguna.
- Mensimulasikan hasil model ML.

Jika pengguna bertanya:
- "Berapa peluang saya disetujui?" → Jawab dengan logika umum, bukan angka.
- "Data saya begini, apakah disetujui?" → Jelaskan faktor umum, bukan prediksi.

===========================
🧠 4. Gaya Jawaban
===========================
- Bahasa Indonesia yang jelas dan mudah dipahami.
- Edukatif, struktural, dengan contoh bila perlu.
- Nada bicara profesional, sopan, tenang.
- Jelaskan konsep finansial secara sederhana.

===========================
🧩 5. Pengetahuan Atribut Fitur
===========================
Anda harus bisa menjelaskan:
- age → umur peminjam
- years_employed → stabilitas kerja
- annual_income → kemampuan bayar
- credit_score → kualitas rekam kredit
- credit_history_years → lamanya menggunakan kredit
- savings_assets → kekuatan cadangan finansial
- current_debt → utang berjalan
- defaults_on_file → gagal bayar sebelumnya
- delinquencies_last_2yrs → keterlambatan 2 tahun terakhir
- derogatory_marks → catatan negatif kredit
- loan_amount → jumlah pinjaman
- interest_rate → suku bunga
- debt_to_income_ratio → rasio utang terhadap pendapatan
- loan_to_income_ratio → pinjaman terhadap penghasilan
- payment_to_income_ratio → cicilan terhadap penghasilan
- occupation_status → status pekerjaan
- product_type → tipe pinjaman
- loan_intent → tujuan pinjaman

===========================
🚦 6. Aturan Interaksi
===========================
- Jawab berdasarkan logika finansial umum.
- Jika data tidak cukup → minta detail tambahan.
- Jika pertanyaan di luar domain → jawab profesional dan netral.
- Hindari kepastian numerik apa pun.

===========================
⭐ 7. Persona
===========================
Anda ramah, baik, edukatif, dan profesional seperti konsultan kredit UMKM nyata.

===========================
🎯 Ringkas
===========================
Anda adalah pendamping finansial UMKM yang memberi edukasi dan arahan berdasarkan
logika keuangan, bukan prediksi model machine learning.
"""

def load_gemini_model():
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        google_api_key=GOOGLE_API_KEY
    )

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        user_input = {}
        for feat in selected_features:
            if feat == "customer_id":
                continue
            user_input[feat] = request.form.get(feat, '')

        # Process input similar to Streamlit code
        numeric_float_feats = [
            "years_employed",
            "credit_history_years",
            "credit_history years",
            "interest_rate",
            "debt_to_income_ratio",
            "loan_to_income_ratio",
            "payment_to_income_ratio",
        ]

        numeric_int_feats = [
            "annual_income",
            "credit_score",
            "savings_assets",
            "savings_asset",
            "current_debt",
            "delinquencies_last_2yrs",
            "derogatory_marks",
            "loan_amount",
            "defaults_on_file",
        ]

        for k in list(user_input.keys()):
            if k in numeric_float_feats:
                user_input[k] = float(user_input[k]) if user_input[k] else 0.0
            elif k in numeric_int_feats:
                user_input[k] = int(user_input[k]) if user_input[k] else 0

        df_user = pd.DataFrame([user_input])

        if "credit_history years" in df_user.columns and "credit_history_years" in selected_features:
            df_user["credit_history_years"] = df_user["credit_history years"]

        if "savings_asset" in df_user.columns and "savings_assets" in selected_features:
            df_user["savings_assets"] = df_user["savings_asset"]

        feature_names = getattr(scaler, "feature_names_in_", None)
        if feature_names is None:
            feature_names = list(
                dict.fromkeys(
                    list(df_user.columns) + list(encoders.keys()) + list(selected_features)
                )
            )
        else:
            feature_names = list(feature_names)

        full_row = {}

        for col in feature_names:
            if col in df_user.columns:
                full_row[col] = df_user[col].iloc[0]
            else:
                if col in encoders:
                    full_row[col] = encoders[col].classes_[0]
                else:
                    full_row[col] = 0.0

        df_full = pd.DataFrame([full_row])

        for col, encoder in encoders.items():
            if col in df_full.columns:
                df_full[col] = encoder.transform(df_full[col].astype(str))

        bad_cols = [c for c in df_full.columns if df_full[c].dtype == object]
        if bad_cols:
            return render_template('predict.html', error=f"Masih ada kolom kategorikal yang belum di-encode: {bad_cols}")

        arr_scaled = scaler.transform(df_full[feature_names])
        df_scaled_full = pd.DataFrame(arr_scaled, columns=feature_names)

        missing_cols = [c for c in selected_features if c not in df_scaled_full.columns]
        if missing_cols:
            return render_template('predict.html', error=f"Ada fitur yang dibutuhkan model tetapi tidak ada: {missing_cols}")

        df_model = df_scaled_full[selected_features]

        pred = model.predict(df_model)[0]
        prob = model.predict_proba(df_model)[0]

        prediction = "Approved" if pred == 1 else "Rejected"
        confidence = prob[1] if pred == 1 else prob[0]

        return render_template('predict.html', prediction=prediction, confidence=f"{confidence:.4f}")

    return render_template('predict.html')

@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    if not hasattr(app, 'llm'):
        app.llm = load_gemini_model()
    if not hasattr(app, 'messages'):
        app.messages = [SystemMessage(content=SYSTEM_PROMPT)]

    if request.method == 'POST':
        user_message = request.form.get('message')
        if user_message:
            app.messages.append(HumanMessage(content=user_message))

            try:
                response = app.llm.invoke(app.messages)
                app.messages.append(response)
                return jsonify({'response': response.content})
            except Exception as e:
                return jsonify({'response': f'Terjadi error: {str(e)}'})

    return render_template('chatbot.html')

if __name__ == '__main__':
    app.run(debug=True)
