from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
import google.generativeai as genai
import torch
from transformers import pipeline
from flask_cors import CORS

# Load environment
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY tidak ditemukan! Pastikan file .env berisi kunci API.")

genai.configure(api_key=GOOGLE_API_KEY)

app = Flask(__name__)
CORS(app)

# === JANGAN DIUBAH: Bagian model ML ===
model = joblib.load("best_loan_approval_model_reduced.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("label_encoders.pkl")

with open("selected_features.txt", "r") as f:
    selected_features = [line.strip() for line in f.readlines() if line.strip()]
# ===================================

SYSTEM_PROMPT = """
Anda adalah Asisten Keuangan UMKM, chatbot edukasi keuangan untuk membantu pemahaman simulasi pinjaman.

Peran: Konsultan pembiayaan UMKM, edukator literasi keuangan, pendamping pengajuan pinjaman.

Tugas: Jelaskan input form (skor kredit, DTI, LTI, dll.), beri saran realistis untuk tingkatkan peluang approval, jelaskan konsep finansial (utang, risiko, suku bunga, rasio), jawab pertanyaan kredit UMKM, edukasi bahasa mudah.

Larangan: Jangan beri prediksi ML, probabilitas, angka approval, kelola data pribadi, nasihat mengikat, perhitungan eksak, simulasi model.

Jawaban: Bahasa Indonesia jelas, edukatif, profesional, sopan. Hindari kepastian numerik.

Pengetahuan fitur: age (umur), years_employed (stabilitas kerja), annual_income (kemampuan bayar), credit_score (rekam kredit), credit_history_years (lama kredit), savings_assets (cadangan), current_debt (utang), defaults_on_file (gagal bayar), delinquencies_last_2yrs (keterlambatan), derogatory_marks (catatan negatif), loan_amount (jumlah pinjaman), interest_rate (suku bunga), debt_to_income_ratio (rasio utang/pendapatan), loan_to_income_ratio (pinjaman/pendapatan), payment_to_income_ratio (cicilan/pendapatan), occupation_status (pekerjaan), product_type (tipe pinjaman), loan_intent (tujuan).

Aturan: Jawab logika umum, minta detail jika kurang, netral jika luar domain.

Persona: Ramah, edukatif, profesional seperti konsultan kredit UMKM.
"""

def call_gemini(prompt, system_instruction=SYSTEM_PROMPT):
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",  # atau "gemini-2.0-flash-exp"
        system_instruction=system_instruction
    )
    response = model.generate_content(prompt)
    return response.text.strip()

def call_local_model(prompt, max_length=256):
    # Kembalikan ke Gemini (Google Generative AI)
    return call_gemini(prompt)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # --- Bagian prediksi TIDAK DIUBAH ---
    if request.method == 'POST':
        user_input = {}
        for feat in selected_features:
            if feat == "customer_id":
                continue
            user_input[feat] = request.form.get(feat, '')

        numeric_float_feats = [
            "years_employed", "credit_history_years", "credit_history years",
            "interest_rate", "debt_to_income_ratio", "loan_to_income_ratio", "payment_to_income_ratio"
        ]

        numeric_int_feats = [
            "credit_score", "savings_assets", "savings_asset", "current_debt",
            "delinquencies_last_2yrs", "derogatory_marks", "loan_amount", "defaults_on_file"
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
            feature_names = list(dict.fromkeys(list(df_user.columns) + list(encoders.keys()) + list(selected_features)))
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

        return render_template('result.html', prediction=prediction, confidence=f"{confidence:.4f}", user_input=user_input)

    return render_template('predict.html')

@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    if request.method == 'POST':
        user_message = request.form.get('message')
        if user_message:
            try:
                # Kembali gunakan Gemini
                response = call_gemini(user_message)
                return jsonify({'response': response})
            except Exception as e:
                return jsonify({'response': f'Terjadi error: {str(e)}'})
    return render_template('chatbot.html')

@app.route('/ask_ai', methods=['POST'])
def ask_ai():
    data = request.get_json()
    prediction = data.get('prediction')
    confidence = data.get('confidence')
    user_input = data.get('user_input', {})

    prompt = f"""
    Berdasarkan data input pengguna berikut dan hasil prediksi yang ditolak, jelaskan secara singkat dan jelas mengapa pengajuan pinjaman ini ditolak.

    Data Input Pengguna:
    {user_input}

    Hasil Prediksi: Ditolak dengan confidence {confidence}

    Berikan penjelasan berdasarkan faktor-faktor umum dalam penilaian kredit, tanpa memberikan angka pasti atau simulasi model.
    """

    try:
        response = call_gemini(prompt, system_instruction="Anda adalah asisten keuangan yang menjelaskan alasan penolakan pinjaman berdasarkan data umum.")
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'response': f'Terjadi kesalahan saat menghubungi AI: {str(e)}'})

@app.route('/company')
def company():
    return render_template('company.html')

if __name__ == '__main__':
    app.run(debug=True)