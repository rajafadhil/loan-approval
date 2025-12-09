import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("API Key tidak ditemukan! Pastikan ada GOOGLE_API_KEY di file .env")
    st.stop()

# Model
def load_gemini_model():
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        google_api_key=GOOGLE_API_KEY
    )

def main():
    st.title("📊 Hasil Prediksi Pinjaman")

    # Check if prediction data exists in session state
    if "prediction_result" not in st.session_state:
        st.error("Tidak ada data prediksi. Silakan kembali ke halaman prediksi.")
        if st.button("Kembali ke Prediksi"):
            st.switch_page("pages/1_Predict.py")
        return

    pred = st.session_state.prediction_result["pred"]
    prob = st.session_state.prediction_result["prob"]
    user_input = st.session_state.prediction_result["user_input"]

    # Display result
    st.subheader("🔍 Hasil Prediksi:")
    if pred == 1:
        st.success(f"✔ Approved (Confidence: {prob[1]:.4f})")
    else:
        st.error(f"❌ Rejected (Confidence: {prob[0]:.4f})")

    # Try Again button
    if st.button("Coba Kembali"):
        st.switch_page("pages/1_Predict.py")

    # Ask AI button only if rejected
    if pred == 0:
        if st.button("Tanya AI"):
            # Load LLM
            if "llm" not in st.session_state:
                st.session_state.llm = load_gemini_model()

            # Prepare prompt for LLM
            prompt = f"""
            Berdasarkan data input pengguna berikut dan hasil prediksi yang ditolak, jelaskan secara singkat dan jelas mengapa pengajuan pinjaman ini ditolak.

            Data Input Pengguna:
            {user_input}

            Hasil Prediksi: Ditolak dengan confidence {prob[0]:.4f}

            Berikan penjelasan berdasarkan faktor-faktor umum dalam penilaian kredit, tanpa memberikan angka pasti atau simulasi model.
            """

            messages = [
                SystemMessage(content="Anda adalah asisten keuangan yang menjelaskan alasan penolakan pinjaman berdasarkan data umum."),
                HumanMessage(content=prompt)
            ]

            try:
                response = st.session_state.llm.invoke(messages)
                st.subheader("🤖 Penjelasan dari AI:")
                st.write(response.content)
            except Exception as e:
                st.error(f"❌ Terjadi error saat memanggil AI:\n\n{e}")

if __name__ == "__main__":
    main()
