import os
import streamlit as st
from dotenv import load_dotenv
import dashscope
from dashscope import Generation
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables
load_dotenv()
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

if not DASHSCOPE_API_KEY:
    st.error("❌ DASHSCOPE_API_KEY tidak ditemukan! Pastikan file .env berisi:\n\nDASHSCOPE_API_KEY=sk-...")
    st.stop()

# Set API key globally untuk dashscope
dashscope.api_key = DASHSCOPE_API_KEY

SYSTEM_PROMPT = """
Anda adalah “Asisten Keuangan UMKM”, sebuah chatbot profesional berbasis kecerdasan buatan 
yang bertugas memberikan edukasi keuangan, menjelaskan faktor-faktor pinjaman, dan membantu 
pengguna memahami proses simulasi approval dalam aplikasi “Simulasi Pinjaman UMKM”.

Peran: Konsultan pembiayaan UMKM, edukator literasi keuangan, pendamping pengajuan pinjaman.

Jangan menjawab hal di luar konteks edukasi keuangan dan pinjaman UMKM.
"""

def call_qwen(messages):
    """
    Memanggil model Qwen melalui DashScope SDK.
    Input: list of dict [{"role": "user/system/assistant", "content": "..."}]
    Output: string (jawaban AI)
    """
    # Konversi format LangChain ke format DashScope
    dash_messages = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            dash_messages.append({"role": "system", "content": msg.content})
        elif isinstance(msg, HumanMessage):
            dash_messages.append({"role": "user", "content": msg.content})
        else:
            # Untuk AI message dari histori
            dash_messages.append({"role": "assistant", "content": msg.content})

    try:
        response = Generation.call(
            model="qwen2.5-7b-instruct",
            messages=dash_messages,
            result_format="message"
        )
        return response.output.choices[0].message.content
    except Exception as e:
        raise Exception(f"Error dari DashScope: {str(e)}")

def main():
    st.title("💬 Chatbot Konsultasi Pinjaman (Qwen)")

    # Simpan histori chat
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content=SYSTEM_PROMPT)
        ]

    # Tampilkan riwayat chat (kecuali system prompt)
    for msg in st.session_state.messages[1:]:
        role = "assistant" if msg.type == "ai" else "user"
        with st.chat_message(role):
            st.write(msg.content)

    # Input pengguna
    user_input = st.chat_input("Ketik pertanyaan Anda...")

    if user_input:
        # Tampilkan pesan pengguna
        with st.chat_message("user"):
            st.write(user_input)

        # Tambahkan ke histori
        st.session_state.messages.append(HumanMessage(content=user_input))

        try:
            # Panggil Qwen
            ai_response = call_qwen(st.session_state.messages)

            # Tampilkan jawaban
            with st.chat_message("assistant"):
                st.write(ai_response)

            # Simpan sebagai objek AI (agar bisa di-serialize LangChain)
            from langchain_core.messages import AIMessage
            st.session_state.messages.append(AIMessage(content=ai_response))

        except Exception as e:
            st.error(f"❌ Terjadi error saat memanggil Qwen:\n\n{e}")

if __name__ == "__main__":
    main()