import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ===============================
# LOAD MODEL & ARTIFACTS
# ===============================
@st.cache_resource
def load_artifacts():
    model = joblib.load("best_loan_approval_model_reduced.pkl")
    scaler = joblib.load("scaler.pkl")
    encoders = joblib.load("label_encoders.pkl")  # dict: kolom -> LabelEncoder
    return model, scaler, encoders

model, scaler, encoders = load_artifacts()

# Load selected features (important_features dari notebook)
with open("selected_features.txt", "r") as f:
    selected_features = [line.strip() for line in f.readlines() if line.strip()]

# ===============================
# HELPER FUNCTIONS
# ===============================
def safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default

def safe_int(v, default=0):
    try:
        if pd.isna(v):
            return default
        return int(v)
    except Exception:
        return default


# ===============================
# STREAMLIT UI
# ===============================
def main():

    st.title("📊 Loan Approval Prediction")
    st.write("Isi form berikut untuk menghasilkan prediksi persetujuan pinjaman.")

    with st.form("loan_form"):
        user_input = {}
        st.subheader("Isi Data Calon Peminjam")

        # NOTE:
        # Kita loop berdasarkan selected_features (fitur yang dipakai model_reduced)
        for feat in selected_features:

            # --------- SKIP FEATURE YG TIDAK LOGIS DIISI USER ---------
            if feat == "customer_id":
                # customer_id akan diisi default (kelas pertama encoder) saat preprocessing
                continue

            # ========== CUSTOM FIELD UNTUK BEBERAPA FITUR ==========
            if feat == "occupation_status":
                # Pakai kelas dari encoder supaya mapping sama persis
                if feat in encoders:
                    classes = list(encoders[feat].classes_)
                    user_input[feat] = st.selectbox("Occupation Status", classes)
                else:
                    user_input[feat] = st.selectbox(
                        "Occupation Status",
                        ["Employed", "Student", "Self-Employed"],
                    )

            elif feat == "years_employed":
                user_input[feat] = st.number_input(
                    "Years Employed", min_value=0.0, max_value=100.0, value=0.0, step=0.1
                )

            elif feat == "annual_income":
                user_input[feat] = st.number_input(
                    "Annual Income", min_value=0, value=30000, step=1000
                )

            elif feat == "credit_score":
                user_input[feat] = st.number_input(
                    "Credit Score", min_value=300, max_value=850, value=650, step=1
                )

            elif feat in ("credit_history_years", "credit_history years"):
                user_input[feat] = st.number_input(
                    "Credit History (Years)",
                    min_value=0.0,
                    max_value=100.0,
                    value=1.0,
                    step=0.1,
                )

            elif feat in ("savings_assets", "savings_asset"):
                user_input[feat] = st.number_input(
                    "Savings Assets", min_value=0, value=1000, step=100
                )

            elif feat == "current_debt":
                user_input[feat] = st.number_input(
                    "Current Debt", min_value=0, value=0, step=100
                )

            elif feat == "defaults_on_file":
                sel = st.selectbox("Defaults on File", ["No", "Yes"])
                user_input[feat] = 1 if sel == "Yes" else 0

            elif feat == "delinquencies_last_2yrs":
                user_input[feat] = st.number_input(
                    "Delinquencies Last 2 Years", min_value=0, value=0, step=1
                )

            elif feat == "derogatory_marks":
                user_input[feat] = st.number_input(
                    "Derogatory Marks", min_value=0, value=0, step=1
                )

            elif feat == "product_type":
                if feat in encoders:
                    classes = list(encoders[feat].classes_)
                    user_input[feat] = st.selectbox("Product Type", classes)
                else:
                    user_input[feat] = st.selectbox(
                        "Product Type",
                        ["Credit Card", "Line of Credit", "Personal Loan"],
                    )

            elif feat == "loan_intent":
                if feat in encoders:
                    classes = list(encoders[feat].classes_)
                    user_input[feat] = st.selectbox("Loan Intent", classes)
                else:
                    user_input[feat] = st.selectbox(
                        "Loan Intent",
                        [
                            "Business",
                            "Home Improvement",
                            "Debt Consolidation",
                            "Education",
                            "Medical",
                            "Personal",
                        ],
                    )

            elif feat == "loan_amount":
                user_input[feat] = st.number_input(
                    "Loan Amount", min_value=0, value=5000, step=100
                )

            elif feat == "interest_rate":
                user_input[feat] = st.number_input(
                    "Interest Rate (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=10.0,
                    step=0.01,
                )

            elif feat in (
                "debt_to_income_ratio",
                "loan_to_income_ratio",
                "payment_to_income_ratio",
            ):
                label = feat.replace("_", " ").title()
                user_input[feat] = st.number_input(
                    label,
                    min_value=0.0,
                    max_value=10.0,
                    value=0.200,
                    step=0.001,
                    format="%.3f"
                )

            # ========== GENERIC: KALAU ADA ENCODER LAIN ==========
            elif feat in encoders:
                classes = list(encoders[feat].classes_)
                user_input[feat] = st.selectbox(
                    feat.replace("_", " ").title(), classes
                )

            # ========== GENERIC NUMERIC ==========
            else:
                user_input[feat] = st.number_input(
                    feat.replace("_", " ").title(), value=0.0
                )

        submitted = st.form_submit_button("Predict")

    # ===============================
    # SAAT TOMBOL PREDICT DIKLIK
    # ===============================
    if not submitted:
        return

    # 1) FIX TIPE DATA RAW
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
            user_input[k] = safe_float(user_input[k])
        elif k in numeric_int_feats:
            user_input[k] = safe_int(user_input[k])
        # kolom kategorikal (occupation_status, product_type, loan_intent) dibiarkan string

    # 2) BANGUN DATAFRAME DARI INPUT USER
    df_user = pd.DataFrame([user_input])

    # 3) HANDLE ALIAS NAMA KOLOM (JAGA-JAGA)
    # credit_history years -> credit_history_years
    if "credit_history years" in df_user.columns and "credit_history_years" in selected_features:
        df_user["credit_history_years"] = df_user["credit_history years"]

    # savings_asset -> savings_assets
    if "savings_asset" in df_user.columns and "savings_assets" in selected_features:
        df_user["savings_assets"] = df_user["savings_asset"]

    # 4) BANGUN FULL FEATURE SET SESUAI SCALER
    feature_names = getattr(scaler, "feature_names_in_", None)
    if feature_names is None:
        # Fallback: pakai gabungan kolom df_user + encoders keys + selected_features
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
            # kalau kolom punya encoder (kategorikal) tapi user tidak isi:
            if col in encoders:
                # isi default dengan kelas pertama supaya aman (tidak unseen)
                full_row[col] = encoders[col].classes_[0]
            else:
                # numeric yang tidak diisi user: default 0
                full_row[col] = 0.0

    df_full = pd.DataFrame([full_row])

    # 5) APPLY LABEL ENCODERS KE SEMUA KOLOM KATEGORIKAL
    for col, encoder in encoders.items():
        if col in df_full.columns:
            try:
                df_full[col] = encoder.transform(df_full[col].astype(str))
            except Exception as e:
                st.error(f"Error saat encoding kolom '{col}': {e}")
                st.write("Nilai kolom tersebut:", df_full[col].unique())
                st.stop()

    # 6) FINAL CHECK: TIDAK BOLEH ADA dtype=object
    bad_cols = [c for c in df_full.columns if df_full[c].dtype == object]
    if bad_cols:
        st.error(f"Masih ada kolom kategorikal yang belum di-encode: {bad_cols}")
        st.dataframe(df_full[bad_cols])
        st.stop()

    # 7) SCALING: PAKAI SCALER YANG SAMA SEPERTI TRAINING
    try:
        arr_scaled = scaler.transform(df_full[feature_names])
    except Exception as e:
        st.error(f"Error saat scaling dengan scaler: {e}")
        st.write("DataFrame sebelum scaling (urut feature_names):")
        st.dataframe(df_full[feature_names])
        st.stop()

    df_scaled_full = pd.DataFrame(arr_scaled, columns=feature_names)

    # 8) AMBIL selected_features UNTUK MODEL
    missing_cols = [c for c in selected_features if c not in df_scaled_full.columns]
    if missing_cols:
        st.error(f"❌ Ada fitur yang dibutuhkan model tetapi tidak ada setelah scaling: {missing_cols}")
        st.stop()

    df_model = df_scaled_full[selected_features]

    # 9) PREDIKSI
    try:
        pred = model.predict(df_model)[0]
        prob = model.predict_proba(df_model)[0]
    except Exception as e:
        st.error(f"Error saat prediksi: {e}")
        st.write("DataFrame yang dikirim ke model:")
        st.dataframe(df_model)
        st.stop()

    # 10) SIMPAN HASIL DAN NAVIGASI KE HALAMAN RESULT
    st.session_state.prediction_result = {
        "pred": pred,
        "prob": prob,
        "user_input": user_input
    }
    st.switch_page("pages/3_Result.py")


if __name__ == "__main__":
    main()
