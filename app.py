import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# Membuat judul
st.title("Titanic Prediction Apps")

# Membuat informasi aplikasi
st.info("Ini adalah aplikasi machine learning prediksi kapal **Titanic** made by **Izzan Faikar Ramadhy**")

# Load data titanic
with st.expander("**Data**"):
    df = pd.read_csv("titanic.csv")
    df_copy = df.copy()
    # Preprocessing data
    # Merubah value data
    df_copy['Survived'] = df_copy['Survived'].replace({0: "Tidak Selamat", 1: "Selamat"})
    df_copy['Pclass'] = df_copy['Pclass'].replace({1: "Jelata", 2: "Normal", 3: "Bangsawan"})

    # Menghapus data
    df_copy.drop(["PassengerId","Name","Cabin", "Ticket","Embarked"], axis = 1, inplace = True)

    # Handling missing values
    df_copy["Age"].fillna(df_copy["Age"].median(), inplace = True)
    df_copy

with st.expander("**Data Information**"):
    kolom1, kolom2, kolom3, kolom4 = st.columns(4, gap = "small")
    if kolom1.button("**1. Survived**", help ="Informasi Jumlah Keselamatan Penumpang"):
        # Hitung jumlah setiap kategori di kolom "Survived"
        count_survived= df_copy['Survived'].value_counts().reset_index()
        count_survived.columns = ['Survived', 'count']
        # Visualisasikan
        st.bar_chart(count_survived, x = "Survived", y = "count", stack = False, use_container_width = True , height = 500, y_label = "Jumlah", horizontal = True)
    if kolom2.button("**2. Gender vs Age**"):
        st.bar_chart(df_copy, x = "Sex", y = "Age", stack = False, use_container_width = True, height = 500)
    if kolom3.button("**3. Age vs Fare**"):
        st.scatter_chart(df_copy, x = "Fare", y = "Age", color = "Sex", use_container_width = True, height = 500)
    if kolom4.button("**4. Age vs Fare**"):
        st.line_chart(df_copy, x = "Age", y = "Fare", use_container_width = True, height = 500)

with st.sidebar:
    st.header("Input Features")
    # Tombol Gender
    selamat = st.selectbox("Kondisi Penumpang", ("Selamat", "Tidak Selamat"))
    kelas = st.selectbox("Kelas Penumpang", ("Jelata", "Normal", "Bangsawan"))
    gender = st.selectbox("Gender", ("male", "female"))
    umur = st.slider("Umur", 0, 80, 0)
    saudara = st.slider("Jumlah Sepupu/Saudara Ikut", 0, 8, 0)
    ortu_anak = st.slider("Jumlah Orang Tua/Anak Ikut", 0, 6, 0)
    tiket = st.slider("Harga Tiket (Fare)", 0, 520, 0)

with st.expander("**Input Features**"):
    # ================================
    # FILTER DATA
    # ================================        
    filtered_df = df_copy.copy()  # Mulai dengan data lengkap

    # Terapkan filter jika bukan "Semua"
    if selamat != "Semua":
        filtered_df = filtered_df[filtered_df["Survived"] == selamat]

    if kelas != "Semua":
        filtered_df = filtered_df[filtered_df["Pclass"] == kelas]

    if gender != "Semua":
        filtered_df = filtered_df[filtered_df["Sex"] == gender]

    filtered_df = filtered_df[filtered_df["Age"] <= umur]
    filtered_df = filtered_df[filtered_df["Fare"] <= tiket]

    # ================================
    # MENAMPILKAN DATA YANG TELAH DIFILTER
    # ================================
    st.subheader("Hasil Filter Data")
    st.write(f"Menampilkan {filtered_df.shape[0]} dari {df_copy.shape[0]} data")
    st.dataframe(filtered_df)

# Data Preparation for Machine Learning
with st.expander("**Data Preparation**"):
    # Prepare Again
    df.drop(["PassengerId","Name","Cabin", "Ticket","Embarked"], axis = 1, inplace = True)
    df["Age"].fillna(df["Age"].median(), inplace = True)

    # Encode Data
    label_encoder = LabelEncoder()
    df['Sex'] = label_encoder.fit_transform(df['Sex'])

    # Tampilkan data yang sudah di-encode
    st.subheader("Data yang Sudah Di-Encode")
    df

# ==========================================================================================
# =================================== Training Model =======================================
# ==========================================================================================

with st.expander("**Training Model**"):
# Pisahkan fitur dan target
    X = df.drop('Survived', axis=1)  # Fitur (X)
    y = df['Survived']  # Target (y)

    # Split data menjadi data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)

    # Panggil model
    logModel = LogisticRegression()

    # Latih model
    logModel.fit(X_train, y_train)

    # Evaluasi Model
    y_pred = logModel.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    st.subheader(f"Akurasi Model: {accuracy*100:.2f}%")

# ==========================================================================================
# ================================== Prediksi Interaktif ===================================
# ==========================================================================================

with st.expander("**Prediction**"):
    st.header("🎯 Prediksi Keselamatan Penumpang Titanic")

    # Form Input Data untuk Prediksi
    st.subheader("Masukkan Data Penumpang")

    col1, col2 = st.columns(2)

    with col1:
        gender_input = st.selectbox("Gender", ("male", "female"), key="gender_input")
        kelas_input = st.selectbox("Kelas Penumpang", ("Jelata", "Normal", "Bangsawan"), key="kelas_input")
        umur_input = st.slider("Umur", 0, 80, 22, key="umur_input")

    with col2:
        saudara_input = st.slider("Jumlah Saudara/Sepupu Ikut", 0, 8, 0, key="saudara_input")
        ortu_anak_input = st.slider("Jumlah Orang Tua/Anak Ikut", 0, 6, 0, key="ortu_anak_input")
        tiket_input = st.slider("Harga Tiket (Fare)", 0, 520, 50, key="tiket_input")

    # Tombol Prediksi
    if st.button("Prediksi Keselamatan 🚢"):
        # ================================
        # PREPROCESSING INPUT DATA
        # ================================
        
        # Konversi Gender ke angka (0 = female, 1 = male)
        gender_encoded = 1 if gender_input == "male" else 0

        # Konversi Kelas ke angka
        kelas_mapping = {"Jelata": 3, "Normal": 2, "Bangsawan": 1}
        kelas_encoded = kelas_mapping[kelas_input]

        # Buat DataFrame dengan format yang sesuai untuk prediksi
        input_data = pd.DataFrame([[kelas_encoded, gender_encoded, umur_input, saudara_input, ortu_anak_input, tiket_input]],
                                columns=["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"])

        # ================================
        # MELAKUKAN PREDIKSI
        # ================================
        prediksi = logModel.predict(input_data)
        probabilitas = logModel.predict_proba(input_data)[0][1]  # Probabilitas selamat

        # ================================
        # MENAMPILKAN HASIL PREDIKSI
        # ================================
        st.subheader("📝 Hasil Prediksi:")
        if prediksi[0] == 1:
            st.success(f"✅ Penumpang diprediksi **Selamat** dengan probabilitas {probabilitas*100:.2f}%")
        else:
            st.error(f"❌ Penumpang diprediksi **Tidak Selamat** dengan probabilitas {(1 - probabilitas)*100:.2f}%")
