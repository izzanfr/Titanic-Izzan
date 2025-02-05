import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier

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

# Pisahkan fitur dan target
X = df.drop('Survived', axis=1)  # Fitur (X)
y = df['Survived']  # Target (y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

# Inisialisasi model sebagai None
trained_model = None

with st.expander("**Training Model**"):
    st.subheader("Konfigurasi Data")
    data_split = st.slider("Pilih Ukuran Test Data (%):", min_value=10, max_value=50, value=25, step=5)
    test_size = data_split / 100.0

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)

    st.subheader("Konfigurasi Model")
    model_choice = st.radio("Pilih Model Machine Learning:", [
        "Logistic Regression", "Decision Tree", "SVM", "XGBoost", "KNN"
    ])
    
    # Hyperparameter configuration
    st.write("**Hyperparameter Configuration**")
    if model_choice == "Decision Tree":
        max_depth = st.slider("Max Depth", 1, 20, 5)
        model = DecisionTreeClassifier(max_depth=max_depth)
    elif model_choice == "SVM":
        C = st.slider("C (Regularization Parameter)", 0.01, 10.0, 1.0)
        model = SVC(C=C, probability=True)
    elif model_choice == "XGBoost":
        learning_rate = st.slider("Learning Rate", 0.01, 0.5, 0.1, step=0.01)
        n_estimators = st.slider("Number of Estimators", 50, 500, 100, step=50)
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', learning_rate=learning_rate, n_estimators=n_estimators)
    elif model_choice == "KNN":
        n_neighbors = st.slider("Number of Neighbors", 1, 20, 5)
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
    elif model_choice == "Logistic Regression":
        model = LogisticRegression()
    
    if st.button("Train Model"):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.subheader(f"Akurasi Model {model_choice}: {accuracy*100:.2f}%")
        st.session_state["trained_model"] = model  # Simpan model ke dalam session state

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
        if st.session_state["trained_model"] is None:
            st.error("Harap latih model terlebih dahulu di bagian Training Model.")
        else:
            # Konversi Gender ke angka (0 = female, 1 = male)
            gender_encoded = 1 if gender_input == "male" else 0

            # Konversi Kelas ke angka
            kelas_mapping = {"Jelata": 3, "Normal": 2, "Bangsawan": 1}
            kelas_encoded = kelas_mapping[kelas_input]

            # Buat DataFrame dengan format yang sesuai untuk prediksi
            input_data = pd.DataFrame([[kelas_encoded, gender_encoded, umur_input, saudara_input, ortu_anak_input, tiket_input]],
                                    columns=["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"])
            
            # Melakukan prediksi
            prediksi = st.session_state["trained_model"].predict(input_data)
            probabilitas = st.session_state["trained_model"].predict_proba(input_data)[0]

            # Menampilkan hasil prediksi
            st.subheader("📝 Hasil Prediksi:")
            if prediksi[0] == 1:
                st.success(f"✅ Penumpang diprediksi **Selamat** dengan probabilitas {probabilitas[1] * 100:.2f}%")
            else:
                st.error(f"❌ Penumpang diprediksi **Tidak Selamat** dengan probabilitas {probabilitas[0] * 100:.2f}%")
            
            # # Fitur penting dalam prediksi 
            # st.subheader("🤔 Fitur Penting dalam Prediksi:")
            # feature_importance = None
            # if hasattr(st.session_state["trained_model"], "feature_importances_"):
            #     feature_importance = st.session_state["trained_model"].feature_importances_
            # elif hasattr(st.session_state["trained_model"], "coef_"):
            #     feature_importance = abs(st.session_state["trained_model"].coef_[0])

            # if feature_importance is not None:
            #     feature_importance_df = pd.DataFrame({"Feature": input_data.columns, "Importance": feature_importance})
            #     feature_importance_df.sort_values(by="Importance", ascending=False, inplace=True)
            #     st.write("Fitur yang paling memengaruhi prediksi:")
            #     st.dataframe(feature_importance_df)
            # else:
            #     st.write("Model ini tidak mendukung penjelasan fitur.")

            # Visualisasi dengan progress bar untuk semua kelas
            st.subheader("📊 Visualisasi Prediksi")
            col1, col2 = st.columns(2)
            with col1:
                st.text("Selamat")
                st.progress(float(probabilitas[1]))
                st.write(f"{probabilitas[1] * 100:.2f}%")
            with col2:
                st.text("Tidak Selamat")
                st.progress(float(probabilitas[0]))
                st.write(f"{probabilitas[0] * 100:.2f}%")