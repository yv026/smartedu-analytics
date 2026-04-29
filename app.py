import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.title("🎓 SmartEdu Analytics - Analyse des performances étudiantes")

FILE = "data.csv"

# Charger données
if os.path.exists(FILE):
    df = pd.read_csv(FILE)
else:
    df = pd.DataFrame(columns=["Age", "Heures étude", "Téléphone", "Sommeil", "Moyenne"])

# FORMULAIRE
with st.form("form"):
    age = st.number_input("Âge", 15, 40)
    heures = st.number_input("Heures d'étude", 0, 15)
    telephone = st.number_input("Temps téléphone", 0, 15)
    sommeil = st.number_input("Heures de sommeil", 0, 12)
    moyenne = st.number_input("Moyenne", 0.0, 20.0)

    submit = st.form_submit_button("Enregistrer")

# ENREGISTREMENT
if submit:
    new_data = pd.DataFrame([{
        "Age": age,
        "Heures étude": heures,
        "Téléphone": telephone,
        "Sommeil": sommeil,
        "Moyenne": moyenne
    }])

    df = pd.concat([df, new_data], ignore_index=True)
    df.to_csv(FILE, index=False)
    st.success("Données enregistrées")

# AFFICHAGE
st.write("### 📋 Données")
st.dataframe(df)

if not df.empty:

    # 📊 STATISTIQUES
    st.write("### 📊 Statistiques descriptives")
    st.write(df.describe())

    # 🔗 CORRELATION
    st.write("### 🔗 Corrélation")
    st.write(df.corr(numeric_only=True))

    # 📈 GRAPHIQUES
    st.write("### 📈 Graphiques")
    st.line_chart(df[["Heures étude", "Moyenne"]])
    st.bar_chart(df[["Téléphone", "Moyenne"]])

    # 📉 REGRESSION SIMPLE
    st.write("### 📉 Régression linéaire simple")
    X = df[["Heures étude"]]
    y = df["Moyenne"]

    if len(df) > 1:
        model = LinearRegression()
        model.fit(X, y)

        df["Prediction"] = model.predict(X)

        st.write("Coefficient :", model.coef_[0])
        st.write("Intercept :", model.intercept_)
        st.line_chart(df[["Moyenne", "Prediction"]])

    # 📉 REGRESSION MULTIPLE
    st.write("### 📉 Régression multiple")
    X_multi = df[["Heures étude", "Téléphone", "Sommeil"]]

    if len(df) > 2:
        model_multi = LinearRegression()
        model_multi.fit(X_multi, y)

        st.write("Coefficients :", model_multi.coef_)

    # 🧩 CLASSIFICATION NON SUPERVISEE (KMeans)
    st.write("### 🧩 Clustering (K-Means)")
    if len(df) > 2:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[["Heures étude", "Téléphone", "Sommeil"]])

        kmeans = KMeans(n_clusters=3, random_state=0)
        df["Cluster"] = kmeans.fit_predict(X_scaled)

        st.dataframe(df)

    # 🧠 ACP (PCA)
    st.write("### 🧠 Réduction de dimension (ACP)")
    if len(df) > 2:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(df[["Heures étude", "Téléphone", "Sommeil"]])

        df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
        st.write(df_pca)
