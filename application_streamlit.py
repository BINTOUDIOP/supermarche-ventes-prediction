import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import nltk
from nltk.corpus import stopwords
from collections import Counter

# nltk.download('punkt')
# nltk.download('stopwords')

# Configuration de la page
st.set_page_config(
    page_title="Analyse des ventes Favorita",
    layout="wide",
    initial_sidebar_state="expanded"
)

# En-tête
st.title("📊 Supermarchés Favorita - Analyse des Ventes")
st.markdown("Bienvenue dans cette application d'analyse interactive des ventes ! "
            "Ce tableau de bord vous permet d'explorer les données historiques de ventes de la chaîne Favorita "
            "en Équateur, issues d'un contexte réel. Vous pouvez filtrer, visualiser et comprendre les tendances de consommation.")

# Chargement des données
@st.cache_data
def load_data():
    return pd.read_csv("data_traites.csv", parse_dates=["date"])

df = load_data()

# Sidebar : Filtres
st.sidebar.header("🔍 Filtres interactifs")
villes = st.sidebar.multiselect("📍 Ville :", options=df["city"].unique(), default=df["city"].unique())
types = st.sidebar.multiselect("🏬 Type de magasin :", options=df["type_x"].unique(), default=df["type_x"].unique())
jours = st.sidebar.multiselect("📆 Jour de la semaine :", options=df["day_of_week"].unique(), default=df["day_of_week"].unique())

# Application des filtres
df_filtered = df[
    df["city"].isin(villes) &
    df["type_x"].isin(types) &
    df["day_of_week"].isin(jours)
]

# Informations générales
st.markdown("### ℹ️ Aperçu des données filtrées")
st.dataframe(df_filtered.head(10))

col1, col2 = st.columns(2)
with col1:
    st.metric("📦 Nombre de lignes", len(df_filtered))
with col2:
    st.metric("🗓️ Période couverte", f"{df_filtered['date'].min().date()} → {df_filtered['date'].max().date()}")

# Graphique 1 : Ventes dans le temps
st.markdown("### 1. Évolution des ventes dans le temps")
fig1, ax1 = plt.subplots(figsize=(12, 4))
df_filtered.groupby("date")["sales"].sum().plot(ax=ax1, color="tab:blue")
ax1.set_title("Total des ventes par jour")
ax1.set_xlabel("Date")
ax1.set_ylabel("Ventes")
st.pyplot(fig1)
st.caption("Ce graphique permet de visualiser les pics de ventes et la saisonnalité.")

# Graphique 2 : Ventes par jour de la semaine
st.markdown("### 2. Répartition des ventes par jour de la semaine")
fig2, ax2 = plt.subplots()
sns.boxplot(data=df_filtered, x="day_of_week", y="sales", ax=ax2, palette="Pastel1")
ax2.set_title("Distribution des ventes par jour")
st.pyplot(fig2)
st.caption("On observe les jours avec les plus fortes variations de ventes.")

# Graphique 3 : Ventes par ville
st.markdown("### 3. Répartition géographique des ventes")
fig3, ax3 = plt.subplots(figsize=(10, 5))
df_filtered.groupby("city")["sales"].sum().sort_values().plot(kind="barh", ax=ax3, color="mediumseagreen")
ax3.set_title("Total des ventes par ville")
st.pyplot(fig3)

# Graphique 4 : Ventes par type de magasin
st.markdown("### 4. Ventes par type de magasin")
fig4, ax4 = plt.subplots()
df_filtered.groupby("type_x")["sales"].sum().sort_values().plot(kind="bar", ax=ax4, color="orchid")
ax4.set_title("Ventes totales par type de magasin")
st.pyplot(fig4)

# Graphique 5 : Impact des jours fériés
st.markdown("### 5. Impact des jours fériés sur les ventes")
fig5, ax5 = plt.subplots()
sns.boxplot(data=df_filtered, x="is_holiday", y="sales", ax=ax5, palette="Set2")
ax5.set_xticklabels(["Jour normal", "Jour férié"])
ax5.set_title("Comparaison des ventes : jours fériés vs jours classiques")
st.pyplot(fig5)

# Graphique 6 : Ventes avec ou sans promotion
st.markdown("### 6. Impact des promotions sur les ventes")
df_filtered["onpromotion"] = df_filtered["onpromotion"].fillna(0).astype(bool)
moy_promo = df_filtered.groupby("onpromotion")["sales"].mean()
fig6, ax6 = plt.subplots()
moy_promo.plot(kind="bar", ax=ax6, color=["lightblue", "salmon"])
ax6.set_xticklabels(["Sans promo", "Avec promo"], rotation=0)
ax6.set_title("Vente moyenne selon la présence de promotion")
ax6.set_ylabel("Ventes moyennes")
st.pyplot(fig6)
st.caption("Ce graphique montre que les produits en promotion sont généralement mieux vendus.")

# Graphique 7 : Wordcloud - article de presse (identique au notebook)
st.markdown("### 7. 📰 Nuage de mots à partir d’un article de presse")
st.markdown("Ce wordcloud est basé sur le même article de presse utilisé dans notre notebook, pour garantir la cohérence.")

with st.expander("📄 Afficher le Wordcloud"):
    article_text = """
    En janvier 2022, hausse du chiffre d’affaires des grandes surfaces alimentaires
    En janvier 2022, le chiffre d’affaires total des grandes surfaces alimentaires (y compris courses en ligne, drive) repart à la hausse (+2,2 %) après une stabilité en décembre 2021. Les ventes en magasin sont en hausse à la fois pour les produits alimentaires (+1,3 % après +1,1 %) et les produits non alimentaires (+1,8 % après −1,7 %). Concernant les ventes de carburants, le chiffre d'affaires rebondit en janvier (+2,1 % après −3,0 %).
    Le chiffre d’affaires des ventes en magasin accélère à la fois dans dans les supermarchés (+1,6 % après +0,6 %) et les hypermarchés (+1,0 % après +0,6 %).
    Hausse du chiffre d’affaires sur un an (+6,1 %)
    Le chiffre d’affaires réalisé par les grandes surfaces alimentaires au cours des trois derniers mois (novembre 2021 à janvier 2022) est en hausse (+6,1 %) par rapport à la même période un an plus tôt. Les ventes diminuent dans les produits alimentaires (−0,6 %). Elles augmentent en revanche dans les produits non alimentaires (+5,0 %), en raison d'un effet de base lié à la fermeture des rayons de produits considérés comme « non essentiels » lors du deuxième confinement, du 29 octobre au 15 décembre 2020. Les ventes de carburants croissent très fortement (+53,0 %), en raison d'une part de ce même effet de base lié au deuxième confinement et, d'autre part, de la hausse des prix.
    En rythme annuel, le chiffre d’affaires de novembre 2021 à janvier 2022 augmente à la fois dans les supermarchés (+5,9 %) et les hypermarchés (+7,8 %).
    """
    article_text = article_text.lower()
    article_text = re.sub(r'[^a-z\s]', '', article_text)
    tokens = re.findall(r'\b\w+\b', article_text)
    stop_words = set(stopwords.words('french'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    word_counts = Counter(filtered_tokens)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(filtered_tokens))

    fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
    ax_wc.imshow(wordcloud, interpolation='bilinear')
    ax_wc.axis("off")
    st.pyplot(fig_wc)
    st.caption("Ce nuage de mots est généré à partir de l’article analysé dans le notebook, garantissant la cohérence entre les deux supports.")

# Footer
st.markdown("---")
st.success("Les visualisations interactives permettent d’explorer en profondeur les comportements d’achat dans les supermarchés Favorita.")