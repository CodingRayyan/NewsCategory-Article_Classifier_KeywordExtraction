import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import streamlit as st
import pickle

# Load the model and TF-IDF vectorizer
with open("model/news_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("model/tfidf_vectorizer.pkl", "rb") as tfidf_file:
    tfidf = pickle.load(tfidf_file)

st.title("📰 News Headline/Article Classification & Keywords Extraction")
st.write("")
st.write("")
st.write("Categories Available: ['ENTERTAINMENT', 'TRAVEL', 'WELLNESS', 'POLITICS', 'STYLE & BEAUTY']")
st.write("Enter a news headline/article and get it's category prediction & keywords!")
st.write("")
st.write("")

#sentence = st.text_input("Enter a headline:")
sentence = st.text_area("Enter a news headline or article:")

if st.button("Predict Category"):
    if sentence:
        # Transform the input sentence
        sentence_vector = tfidf.transform([sentence])

        # Predict the category
        prediction = model.predict(sentence_vector)[0]

        st.success(f"**Predicted Category:** {prediction}")
        st.write("")

        # Extract top 20 keywords based on TF-IDF scores
        feature_names = tfidf.get_feature_names_out()
        tfidf_scores = sentence_vector.toarray().flatten()

        keyword_scores = dict(zip(feature_names, tfidf_scores))
        sorted_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)

        top_keywords = [kw for kw, score in sorted_keywords if score > 0][:50]

        if top_keywords:
            st.write("")
            st.markdown("### 🔑 Top Keywords Extracted:")
            st.write("")
            st.write(", ".join(top_keywords))
        else:
            st.warning("No significant keywords extracted. Try a longer sentence or paragraph.")
    else:
        st.warning("⚠️ Please enter a headline before clicking Predict!")


st.write("")
st.write("")
st.markdown("<hr style='border: 2px solid red;'>", unsafe_allow_html=True)
st.write("")
st.write("")

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter

df = pd.read_csv("data/NewsCategorizer21.csv")

# -------------------------------- 1st Row: Plot 1 and Plot 2 --------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Category Distribution - Pie Chart")

    category_counts = df['category'].value_counts()
    labels = category_counts.index
    sizes = category_counts.values
    explode = [0.1] * len(sizes)

    fig1, ax1 = plt.subplots(figsize=(5, 5))
    ax1.pie(
        sizes, 
        labels=labels, 
        autopct='%1.3f%%', 
        shadow=True,
        explode=explode
    )
    ax1.set_title("\nCategory Distribution")
    st.pyplot(fig1)

with col2:
    st.subheader("📈 Category Distribution - Countplot")

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.countplot(
        x=df['category'], 
        order=df['category'].value_counts().index,
        ax=ax2
    )
    ax2.set_title("Category Distribution")
    ax2.set_xlabel("Category")
    ax2.set_ylabel("Count")
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(color='black')

    st.pyplot(fig2)

st.write("")
# -------------------------------- 2nd Row: Plot 3 and Plot 4 --------------------------------------
st.write("")

col3, col4 = st.columns(2)

with col3:
    st.subheader("✨ Enhanced Category Count with Labels")

    fig3, ax3 = plt.subplots(figsize=(9, 3))
    sns.set_style("darkgrid")

    ax3 = sns.countplot(
        x=df['category'], 
        order=df['category'].value_counts().index, 
        palette="viridis",  
        edgecolor="black"
    )

    ax3.set_title("Distribution of News Categories", fontsize=16, fontweight="bold", color="darkred")
    ax3.set_xlabel("Category", fontsize=14, fontweight="bold", color="darkblue")
    ax3.set_ylabel("Count", fontsize=14, fontweight="bold", color="darkblue")
    plt.xticks(rotation=45, ha="right", fontsize=12)

    for p in ax3.patches:
        ax3.annotate(f'{int(p.get_height())}', 
                     (p.get_x() + p.get_width() / 2, p.get_height()), 
                     ha='center', va='bottom', fontsize=12, color="black", fontweight="bold")

    plt.grid(color='red')
    st.pyplot(fig3)

with col4:
    st.subheader("🔑 Top 15 Most Frequent Keywords")

    keywords = " ".join(df['keywords'].dropna().astype(str)).split("-")
    top_keywords = dict(Counter(keywords).most_common(15))

    fig4, ax4 = plt.subplots(figsize=(12, 6))
    sns.barplot(x=list(top_keywords.keys()), y=list(top_keywords.values()), palette="magma", ax=ax4)
    plt.xticks(rotation=45)
    plt.xlabel("Keywords")
    plt.ylabel("Frequency")
    plt.title("Top 15 Most Frequent Keywords")

    st.pyplot(fig4)

# Developers dropdown
with st.sidebar.expander("👨‍💻 Developers"):
    st.markdown("- **Rayyan Ahmed (22F-BSAI-11)**")
    st.markdown("- **Wajahat Tariq (22F-BSAI-17)**")
    st.markdown("- **Muhammed Sami (22F-BSAI-43)**")

with st.sidebar.expander("👨‍💻 Rayyan's Intro"):
    st.markdown("- **IBM Certifed Advanced LLM FineTuner**")
    st.markdown("- **Google Certified Soft Skill Professional**")
    st.markdown("- **Hugging Face Certified in Fundamentals of Large Language Models (LLMs)**")
    st.markdown("- **Have expertise in EDA, ML, Reinforcement Learning, ANN, CNN, CV, RNN, NLP, LLMs.**")
    st.markdown("[💼Visit Rayyan's LinkedIn Profile](https://www.linkedin.com/in/rayyan-ahmed-504725321/)")

with st.sidebar.expander("👨‍💻 Wajahat's Intro"):
    st.markdown("- **2 years of experienced Graphic Designer**")
    st.markdown("- **Web Developer**")
    st.markdown("- **Have expertise in Python3, Data Analysis, EDA, ML, ANN.**")
    st.markdown("[💼Visit Wajahat's LinkedIn Profile](https://www.linkedin.com/in/muhammad-wajahat-tariq/?originalSubdomain=pk)")

with st.sidebar.expander("👨‍💻 Sami's Intro"):
    st.markdown("- **Jawan Pakistan Certified Front Web Developer**")
    st.markdown("- **SMIT Certified Data Analyst**")
    st.markdown("[💼Visit Sami's LinkedIn Profile](https://www.linkedin.com/in/sami-rajput19/)")

# Libraries used dropdown
with st.sidebar.expander("📦 Libraries Used"):
    st.markdown("- **Streamlit**")
    st.markdown("- **TensorFlow**")
    st.markdown("- **scikit-learn**")
    st.markdown("- **Pandas**")
    st.markdown("- **Matplotlib**")
    st.markdown("- **Seaborn**")
    st.markdown("- **Pickle**")

with st.sidebar.expander("📚 References Used"):
    st.markdown("[🔗NewsCatgory DataSet](https://www.kaggle.com/datasets/timilsinabimal/newsarticlecategories)")
    st.markdown("[🔗SkLearn.org TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)")
    st.markdown("[🔗Streamlit Docs](https://docs.streamlit.io/develop/tutorials)")
    st.markdown("[🔗Rayyan's Github](https://github.com/CodingRayyan)")
    st.markdown("[🔗Plotting Docs](https://matplotlib.org/stable/plot_types/index.html)")

with st.sidebar.expander("📝 Feedback"):
    st.markdown("We appreciate your feedback!")
    feedback_text = st.text_area("Please write your feedback here:")
    if st.button("Submit Feedback"):
        if feedback_text.strip():
            st.success("Thank you for your feedback!")
            # Here, you can add code to save or send feedback, e.g., save to file, send email, etc.
        else:
            st.warning("Please enter some feedback before submitting.")


st.write("")
st.write("")
st.markdown("<hr style='border: 2px solid red;'>", unsafe_allow_html=True)
st.write("")
st.write("")

st.markdown("**Advanced features coming soon...**")