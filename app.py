import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import re
from sklearn.metrics import confusion_matrix
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import streamlit as st

st.set_page_config(
    page_title="Spam AI Detector",
    page_icon="📧",
    layout="centered"
)
# LOGIN STATUS
if "logged_in" not in st.session_state:

    st.session_state.logged_in = False

# =========================
# CLEAN TEXT FUNCTION
# =========================
def clean_text(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z\s]', '', text)

    return text


# =========================
# SIDEBAR
# =========================
st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/2092/2092663.png",
    width=100
)

st.sidebar.title("Spam Detector ⭐")

st.sidebar.write("Built using Machine Learning 🤖")

st.sidebar.markdown("---")

if st.sidebar.button("🚪 Logout"):
    st.session_state.logged_in = False
    st.rerun()

threshold = st.sidebar.slider(
    "Confidence Threshold",
    0,
    100,
    50
)

model_type = st.sidebar.selectbox(
    "🤖 Select Model",
    [
        "Naive Bayes",
        "Logistic Regression",
        "SVM",
        "Random Forest"
    ]
)
st.sidebar.button("Clear Cache")

# =========================
# LOGIN PAGE
# =========================

if not st.session_state.logged_in:

    st.title("🔐 Login Page")

    username = st.text_input("Username")

    password = st.text_input(
        "Password",
        type="password"
    )

    if st.button("Login"):

        if username == "admin" and password == "1234":

            st.session_state.logged_in = True

            st.success("✅ Login Successful")

            st.rerun()

        else:

            st.error("❌ Invalid Username or Password")

    st.stop()


# =========================
# MAIN TITLE
# =========================
st.title("Spam Email Classifier")

st.subheader("Machine Learning Spam Detection System ⭐")

# 👉 Tabs should come directly under title
tab1, tab2, tab3 = st.tabs(
    ["📩 Prediction", "📊 Analytics", "📂 Dataset"]
)


# =========================
# LOAD DATASET
# =========================
data = pd.read_csv(
    "spam.csv",
    sep='\t',
    names=['label', 'message']
)

# Convert labels
data['label'] = data['label'].map({
    'ham': 0,
    'spam': 1
})

# Input and output
x = data['message'].apply(clean_text)

y = data['label']


# =========================
# VECTORIZATION
# =========================
vectorizer = TfidfVectorizer()

x = vectorizer.fit_transform(x)


# =========================
# SPLIT DATASET
# =========================
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)


# =========================
# TRAIN MODEL
# =========================
if model_type == "Naive Bayes":

    model = MultinomialNB()

elif model_type == "Logistic Regression":

    model = LogisticRegression()

elif model_type == "SVM":

    model = SVC(probability=True)

elif model_type == "Random Forest":

    model = RandomForestClassifier()

# Train selected model
model.fit(x_train, y_train)


# =========================
# ACCURACY
# =========================
y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)


# =========================
# TAB 2 → ANALYTICS
# =========================
with tab2:

    st.metric(
        "📊 Model Accuracy",
        f"{accuracy*100:.2f}%"
    )

    with tab2:
        graph1, graph2, graph3 = st.columns(3)

    # =========================
    # CONFUSION MATRIX
    # =========================
    with graph1:

        st.subheader("📊 Confusion Matrix")

        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(7, 6))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax
        )

        st.pyplot(fig)

    # =========================
    # PIE CHART
    # =========================
    with graph2:

        st.subheader("📈 Model Performance")

        labels = ["Accuracy", "Error"]

        values = [
            accuracy * 100,
            (1 - accuracy) * 100
        ]

        fig2, ax2 = plt.subplots(figsize=(7, 6))

        ax2.pie(
            values,
            labels=labels,
            autopct='%1.1f%%'
        )

        st.pyplot(fig2)

    with graph3:
        st.subheader("📊 Spam vs Ham")
        spam_count = data['label'].replace(
            {0: "Ham", 1: "Spam"}
            ).value_counts()
        st.bar_chart(spam_count)

    # =========================
    # WORD CLOUD
    # =========================
    st.subheader("☁ Spam Word Cloud")

    spam_text = " ".join(
        data[data['label'] == 1]['message']
    )

    wc = WordCloud(
        width=800,
        height=400,
        background_color="white"
    ).generate(spam_text)

    fig3, ax3 = plt.subplots(figsize=(10, 5))

    ax3.imshow(wc)

    ax3.axis("off")

    st.pyplot(fig3)


# =========================
# TAB 3 → DATASET
# =========================
with tab3:

    st.subheader("📂 Dataset Preview")

    st.dataframe(data.head())

    st.write("📧 Total Emails:", len(data))

    st.write("Dataset Shape:", data.shape)

    st.write("Spam Count:")

    st.write(data['label'].value_counts())
    
    with st.expander("📊 See Dataset Info"):
        st.write(data.describe())

    search = st.text_input("🔍 Search Message")

    if search != "":

        filtered = data[
            data['message'].str.contains(
                search,
                case=False
            )
        ]

        st.dataframe(filtered.head(10))


# =========================
# TAB 1 → PREDICTION
# =========================
with tab1:

    # User Input
    user_message = st.text_area(
        "Enter your message"
    )
    st.write(f"📏 Characters: {len(user_message)}")
# FILE UPLOAD
uploaded_file = st.file_uploader(
    "📂 Upload Email File",
    type=["txt"]
)

# Session State
if "history" not in st.session_state:
        st.session_state.history = []

if st.button("Predict"):

    with st.spinner("Analyzing Message..."):
        if uploaded_file is not None:
            user_message = uploaded_file.read().decode("utf-8")

        if user_message == "":

            st.warning("⚠ Please enter a message")

        else:

            cleaned_message = clean_text(user_message)

            msg = vectorizer.transform(
                [cleaned_message]
            )

            prediction = model.predict(msg)

            probability = model.predict_proba(msg)

            confidence = max(
                probability[0]
            ) * 100

            st.progress(int(confidence))

            st.metric(
                "🎯 Confidence Score",
                f"{confidence:.2f}%"
            )

            if prediction[0] == 1:

                st.error("🚨 Spam Email")

            else:

                st.success("✅ Not Spam")

                st.balloons()


            # SAVE HISTORY
            st.session_state.history.append(
                (
                    user_message,
                    "Spam" if prediction[0] == 1
                    else "Not Spam"
                )
            )

            st.write(
                "Your Message:",
                user_message
            )

    # =========================
    # HISTORY
    # =========================
    st.subheader("📜 Prediction History")

    st.write(st.session_state.history)

    # =========================
    # DOWNLOAD REPORT
    # =========================
    df = pd.DataFrame(
        st.session_state.history,
        columns=["Message", "Prediction"]
    )

    st.download_button(
        "📥 Download Report",
        df.to_csv(index=False),
        "spam_report.csv",
        "text/csv"
    )

    # =========================
    # CLEAR BUTTON
    # =========================
    if st.button("🗑 Clear All"):

        st.session_state.history = []

        st.rerun()

st.markdown("---")

st.write("Made with ❤️ using Streamlit & Machine Learning") 