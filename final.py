import streamlit as st
import pandas as pd
from textblob import TextBlob
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from collections import defaultdict, Counter
import difflib
import os
import re
import cv2
import numpy as np
from firebase_admin import credentials, db
import firebase_admin
from pyneuphonic import Neuphonic, TTSConfig
from pyneuphonic.player import AudioPlayer

client = Neuphonic(api_key='6fd9ed5d3158ba486953e510dc9dc03cf8915684f8b82d0e29128ec2c2cf19e3.58ffe10e-5c02-488e-a227-27c5608ada2a')

sse = client.tts.SSEClient()

# TTSConfig is a pydantic model so check out the source code for all valid options
tts_config = TTSConfig(
    speed=1.05,
    lang_code='en', # replace the lang_code with the desired language code.
    voice_id='e564ba7e-aa8d-46a2-96a8-8dffedade48f'  # use client.voices.list() to view all available voices
)

# Load cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

# --------- SENTIMENT ANALYSIS ---------
def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    if sentiment_score > 0:
        return "Positive"
    elif sentiment_score < 0:
        return "Negative"
    else:
        return "Neutral"

# --------- DIAGNOSIS MODEL SETUP ---------
@st.cache_resource
def load_model_data():
    df, meta = arff.loadarff("cleaned_diagnosis_dataset.arff")
    df = pd.DataFrame(df)
    df = df.applymap(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)

    X = df.drop(columns="prognosis").astype(int)
    le = LabelEncoder()
    y = le.fit_transform(df["prognosis"])
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    accuracy = accuracy_score(y, model.predict(X))

    return model, le, X.columns, accuracy, df, X, y

def load_synonym_map(csv_path, selected_lang="en"):
    synonym_map = defaultdict(list)
    terms = set()
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        filtered = df[df["language"] == selected_lang]
        for _, row in filtered.iterrows():
            key = row["input_term"].strip().lower()
            val = row["mapped_symptom"].strip().lower()
            synonym_map[key].append(val)
            terms.add(key)
    return synonym_map, list(terms)

def suggest_term(input_term, term_list):
    return difflib.get_close_matches(input_term, term_list, n=3, cutoff=0.6)

def predict_prognosis(symptoms, model, label_encoder, feature_columns, top_n=5):
    input_data = {symptom: 0 for symptom in feature_columns}
    for symptom in symptoms:
        if symptom in input_data:
            input_data[symptom] = 1
    input_df = pd.DataFrame([input_data])
    probs = model.predict_proba(input_df)[0]
    top_indices = probs.argsort()[::-1][:top_n]
    return [(label_encoder.inverse_transform([i])[0], probs[i]) for i in top_indices]

def suggest_more_symptoms(df, X, top_prognosis, user_symptoms, label_col='prognosis'):
    top_df = df[df[label_col] == top_prognosis]
    if top_df.empty:
        return []
    symptom_counts = Counter()
    for _, row in top_df.iterrows():
        for col in X.columns:
            if col not in user_symptoms and row[col] == 1:
                symptom_counts[col] += 1
    return [symptom for symptom, _ in symptom_counts.most_common(5)]

# --------- STREAMLIT TABS ---------
def mental_health_tab():
    st.header("🧠 Mental Health Check-In")
    user_input = st.text_area("Write about how you're feeling today:")
    if st.button("Analyze Sentiment"):
        if user_input:
            sentiment = analyze_sentiment(user_input)
            st.subheader("Sentiment Analysis Result:")
            st.info(f"Detected Sentiment: **{sentiment}**")
            if sentiment == "Negative":
                st.markdown("You have negative emotions, play this relaxing game to be happy 🙂")
                st.markdown(
                    """
                    <div style="width: 100%; display: flex; justify-content: center;">
                        <div style="position: relative; width: 80%; padding-top: 30.25%; padding-bottom: 30.25%; overflow: hidden;">
                            <iframe src="https://archive.org/embed/msdos_DOOM_1993"
                                    style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; border: none;"
                                    allowfullscreen
                                    scrolling="no">
                            </iframe>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            elif sentiment == "Neutral":
                with open("tetris.html", "r") as file:
                    tetris_html = file.read()
                st.components.v1.html(tetris_html, width=350, height=700, scrolling=False)
            elif sentiment == "Positive":
                st.title("You are happy. Enjoy a game of minecraft.")
                st.markdown(
    """
    <div style="position: relative; width: 100%; max-width: 900px; height: 500px; margin: 20px auto; border: 5px solid white; border-radius: 10px; overflow: hidden;">
        <iframe src="https://minecraftforfreex.com/eaglercraft/" width="100%" height="100%" style="border: none;"></iframe>
    </div>
    """,
    unsafe_allow_html=True,
)
        else:
            st.warning("Please write something first.")

def symptom_checker_tab(model, label_encoder, features, accuracy, df, X):
    st.header("🩺 Symptom Checker")
    
    synonym_map, synonym_terms = load_synonym_map("symptom_synonyms_multilang_full.csv", "en")

    entered_symptom = st.text_input("Type a symptom (e.g., headache):").strip().lower()
    if entered_symptom:
        suggestions = suggest_term(entered_symptom, synonym_map.keys())
        if suggestions:
            selected = st.selectbox("Did you mean one of these?", suggestions)
        else:
            st.warning("No close matches found.")
            selected = None
    else:
        selected = None

    
    # Initialize or get the current symptoms
    if "selected_symptoms" not in st.session_state:
        st.session_state.selected_symptoms = set()
    selected_symptoms = st.session_state.selected_symptoms

    


    if selected:
        selected_lower = selected.lower()
        if synonym_map[selected_lower]:
            chosen = st.multiselect("Mapped symptoms to add:", synonym_map[selected_lower])
            selected_symptoms.update(chosen)

    st.session_state.selected_symptoms = selected_symptoms

    if selected_symptoms:
        st.markdown("### Selected Symptoms:")
        st.write(", ".join(selected_symptoms))

        if st.button("🔄 Reset Symptoms"):
            st.session_state.selected_symptoms = set()
            selected_symptoms = set()
            st.rerun()
        
        if st.button("Predict Condition"):
            top_preds = predict_prognosis(selected_symptoms, model, label_encoder, features)
            top_condition = top_preds[0][0]
            if "national_id" in st.session_state:
                update_user_prognosis(st.session_state.national_id, top_condition)
            st.success("Top Diagnoses:")
            for diag, prob in top_preds:
                st.write(f"- **{diag}** (Confidence: {prob*100:.2f}%)")

            diagnosis_text = f"Your top diagnosis is {top_condition}. Here are the top predictions: "
            for diag, prob in top_preds:
                diagnosis_text += f"{diag} with {prob*100:.0f} percent confidence. "

            # Speak the result
            with AudioPlayer() as player:
                response = sse.send(diagnosis_text, tts_config=tts_config)
                player.play(response)
            
            suggestions = suggest_more_symptoms(df, X, top_condition, selected_symptoms)
            if suggestions:
                st.markdown("💡 You may also want to check for these symptoms:")
                st.write(", ".join(suggestions))
            st.caption(f"Model accuracy: {accuracy*100:.2f}%")
    else:
        st.info("Start by entering a symptom above.")


def facial_emotion_tab():
    st.header("😊 Facial Emotion Recognition (Basic Geometry-Based)")
    picture = st.camera_input("Take a photo")

    if picture is not None:
        file_bytes = np.asarray(bytearray(picture.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) == 0:
            st.warning("No face detected.")
            return

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]

            # Estimate mouth region as bottom third of face
            mouth_region = face_roi[int(h * 0.65):h, :]
            _, mouth_thresh = cv2.threshold(mouth_region, 70, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(mouth_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            mar = 0
            if contours:
                c = max(contours, key=cv2.contourArea)
                _, _, mw, mh = cv2.boundingRect(c)
                if mw > 0:
                    mar = round(mh / mw, 2)

            # st.image(mouth_region, clamp=True, caption="Mouth ROI")  # Optional debug
            st.write(f"MAR: {mar:.2f}")

            if mar > 0.35:
                emotion = "Happy"
                color = (0, 255, 0)
            elif mar < 0.15:
                emotion = "Sad"
                color = (0, 0, 255)
            else:
                emotion = "Neutral"
                color = (255, 255, 0)

            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        st.image(img, channels="BGR", caption=f"Detected Emotion: {emotion}")
def mouth_aspect_ratio(region):
    if region is None or region.size == 0:
        return 0

    blurred = cv2.GaussianBlur(region, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        _, _, w, h = cv2.boundingRect(c)
        if w == 0:
            return 0
        return round(h / float(w), 2)
    return 0


# --- Firebase Setup ---
if "firebase_initialized" not in st.session_state:
    cred = credentials.Certificate("healthcheck-ba556-firebase-adminsdk-fbsvc-fd7e3ce955.json")
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://healthcheck-ba556-default-rtdb.europe-west1.firebasedatabase.app/'
    })
    st.session_state.firebase_initialized = True

def load_users():
    ref = db.reference("/users")
    users_data = ref.get()
    return users_data if users_data else {}

def save_user(national_id, name):
    ref = db.reference("/users")
    ref.child(national_id).set({"name": name})


def update_user_prognosis(national_id, prognosis):
    ref = db.reference(f"/users/{national_id}")
    ref.update({"current_prognosis": prognosis})

def handle_registration():
    st.subheader("Register")
    new_name = st.text_input("Enter your Name", key="reg_name")
    new_passport_id = st.text_input("Enter your Sudanese National Number", key="reg_id")
    sudan_ID_regex = r'^\d{11}$'
    sudan_name_regex = r"^[A-Za-z][A-Za-z'\-]{1,}(\s+[A-Za-z][A-Za-z'\-]{1,})+$"

    if st.button("Register"):
        if not re.fullmatch(sudan_ID_regex, new_passport_id):
            st.error("National ID must be 11 digits.")
        elif not re.fullmatch(sudan_name_regex, new_name):
            st.error("Write your full name correctly.")
        elif not new_name:
            st.error("Name cannot be empty.")
        else:
            users = st.session_state.users
            if new_passport_id in users:
                st.error("User already registered. Please log in.")
            else:
                users[new_passport_id] = new_name
                save_user(new_passport_id, new_name)
                st.success("Registration successful! You can now log in.")
                st.session_state.show_login = True
    if st.button("Already have an account? Login"):
        st.session_state.show_login = True

def handle_login():
    st.subheader("Login")
    name = st.text_input("Enter your Name", key="login_name")
    passport_id = st.text_input("Enter your Sudanese National Number", key="login_id")
    sudan_ID_regex = r'^\d{11}$'

    if st.button("Login"):
        if not re.fullmatch(sudan_ID_regex, passport_id):
            st.error("National ID must be 11 digits.")
        elif not name:
            st.error("Name cannot be empty.")
        elif passport_id in st.session_state.users and st.session_state.users[passport_id]["name"] == name:
            st.session_state.logged_in = True
            st.session_state.name = name
            st.session_state.national_id = passport_id
            st.success("Login successful!")
        else:
            st.error("Invalid credentials or user not registered.")

    if st.button("Don't have an account? Register"):
        st.session_state.show_login = False

def main():
    st.set_page_config("MedTech AI App", layout="centered")
    st.title("MedTech AI Application")

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "show_login" not in st.session_state:
        st.session_state.show_login = True
    st.session_state.users = load_users()

    if not st.session_state.logged_in:
        if st.session_state.show_login:
            handle_login()
        else:
            handle_registration()
    else:
        st.sidebar.success(f"Logged in as {st.session_state.name}")
        if st.sidebar.button("Log Out"):
            st.session_state.logged_in = False
            st.session_state.name = ""
            st.session_state.national_id = ""
            st.session_state.show_login = True
            st.success("You have been logged out. Redirecting to login page...")
            st.rerun()
        with st.sidebar:
            st.markdown("Sponsored Ads 💸")
            
            ads = [
                ("https://valyfy.com/", "https://media.licdn.com/dms/image/v2/D4E0BAQGQt7cO9SJY7Q/company-logo_200_200/company-logo_200_200/0/1702578949480/jigsawcareers_logo?e=2147483647&v=beta&t=vP-z6KmkG4VhggTKuOk-bdqWlQx0KrcBxsA2ETS6OUU"),
                ("https://neuphonic.com/", "https://media.licdn.com/dms/image/v2/D4E0BAQGgHA974hXLcQ/company-logo_200_200/company-logo_200_200/0/1712066348139/neuphonic_logo?e=2147483647&v=beta&t=EGFxvqwjnAZw0WgfMbPAeAZHc1GIH7OZUxPikEfbFDQ"),  
                ("https://majestic.com/", "https://media.licdn.com/dms/image/v2/C4D0BAQEkDUUuwUIGPw/company-logo_200_200/company-logo_200_200/0/1630507103142/majesticseo_logo?e=2147483647&v=beta&t=rfzNDTK5h9SKHvnZZinL5UmbpT6plQWXNiHfrxmxUfY"),  
                ("https://www.kainos.com/", "https://media.licdn.com/dms/image/v2/D4E0BAQHFsetR0VYa4A/company-logo_200_200/company-logo_200_200/0/1688137763667/kainos_logo?e=2147483647&v=beta&t=e3UCqRpIQuZlzaIFGsLJzUqv-BWp25rgPSBgqp0-slg"), 
                ("https://www.thetradedesk.com/", "https://www.opentext.com/assets/images/resources/customer-success/the-trade-desk-logo.jpg")
            ]

            # Loop through the ads and display them in two columns per row
            for i in range(0, len(ads), 2):
                cols = st.columns(2)  # Create two equal-sized columns
                for j in range(2):
                    if i + j < len(ads):
                        ad_url, ad_image = ads[i + j]
                        with cols[j]:
                            st.markdown(
                                f'<a href="{ad_url}" target="_blank">'
                                f'<img src="{ad_image}" style="width:100%; max-width:80px; border-radius:10px;"></a>',
                                unsafe_allow_html=True
                            )
            
            st.markdown("Partners 🤝")

            partners = [
                ("https://linktr.ee/algosoc", "https://media.licdn.com/dms/image/v2/D4E0BAQE86vcSbw1MrQ/company-logo_200_200/company-logo_200_200/0/1726133446659?e=2147483647&v=beta&t=0eiz6pXNsLLiO7pfvMLlk6Jrz0IAQbEdHFSEN6HTDak"),
                ("https://www.afnom.net/", "https://www.afnom.net/files/afnom_logo.png"),  
                ("https://www.birmingham.ac.uk/university/colleges/eps/eps-community/students/societies/games-development", "https://scontent-lhr6-2.xx.fbcdn.net/v/t39.30808-6/467707993_10231704946179830_4460555190326589190_n.jpg?stp=dst-jpg_s960x960_tt6&_nc_cat=105&ccb=1-7&_nc_sid=2285d6&_nc_ohc=hzf4o8MJd7YQ7kNvgENSko2&_nc_oc=Adlc6sN_8HTawug4kxIe6tNFx738pCJHMGmbuCrJnP-qfa8-HmsOXbL1RUFV15VmKiQ&_nc_zt=23&_nc_ht=scontent-lhr6-2.xx&_nc_gid=DkZstws8R5z12XI6_peEWg&oh=00_AYGwYeNATrIYIaDdndLdV0TuzaNhMQBfXVOAAmESXS97Nw&oe=67E53515")  
                #("https://www.thetradedesk.com/", "https://www.opentext.com/assets/images/resources/customer-success/the-trade-desk-logo.jpg"),
                #("https://www.kainos.com/", "https://media.licdn.com/dms/image/v2/D4E0BAQHFsetR0VYa4A/company-logo_200_200/company-logo_200_200/0/1688137763667/kainos_logo?e=2147483647&v=beta&t=e3UCqRpIQuZlzaIFGsLJzUqv-BWp25rgPSBgqp0-slg")  
            ]

            # Loop through the ads and display them in two columns per row
            for i in range(0, len(partners), 2):
                cols = st.columns(2)  # Create two equal-sized columns
                for j in range(2):
                    if i + j < len(partners):
                        ad_url, ad_image = partners[i + j]
                        with cols[j]:
                            st.markdown(
                                f'<a href="{ad_url}" target="_blank">'
                                f'<img src="{ad_image}" style="width:100%; max-width:80px; border-radius:10px;"></a>',
                                unsafe_allow_html=True
                            )


        model, le, features, acc, df, X, y = load_model_data()
        tab1, tab2, tab3 = st.tabs(["🧠 Mental Health", "🩺 Symptom Checker", "😊 Facial Emotion"])
        with tab1:
            mental_health_tab()
        with tab2:
            symptom_checker_tab(model, le, features, acc, df, X)
        with tab3:
            facial_emotion_tab()

if __name__ == "__main__":
    main()
