"""
SMS Spam Detector — Streamlit Web UI
Compares AdaBoost, XGBoost, LightGBM, CatBoost predictions side-by-side.

Run locally:   streamlit run app.py
Run in Colab:  see the notebook's launcher cell
"""
import os
import re
import pickle
import streamlit as st
import contractions
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)
nltk.download('omw-1.4',   quiet=True)

# ─── Slang dictionary (must match training) ──────────────────────────────────
SLANG_DICT = {
    "u": "you", "ur": "your", "r": "are", "b": "be", "y": "why", "k": "ok",
    "m": "am", "n": "and", "2": "to", "4": "for", "4u": "for you", "b4": "before",
    "2day": "today", "2nite": "tonight", "2moro": "tomorrow", "2morrow": "tomorrow",
    "2mrw": "tomorrow", "4got": "forgot", "4ever": "forever", "2gether": "together",
    "gm": "good morning", "gn": "good night", "gud": "good", "gr8": "great",
    "g8": "great", "gr8t": "great", "lol": "laugh out loud", "lmao": "laugh my ass off",
    "rofl": "rolling on the floor laughing", "omg": "oh my god", "omfg": "oh my god",
    "wtf": "what the hell", "wth": "what the hell", "idk": "i do not know",
    "imo": "in my opinion", "imho": "in my humble opinion", "tbh": "to be honest",
    "ngl": "not gonna lie", "fyi": "for your information", "btw": "by the way",
    "jk": "just kidding", "brb": "be right back", "bbl": "be back later",
    "bbs": "be back soon", "afk": "away from keyboard", "np": "no problem",
    "yw": "you are welcome", "ty": "thank you", "thx": "thanks", "thnx": "thanks",
    "tnx": "thanks", "tx": "thanks", "pls": "please", "plz": "please", "plx": "please",
    "sry": "sorry", "srry": "sorry", "nvm": "never mind", "nvr": "never", "evr": "ever",
    "wut": "what", "wat": "what", "wats": "what is", "abt": "about", "hw": "how",
    "hru": "how are you", "wen": "when", "whr": "where", "wer": "where",
    "der": "there", "dere": "there", "dis": "this", "dat": "that", "dats": "that is",
    "d": "the", "da": "the", "de": "the", "dem": "them", "dey": "they",
    "asap": "as soon as possible", "atm": "at the moment", "nite": "night",
    "nyt": "night", "tmrw": "tomorrow", "tmr": "tomorrow", "tmo": "tomorrow",
    "l8r": "later", "l8": "late", "ltr": "later", "ttyl": "talk to you later",
    "ttys": "talk to you soon", "tc": "take care", "cya": "see you", "cu": "see you",
    "cul8r": "see you later", "b4n": "bye for now", "wanna": "want to",
    "gonna": "going to", "gotta": "got to", "kinda": "kind of", "sorta": "sort of",
    "dunno": "do not know", "lemme": "let me", "gimme": "give me", "cmon": "come on",
    "w8": "wait", "h8": "hate", "m8": "mate", "d8": "date", "f8": "fate",
    "sk8": "skate", "ne1": "anyone", "ne": "any", "neway": "anyway",
    "nething": "anything", "evry1": "everyone", "every1": "everyone",
    "som1": "someone", "sum1": "someone", "no1": "no one", "txt": "text",
    "msg": "message", "msgs": "messages", "mob": "mobile", "mobi": "mobile",
    "rply": "reply", "rpl": "reply", "stp": "stop", "dont": "do not",
    "doesnt": "does not", "cant": "cannot", "wont": "will not", "couldnt": "could not",
    "wouldnt": "would not", "shouldnt": "should not", "im": "i am", "ive": "i have",
    "id": "i would", "ill": "i will", "isnt": "is not", "arent": "are not",
    "wasnt": "was not", "werent": "were not", "havent": "have not", "hasnt": "has not",
    "hadnt": "had not", "didnt": "did not", "theyre": "they are", "theyd": "they would",
    "theyll": "they will", "theyve": "they have", "youre": "you are", "youd": "you would",
    "youll": "you will", "youve": "you have", "hes": "he is", "shes": "she is",
    "its": "it is", "weve": "we have", "wed": "we would", "wk": "week",
    "wkly": "weekly", "mth": "month", "yr": "year", "min": "minute", "mins": "minutes",
    "hr": "hour", "hrs": "hours", "wks": "weeks", "acc": "account", "acct": "account",
    "amt": "amount", "cd": "could", "gt": "got", "hav": "have", "hve": "have",
    "nd": "and", "sm": "some", "sme": "some", "ppl": "people", "nt": "not",
    "nw": "now", "nxt": "next", "smth": "something", "sumthing": "something",
    "grt": "great", "amzing": "amazing", "cust": "customer", "custmr": "customer",
    "dlr": "dollar", "dlrs": "dollars", "reciev": "receive", "recive": "receive",
    "receve": "receive", "reedem": "redeem", "redeme": "redeem",
    "subscrib": "subscribe", "unsubscrib": "unsubscribe", "exclusiv": "exclusive",
    "xclusiv": "exclusive", "mrkt": "market", "mkt": "market", "priz": "prize",
}

_slang_keys_sorted = sorted(SLANG_DICT.keys(), key=len, reverse=True)
_slang_pattern = re.compile(
    r'\b(' + '|'.join(re.escape(k) for k in _slang_keys_sorted) + r')\b'
)

STOP_WORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text: str, normalize_slang: bool = True) -> str:
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' urltoken ', text)
    text = re.sub(r'\b(?:\+?\d[\d\s\-().]{7,})\b', ' phonetoken ', text)
    text = contractions.fix(text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    if normalize_slang:
        text = _slang_pattern.sub(lambda m: SLANG_DICT[m.group(0)], text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and not t.isdigit()]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)


# ─── Model loading (cached) ───────────────────────────────────────────────────
@st.cache_resource
def load_artifact(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


MODELS_DIR = 'models'
MODEL_NAMES = ['AdaBoost', 'XGBoost', 'LightGBM', 'CatBoost']
MODEL_COLORS = {'AdaBoost': '#1f77b4', 'XGBoost': '#d62728',
                'LightGBM': '#2ca02c', 'CatBoost': '#9467bd'}


# ─── Page setup ───────────────────────────────────────────────────────────────
st.set_page_config(page_title='SMS Spam Detector', page_icon='📩', layout='wide')
st.title('📩 SMS Spam Detector')
st.caption('Boosting model comparison • AdaBoost / XGBoost / LightGBM / CatBoost')
st.markdown('---')


# ─── Sidebar configuration ────────────────────────────────────────────────────
with st.sidebar:
    st.header('⚙️ Configuration')
    normalize = st.checkbox('Use slang normalization', value=True,
                            help='Applies the SMS slang lexicon (e.g., "u" → "you").')
    ngram = st.selectbox('N-gram range', ['Unigram', 'Bigram', 'Trigram'],
                         help='Unigram = single words. Bigram/Trigram include 2- and 3-word phrases.')
    st.markdown('---')
    st.markdown('### About')
    st.markdown(
        'Evaluating the impact of **slang normalization** and **N-gram features** '
        'on boosting models for SMS spam detection.\n\n'
        '**Dataset:** UCI SMS Spam Collection (5,572 messages)\n\n'
        '**Authors:**\n- Alexander C. S. Linggodigdo\n- Fanny O. Pangestu\n- Howard F. Goh'
    )


# ─── Build artifact keys ──────────────────────────────────────────────────────
norm_slug = 'With_Normalization' if normalize else 'No_Normalization'
config_slug = f'{norm_slug}_{ngram}'
vec_path = os.path.join(MODELS_DIR, f'vec_{config_slug}.pkl')

if not os.path.exists(vec_path):
    st.error(
        f'❌ Trained models not found at `{MODELS_DIR}/`. '
        'Run the training notebook (`spam_detection_colab.ipynb`) first '
        'and make sure the `models/` folder is in the same directory as `app.py`.'
    )
    st.stop()


# ─── Input area ───────────────────────────────────────────────────────────────
st.subheader('✉️ Enter an SMS message')

EXAMPLES = {
    '— Try a sample —': '',
    'Sample (spam): Free prize':
        'Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Txt FA to 87121 to receive entry',
    'Sample (spam): Claim winner':
        "Congrats! Ur the winner of a gr8 1000 dollar prize! Claim ur reward now at http://win-now.example",
    'Sample (ham): Casual':
        'Hey, are you coming to the party tonight? Let me know if you need a ride.',
    'Sample (ham): Slang':
        'lol u r so funny, ttyl gotta finish hw first',
}

example_choice = st.selectbox('Quick samples:', list(EXAMPLES.keys()))
default_text = EXAMPLES[example_choice]

text = st.text_area('Message:', value=default_text, height=120,
                    placeholder='Paste an SMS message here…')

predict_clicked = st.button('🔍 Predict', type='primary')


# ─── Run prediction ───────────────────────────────────────────────────────────
if predict_clicked:
    if not text.strip():
        st.warning('Please enter a message first.')
        st.stop()

    processed = preprocess(text, normalize_slang=normalize)

    # Preprocessing trace
    st.markdown('### 🔧 Preprocessing Trace')
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('**Original input**')
        st.code(text, language=None)
    with c2:
        st.markdown(f'**After preprocessing** ({norm_slug.replace("_", " ")}, {ngram})')
        st.code(processed if processed.strip() else '(empty — message contained only stopwords/punctuation)',
                language=None)

    # Vectorize
    vec = load_artifact(vec_path)
    X = vec.transform([processed])

    if X.nnz == 0:
        st.warning(
            '⚠️ The preprocessed text contains no vocabulary the models have seen. '
            'Predictions may be unreliable.'
        )

    # Show per-model predictions
    st.markdown('### 🤖 Model Predictions')
    cols = st.columns(4)

    spam_count = 0
    probs = {}
    for col, model_name in zip(cols, MODEL_NAMES):
        model_path = os.path.join(MODELS_DIR, f'model_{config_slug}_{model_name}.pkl')
        model = load_artifact(model_path)
        prob_spam = float(model.predict_proba(X)[0, 1])
        pred = int(prob_spam >= 0.5)
        probs[model_name] = prob_spam
        if pred == 1:
            spam_count += 1

        with col:
            color = MODEL_COLORS[model_name]
            st.markdown(
                f"<div style='border-left: 4px solid {color}; padding-left: 12px;'>"
                f"<h4 style='margin-bottom: 4px;'>{model_name}</h4></div>",
                unsafe_allow_html=True
            )
            if pred == 1:
                st.markdown(
                    "<h2 style='color: #d62728; margin-top: 8px;'>🔴 SPAM</h2>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    "<h2 style='color: #2ca02c; margin-top: 8px;'>🟢 HAM</h2>",
                    unsafe_allow_html=True
                )
            st.progress(prob_spam)
            st.metric(label='Spam probability', value=f'{prob_spam*100:.2f}%')
            st.caption(f'Ham probability: {(1-prob_spam)*100:.2f}%')

    # Consensus
    st.markdown('### 📊 Model Consensus')
    if spam_count == 4:
        st.error(f'⚠️ **Strong spam signal** — all 4 models flagged this message as SPAM.')
    elif spam_count == 0:
        st.success(f'✅ **Clean message** — all 4 models classified this as HAM.')
    else:
        st.warning(
            f'⚖️ **Mixed verdict** — {spam_count} of 4 models flagged this as SPAM. '
            'Models disagree; the message is borderline.'
        )

    # Probability summary table
    import pandas as pd
    summary = pd.DataFrame({
        'Model': MODEL_NAMES,
        'Prediction': ['SPAM' if probs[m] >= 0.5 else 'HAM' for m in MODEL_NAMES],
        'Spam %': [f'{probs[m]*100:.2f}%' for m in MODEL_NAMES],
        'Ham %':  [f'{(1-probs[m])*100:.2f}%' for m in MODEL_NAMES],
    })
    st.dataframe(summary, hide_index=True, use_container_width=True)
