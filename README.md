# SMS Spam Detector — Boosting Model Comparison

Interactive web demo for the paper *"Evaluasi Dampak Slang Normalization dan
N-gram Feature terhadap Performa Model Boosting (XGBoost, LightGBM, CatBoost,
AdaBoost) Dalam Deteksi Spam"*.

Paste an SMS message -> see predictions from all 4 boosting models side-by-side,
with confidence scores, preprocessing trace, and consensus verdict.

## Authors
- Alexander Christian Suryanto Linggodigdo
- Fanny Octaviana Pangestu
- Howard Frelindo Goh

Computer Science — BINUS University

## Tech
- Preprocessing: case folding, entity replacement, contraction expansion,
  slang normalization, lemmatization
- Features: TF-IDF (Unigram / Bigram / Trigram)
- Models: AdaBoost, XGBoost, LightGBM, CatBoost
- UI: Streamlit
