import streamlit as st
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from gensim.models.phrases import Phrases, Phraser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(page_title="Mini Search Engine", layout="centered")

# --- Konstanta ---
SEARCH_CSV_PATH = "data/raw-text.csv"
CHATBOT_CSV_PATH = "data/chatbot-data.csv"
COSINE_SIMILARITY_THRESHOLD = 0.4
SNIPPET_MAX_LENGTH = 250

# --- Inisialisasi NLTK dan Sastrawi (dilakukan sekali) ---
@st.cache_resource
def initialize_nlp_resources():
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    stemmer_factory = StemmerFactory()
    stemmer_instance = stemmer_factory.create_stemmer()
    stop_words_set = set(stopwords.words('indonesian'))
    return stemmer_instance, stop_words_set

STEMMER, STOP_WORDS = initialize_nlp_resources()

# --- Preprocessing Utilities ---
def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'\d+', '', text)
    return ' '.join(text.lower().split())

def tokenize_and_filter(text: str, stop_words_set: set) -> list[str]:
    tokens = word_tokenize(text)
    return [token for token in tokens if token not in stop_words_set]

def identify_and_process_phrases(tokens_list: list[list[str]]) -> list[list[str]]:
    phrases = Phrases(tokens_list, min_count=1, threshold=1)
    phraser = Phraser(phrases)
    unwanted_phrases = {"raya_lagu"}

    processed_phrase_tokens = []
    for tokens in tokens_list:
        result = []
        current_unwanted = set(unwanted_phrases)
        for phrase in phraser[tokens]:
            if phrase in current_unwanted:
                result.extend(phrase.split("_"))
            else:
                result.append(phrase)
        processed_phrase_tokens.append(result)
    return processed_phrase_tokens

def stem_tokens(tokens_list: list[list[str]], stemmer_instance) -> list[list[str]]:
    return [[stemmer_instance.stem(token) for token in tokens] for tokens in tokens_list]

@st.cache_data(show_spinner=False)
def preprocess_documents(texts: list[str]) -> list[list[str]]:
    cleaned_texts = [clean_text(text) for text in texts]
    tokenized_texts = [tokenize_and_filter(text, STOP_WORDS) for text in cleaned_texts]
    phrased_tokens = identify_and_process_phrases(tokenized_texts)
    stemmed_tokens = stem_tokens(phrased_tokens, STEMMER)
    return stemmed_tokens

@st.cache_data(show_spinner=False)
def compute_tfidf_matrix(docs: list[list[str]]):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([' '.join(doc) for doc in docs])
    return tfidf_matrix, vectorizer

# Menggunakan fungsi reweighting
def get_similarity_scores_with_reweighting(vectorizer: TfidfVectorizer, tfidf_matrix, query_processed: list[str]) -> list[float]:
    query_text = ' '.join(query_processed)
    query_vector = vectorizer.transform([query_text])
    feature_names = vectorizer.get_feature_names_out()
    
    modified_query_vector = query_vector.copy()
    reweight_factor = 2.0 

    if 'siapa' in feature_names and 'siapa' in query_processed:
        siapa_idx = np.where(feature_names == 'siapa')[0][0]
        if siapa_idx < modified_query_vector.shape[1]:
            modified_query_vector[0, siapa_idx] *= reweight_factor

    if 'cipta' in feature_names and 'cipta' in query_processed: 
        cipta_idx = np.where(feature_names == 'cipta')[0][0]
        if cipta_idx < modified_query_vector.shape[1]:
            modified_query_vector[0, cipta_idx] *= reweight_factor
            
    return cosine_similarity(modified_query_vector, tfidf_matrix)[0]

# --- Pemuatan dan Pemrosesan Data ---
@st.cache_resource(show_spinner="⏳ Memuat dan memproses data pencarian...")
def load_search_data(path: str):
    try:
        df = pd.read_csv(path)
        if 'Content' not in df.columns:
            st.error("❌ Kolom 'Content' tidak ditemukan di file CSV pencarian.")
            return None, None, None
        contents = df['Content'].astype(str).tolist()
        processed_docs = preprocess_documents(contents)
        tfidf_matrix, vectorizer = compute_tfidf_matrix(processed_docs)
        return df, tfidf_matrix, vectorizer
    except FileNotFoundError:
        st.error(f"❌ File pencarian tidak ditemukan di path: `{path}`")
        return None, None, None
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat data pencarian: {e}")
        return None, None, None

@st.cache_resource(show_spinner="⏳ Memuat dan memproses data...")
def load_chatbot_data(path: str):
    try:
        df = pd.read_csv(path)
        if 'question' not in df.columns or 'answer' not in df.columns:
            st.error("❌ Kolom 'question' atau 'answer' tidak ditemukan di file CSV chatbot.")
            return None, None, None, None
        
        # Pastikan kolom 'image_url' ada. Jika tidak ada, tambahkan dengan nilai kosong.
        if 'image_url' not in df.columns:
            df['image_url'] = ''
        
        # Isi list dari kolom yang dibutuhkan
        questions = df['question'].astype(str).tolist()
        answers = df['answer'].astype(str).tolist()
        
        # Mengubah nilai NaN menjadi string kosong untuk image_url
        image_urls = df['image_url'].fillna('').astype(str).tolist() 

        processed_questions = preprocess_documents(questions)
        tfidf_matrix, vectorizer = compute_tfidf_matrix(processed_questions)
        
        # Mengembalikan juga list dari image_urls
        return df, answers, image_urls, tfidf_matrix, vectorizer
    except FileNotFoundError:
        st.error(f"❌ File chatbot tidak ditemukan di path: `{path}`")
        return None, None, None, None, None
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat data chatbot: {e}")
        return None, None, None, None, None

# --- CSS untuk Tampilan UI ---
st.markdown("""
    <style>
        h1, h2, h3, h4, h5, h6, p {
            font-family: "Segoe UI", sans-serif !important;
        }
        .title {
            text-align: center;
            font-size: 32px;
            margin: 20px 0 30px;
        }
        .result {
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            transition: background-color 0.3s ease;
            border: 1px solid transparent;
        }
        .result:hover {
            background-color: rgba(100, 149, 237, 0.1);
        }
        .result-title {
            color: #6992cd;
            font-weight: bold;
            font-size: 1.1rem;
            margin-bottom: 0.3rem;
        }
        .result-snippet {
            font-size: 0.95rem;
        }
        .result-meta {
            font-size: 0.8rem;
            color: gray;
            margin-top: 0.5rem;
        }
        #hasil-pencarian {
            padding-bottom: 30px;
        }
        .qa-overview {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 1.2rem;
            box-shadow: 0 0 0 2px rgba(57,229,140,1), 8px 8px 0 0 rgba(57,229,140,1);
            border-radius: 10px;
            margin-bottom: 1.5rem;
        }
        .qa-overview h3 {
            margin-top: 0;
            font-size: 1.25rem;
        }
        .qa-overview p {
            font-size: 1rem;
            line-height: 1.6;
            margin-bottom: 0.5rem;
        }
        .qa-overview img {
            max-width: 250px;
            height: auto;
            display: block;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🔎 Mini Search Engine</div>', unsafe_allow_html=True)

# --- Pemuatan Data Global ---
SEARCH_DF, SEARCH_TFIDF_MATRIX, SEARCH_VECTORIZER = load_search_data(SEARCH_CSV_PATH)
# Perbarui pemanggilan load_chatbot_data dan penangkapan variabel
CHATBOT_DF, CHATBOT_ANSWERS, CHATBOT_IMAGE_URLS, CHATBOT_TFIDF_MATRIX, CHATBOT_VECTORIZER = load_chatbot_data(CHATBOT_CSV_PATH)


# --- Input Pencarian Utama ---
with st.form("main_search_form", clear_on_submit=False):
    col1, col2 = st.columns([6, 1])
    with col1:
        query = st.text_input(
            "",
            placeholder="Ketik pertanyaan atau kata kunci pencarian...",
            label_visibility="collapsed",
            key="unified_search_input"
        )
    with col2:
        submitted = st.form_submit_button("Search")

# --- Logika Tampilan Hasil ---
if submitted and query:
    st.markdown("## Hasil Pencarian:")

    if CHATBOT_DF is not None and CHATBOT_TFIDF_MATRIX is not None and CHATBOT_VECTORIZER is not None and CHATBOT_IMAGE_URLS is not None:
        with st.spinner("🤖 Mencari jawaban Q&A..."):
            processed_query_for_chatbot = preprocess_documents([query])[0]
            sim_scores_chatbot = get_similarity_scores_with_reweighting(CHATBOT_VECTORIZER, CHATBOT_TFIDF_MATRIX, processed_query_for_chatbot)
            top_idx_chatbot = sim_scores_chatbot.argmax()
            top_score_chatbot = sim_scores_chatbot[top_idx_chatbot]

            ai_answer = None
            ai_image_url_to_display = None # Variabel baru untuk URL gambar

            if "siapa yang membuat anda" in query.lower() or "siapa pembuatmu" in query.lower():
                ai_answer = "Saya adalah model bahasa yang dibuat oleh 3 mahasiswa paling Sigma di Primakara University yaitu Galang, Pantera, dan Radika."
            elif top_score_chatbot > COSINE_SIMILARITY_THRESHOLD:
                ai_answer = CHATBOT_ANSWERS[top_idx_chatbot]
                # Ambil URL gambar yang sesuai dari daftar
                ai_image_url_to_display = CHATBOT_IMAGE_URLS[top_idx_chatbot] 

            if ai_answer:
                # Siapkan HTML untuk gambar jika URL-nya ada
                image_html_content = ""
                # Cek apakah URL gambar tidak kosong dan bukan string 'None' (dari fillna)
                if ai_image_url_to_display and ai_image_url_to_display != 'None':
                    # Gunakan tag <img> dengan URL langsung
                    image_html_content = f'<img src="{ai_image_url_to_display}" alt="Gambar relevan">'

                st.markdown(f"""
                    <div class="qa-overview">
                        <div>
                            <h3>✨ Overview:</h3>
                            <p>{ai_answer}</p>
                        </div>
                        {image_html_content}
                    </div>
                """, unsafe_allow_html=True)

    # --- Bagian Hasil Pencarian Web dari raw-text ---
    if SEARCH_DF is not None and SEARCH_TFIDF_MATRIX is not None and SEARCH_VECTORIZER is not None:
        with st.spinner("🔍 Mencari dokumen relevan..."):
            processed_query_for_search = preprocess_documents([query])[0]
            similarities_search = get_similarity_scores_with_reweighting(SEARCH_VECTORIZER, SEARCH_TFIDF_MATRIX, processed_query_for_search)

            search_results_df = SEARCH_DF.copy()
            search_results_df['Cosine Similarity'] = similarities_search
            results_filtered = search_results_df[search_results_df['Cosine Similarity'] > 0].sort_values(by='Cosine Similarity', ascending=False)

            st.markdown(f"<p>Ditemukan <b>{len(results_filtered)}</b> hasil dokumen untuk: <b>'{query}'</b></p>", unsafe_allow_html=True)

            if results_filtered.empty:
                st.warning("❗ Tidak ditemukan dokumen yang relevan.")
            else:
                for _, row in results_filtered.iterrows():
                    snippet = row['Content']
                    if len(snippet) > SNIPPET_MAX_LENGTH:
                        snippet = snippet[:SNIPPET_MAX_LENGTH] + "..."

                    st.markdown(f"""
                        <div class="result">
                            <div class="result-title">{row['Content'][:70]}...</div>
                            <div class="result-snippet">{snippet}</div>
                            <div class="result-meta">Cosine Similarity: {row['Cosine Similarity']:.4f}</div>
                        </div>
                    """, unsafe_allow_html=True)
    else:
        st.info("Data pencarian (raw-text) tidak tersedia untuk hasil dokumen.")