# movie_app.py
# CineAI — Cinematic UI + Content-based recommender + TMDB posters & trailer modal
import streamlit as st
import pandas as pd
import requests
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import html

# ---------------- CONFIG ----------------
st.set_page_config(page_title="CineAI — Cinematic Recs", page_icon="🎬", layout="wide")

# ---------------- USER SETTINGS ----------------
TMDB_API_KEY = "048decb0f4375496911bac36e04f376e"  # <-- your TMDB API key

# ---------------- STYLES ----------------
st.markdown(
    """
    <style>
    :root{
      --bg:#050505; --surface:#0f1113; --accent:#e50914; --muted:#9aa0a6;
    }
    body {
      background: linear-gradient(180deg,var(--bg) 0%, #081018 100%);
      color: #e6e6e6;
      font-family: 'Poppins', sans-serif;
    }
    .topbar{
      display:flex;
      align-items:center;
      gap:16px;
      padding:16px 28px;
    }
    .brand{
      font-weight:800;
      color:var(--accent);
      font-size:20px;
    }

    /* 🎯 Equal spacing top-bottom + left-right */
    .grid-wrap{
      padding:28px 28px;                /* more space top-bottom */
    }
    .card-grid{
      display:grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap:24px 24px;                    /* equal vertical + horizontal spacing */
    }
    .card{
      background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(0,0,0,0.35));
      border-radius:12px;
      overflow:hidden;
      box-shadow: 0 8px 30px rgba(0,0,0,0.6);
      transition: transform .22s ease, box-shadow .22s ease;
      position:relative;
      height:360px;
      display:flex;
      flex-direction:column;
      justify-content:flex-start;
      margin-bottom:26px;              /* 🔥 vertical gap between rows */
    }
    .card:hover{
      transform: translateY(-8px) scale(1.02);
      box-shadow: 0 28px 60px rgba(0,0,0,0.7),
                  0 0 30px rgba(229,9,20,0.06);
    }
    .poster{
      width:100%;
      height:220px;
      object-fit:cover;
      display:block;
      transition: transform .35s;
    }
    .card:hover .poster{ transform: scale(1.06); }
    .card-body{
      padding:10px 12px;
      display:flex;
      flex-direction:column;
      gap:8px;
      flex:1;
    }
    .title{font-weight:700; color:#fff; font-size:15px; margin:0;}
    .meta{color:var(--muted); font-size:13px;}

    /* 🎬 Play icon overlay only (no text) */
    .play-overlay{
      position:absolute;
      left:50%; top:50%;
      transform:translate(-50%,-50%);
      z-index:5;
      background: rgba(0,0,0,0.55);
      padding:16px 18px;
      border-radius:50%;
      display:flex;
      align-items:center;
      justify-content:center;
      cursor:pointer;
      opacity:0;
      transition: opacity .18s, transform .18s;
    }
    .card:hover .play-overlay{ opacity:1; transform:translate(-50%,-50%) scale(1.08); }
    .play-triangle{
      width:0;
      height:0;
      border-left:14px solid #fff;
      border-top:9px solid transparent;
      border-bottom:9px solid transparent;
    }

    /* 🎞 Trailer modal */
    .modal {
      position: fixed;
      top:0; left:0;
      width:100%; height:100%;
      display:flex;
      align-items:center; justify-content:center;
      z-index:99999;
      background: rgba(0,0,0,0.75);
    }
    .modal-card{
      width:90%;
      max-width:1100px;
      background:#040405;
      border-radius:12px;
      padding:12px;
      box-shadow: 0 30px 80px rgba(0,0,0,0.8);
    }
    .modal-header{
      display:flex;
      justify-content:space-between;
      align-items:center;
      padding:6px 8px;
    }
    .close-btn{
      background:#111;
      color:#fff;
      border-radius:8px;
      padding:8px 12px;
      border:0;
      font-weight:700;
      cursor:pointer;
    }
    .iframe-wrap{
      width:100%;
      height:66vh;
      border-radius:8px;
      overflow:hidden;
    }

    /* 🧩 Fix: Add equal row space for Streamlit layout */
    [data-testid="stHorizontalBlock"] > div {
      margin-bottom: 26px;  /* uniform vertical space between card rows */
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# ---------------- DATA LOAD ----------------
@st.cache_data
def load_movies():
    movies = pd.read_csv("Data/tmdb_5000_movies.csv")
    credits = pd.read_csv("Data/tmdb_5000_credits.csv")

    def safe_eval(x):
        try:
            return literal_eval(x)
        except:
            return []

    for c in ["genres", "keywords"]:
        movies[c] = movies[c].fillna("[]").apply(safe_eval)
    for c in ["cast", "crew"]:
        credits[c] = credits[c].fillna("[]").apply(safe_eval)

    cr = credits[["movie_id", "cast", "crew"]].copy().rename(columns={"movie_id": "id"})
    movies = movies.merge(cr, on="id", how="left")

    def top_cast(cast_list, n=3):
        return [x.get("name") for x in (cast_list or [])[:n] if x.get("name")]

    def director(crew_list):
        for p in (crew_list or []):
            if p.get("job") == "Director":
                return p.get("name")
        return ""

    movies["cast_names"] = movies["cast"].apply(lambda x: top_cast(x, 3))
    movies["director"] = movies["crew"].apply(lambda x: director(x))
    movies["overview"] = movies["overview"].fillna("")
    movies["genre_names"] = movies["genres"].apply(lambda g: [x.get("name") for x in g])

    def clean_list(x):
        if isinstance(x, list):
            return [str(i).lower().replace(" ", "") for i in x]
        return []

    movies["soup"] = movies.apply(
        lambda r: " ".join(
            [r["overview"].lower()]
            + clean_list(r["genre_names"])
            + clean_list(r["cast_names"])
            + [str(r["director"]).lower().replace(" ", "")]
        ),
        axis=1,
    )
    return movies


movies = load_movies()


@st.cache_data
def build_similarity():
    tf = TfidfVectorizer(stop_words="english", max_features=10000)
    mat = tf.fit_transform(movies["soup"])
    sim = linear_kernel(mat, mat)
    idx = pd.Series(movies.index, index=movies["title"].str.lower()).drop_duplicates()
    return sim, idx


cosine_sim, indices = build_similarity()

# ---------------- TMDB HELPERS ----------------
def tmdb_search_first(title):
    try:
        url = f"https://api.themoviedb.org/3/search/movie?api_key=048decb0f4375496911bac36e04f376e&query={requests.utils.quote(title)}"
        r = requests.get(url, timeout=8).json()
        if r.get("results"):
            return r["results"][0]
    except Exception:
        return None
    return None


@st.cache_data
def fetch_poster_and_trailer(title):
    try:
        r = tmdb_search_first(title)
        if not r:
            return "https://via.placeholder.com/500x750?text=No+Image", None
        poster_path = r.get("poster_path") or r.get("backdrop_path")
        poster = (
            f"https://image.tmdb.org/t/p/w500{poster_path}"
            if poster_path
            else "https://via.placeholder.com/500x750?text=No+Image"
        )
        movie_id = r.get("id")
        if not movie_id:
            return poster, None
        vids = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key=048decb0f4375496911bac36e04f376e",
            timeout=8,
        ).json()
        yt = None
        for v in vids.get("results", []):
            if v.get("site") == "YouTube" and v.get("type") in ("Trailer", "Teaser"):
                yt = f"https://www.youtube.com/embed/{v.get('key')}"
                break
        return poster, yt
    except Exception:
        return "https://via.placeholder.com/500x750?text=Error", None


# ---------------- RECOMMENDATION ----------------
def recommend(title, top_n=18):
    t = title.lower()
    if t not in indices:
        return pd.DataFrame()
    idx = indices[t]
    sims = list(enumerate(cosine_sim[idx]))
    sims = sorted(sims, key=lambda x: x[1], reverse=True)[1 : top_n + 1]
    ids = [s[0] for s in sims]
    return movies.iloc[ids][["title", "vote_average", "release_date", "overview"]]


# ---------------- UI ----------------
st.markdown("<div class='topbar'><div class='brand'>AI_Movie_Recommender</div></div>", unsafe_allow_html=True)

# --- Search bar with aligned Trending button ---
st.markdown("""
<style>
.search-container {
    display: flex;
    align-items: center;
    justify-content: flex-start;
    gap: 10px;
    padding: 0 28px;
    margin-top: -10px;
}
.search-box {
    flex: 1;
}
.trend-btn button {
    background: #e50914;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 8px 18px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s ease;
}
.trend-btn button:hover {
    background: #ff0f1e;
}
</style>
""", unsafe_allow_html=True)

# Create custom HTML layout for search bar + trending
search_col1, search_col2 = st.columns([8, 2])
with search_col1:
    default_val = st.session_state.get("selected_movie", "")
    q = st.text_input("", value=default_val, placeholder="Search movie (e.g. Inception, Avatar)...", key="search_q")
with search_col2:
    align_btn_html = """
    <div class="trend-btn" style="margin-top:26px; display:flex; justify-content:flex-start;">
        <form><button name="trend_btn" type="submit">Trending</button></form>
    </div>
    """
    st.markdown(align_btn_html, unsafe_allow_html=True)

# Handle Trending button click (Streamlit native way)
if st.session_state.get("trend_btn", False):
    st.session_state["selected_movie"] = "Avatar"
    st.session_state["search_trigger"] = True
    st.rerun()

# --- Suggestions list ---
typed = st.session_state.get("search_q", "").strip()
if typed:
    matches = [t for t in movies["title"].dropna().unique() if typed.lower() in t.lower()][:8]
    if matches:
        st.write("Suggestions:")
        cols = st.columns(4)
        for i, m in enumerate(matches):
            c = cols[i % 4]
            if c.button(m, key=f"sug_{i}"):
                st.session_state["selected_movie"] = m
                st.session_state["search_trigger"] = True
                st.rerun()

# --- Show recommendations ---
if st.button("🔍 Search") or st.session_state.get("search_trigger", False):
    st.session_state["search_trigger"] = False
    query = st.session_state.get("selected_movie") or st.session_state.get("search_q") or "Avatar"
    recs = recommend(query, top_n=18)
    if recs.empty:
        recs = movies.sort_values(by="popularity", ascending=False).head(18)[["title", "vote_average", "release_date", "overview"]]

    st.markdown(f"<div class='grid-wrap'><h3 style='color:var(--accent)'>Recommendations for <b>{html.escape(query)}</b></h3></div>", unsafe_allow_html=True)

    rec_list = recs.reset_index().to_dict(orient="records")
    cols_per_row = 6
    for i in range(0, len(rec_list), cols_per_row):
        row_chunk = rec_list[i:i+cols_per_row]
        cols = st.columns(len(row_chunk))
        for j, r in enumerate(row_chunk):
            with cols[j]:
                title = r["title"]
                vote = r.get("vote_average", "")
                release = r.get("release_date", "")
                poster, trailer = fetch_poster_and_trailer(title)
                card_html = f"""
                <div class='card'>
                  <img src="{poster}" class="poster" alt="{html.escape(title)} poster"/>
                  <div class='card-body'>
                    <div><div class='title'>{html.escape(title)}</div>
                    <div class='meta'>⭐ {vote} • {release}</div></div>
                  </div>
                  <div class='play-overlay'>
                    <div class='play-triangle'></div>
                    <div class='play-text'>Play Trailer</div>
                  </div>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)
# 🎯 removed all below-poster buttons/text
# now only hover overlay (play icon) will appear when trailer available

# --- Trailer modal popup ---
if st.session_state.get("modal_open"):
    trailer_url = st.session_state.get("modal_trailer")
    modal_title = st.session_state.get("modal_title", "Trailer")
    st.markdown(f"""
      <div class="modal">
        <div class="modal-card">
          <div class="modal-header">
            <div style="font-weight:800;color:var(--accent)">{html.escape(modal_title)} — Trailer</div>
            <form><button class="close-btn" onclick="document.querySelector('#close_streamlit_btn').click(); return false;">Close</button></form>
          </div>
          <div class="iframe-wrap">
            <iframe src="{trailer_url}" allow="autoplay; encrypted-media" allowfullscreen></iframe>
          </div>
        </div>
      </div>
    """, unsafe_allow_html=True)
    if st.button("Close Trailer", key="close_streamlit_btn"):
        st.session_state["modal_open"] = False
        st.rerun()

st.markdown("<div style='text-align:center;color:gray;padding:20px;'>AI_Movie_Recommender • Posters & Trailers via TMDB • For educational use only</div>", unsafe_allow_html=True)

