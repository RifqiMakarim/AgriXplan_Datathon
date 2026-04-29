import warnings
warnings.filterwarnings("ignore")

import os
import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from shapely.geometry import Point

# GWR / mgwr
from mgwr.gwr import GWR

# Azure ML
from dotenv import load_dotenv
load_dotenv()

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

@st.cache_resource
def get_azure_credential():
    """Mengambil kredensial Azure secara otomatis untuk Server/Produksi."""
    # DefaultAzureCredential akan mendeteksi kredensial dari Environment Variables, 
    # Managed Identity (di Azure), atau otentikasi Azure CLI secara otomatis.
    return DefaultAzureCredential()

def connect_azure(): 
    try:
        credential = get_azure_credential()
        
        ml_client = MLClient(
            credential=credential,
            subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID"),
            resource_group_name=os.getenv("AZURE_RESOURCE_GROUP"),
            workspace_name=os.getenv("AZURE_WORKSPACE_NAME")
        )
        
        # Tes panggil workspace
        ws = ml_client.workspaces.get(name=os.getenv("AZURE_WORKSPACE_NAME"))
        return f"✅ Berhasil Terhubung ke Azure ML: {ws.name}"
    except Exception as e:
        return f"⚠️ Mode Lokal (Gagal Connect Azure): {e}"

st.set_page_config(
    page_title="AgriXplan - Sistem Pemetaan Produktivitas Padi Indonesia",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Background utama */
  .stApp { background: #0f1117; color: #e2e8f0; }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: #161b27;
    border-right: 1px solid #1e293b;
  }
  [data-testid="stSidebar"] * { color: #cbd5e1 !important; }

  /* Metric cards */
  [data-testid="metric-container"] {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1rem 1.2rem;
  }
  [data-testid="metric-container"] label {
    color: #94a3b8 !important; font-size: 0.78rem;
    text-transform: uppercase; letter-spacing: 0.05em;
  }
  [data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #38bdf8 !important; font-size: 1.45rem; font-weight: 700;
  }

  [data-testid="stMetricDelta"] svg {
    display: none !important;
  }

  [data-testid="column"]:nth-child(3) [data-testid="stMetricDelta"] {
    background: rgba(74, 222, 128, 0.15) !important;
    color: #4ade80 !important;
    padding: 2px 10px;
    border-radius: 12px;
    font-weight: 600;
    width: fit-content;
  }

  [data-testid="column"]:nth-child(4) [data-testid="stMetricDelta"] {
    background: rgba(248, 113, 113, 0.15) !important;
    color: #f87171 !important;
    padding: 2px 10px;
    border-radius: 12px;
    font-weight: 600;
    width: fit-content;
  }

  /* Section headers */
  .section-header {
    font-size: 1.05rem; font-weight: 600; color: #38bdf8;
    border-left: 3px solid #38bdf8;
    padding-left: 0.6rem; margin: 1.2rem 0 0.8rem;
  }

  /* Divider */
  hr { border-color: #1e293b; }

  /* Dataframe */
  [data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

  /* Focus / outline tanpa oranye */
  *:focus { outline-color: #38bdf8 !important; }

  /* Selectbox */
  .stSelectbox > div > div {
    background: #1e293b; border-color: #334155;
    border-radius: 8px; color: #e2e8f0;
  }

  /* Warning badge */
  .data-badge {
    display: inline-block;
    background: #1e3a5f; color: #7dd3fc;
    border-radius: 6px; padding: 2px 8px;
    font-size: 0.75rem; font-weight: 500;
    margin-left: 6px;
  }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD & PREPROCESSING DATA (Azure ML + Local Fallback)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_azure_dataset():
    """Mengambil dataset 'datathon_agri_clean' langsung dari Azure ML."""
    try:
        import mltable
        credential = get_azure_credential()
        ml_client = MLClient(
            credential=credential,
            subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID"),
            resource_group_name=os.getenv("AZURE_RESOURCE_GROUP"),
            workspace_name=os.getenv("AZURE_WORKSPACE_NAME")
        )
        data_asset = ml_client.data.get(name="datathon_agri_clean", version="1")
        tbl = mltable.load(data_asset.path)
        return tbl.to_pandas_dataframe()
    except Exception as e:
        return None

@st.cache_data(show_spinner=False)
def load_and_preprocess(filepath: str = "AgriData.csv") -> gpd.GeoDataFrame:
    """
    Membaca file CSV lokal dan menggabungkan dengan GeoJSON koordinat.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    local_path = os.path.join(current_dir, filepath)
    df_raw = pd.read_csv(local_path, header=0)

    # ── Rename kolom ke nama internal yang bersih ──────────────────────────
    col_map = {
        df_raw.columns[0]: "Provinsi",
        df_raw.columns[1]: "Y_Produktivitas",   # ku/ha
        df_raw.columns[2]: "X1_OPT",            # Lahan OPT (ha)
        df_raw.columns[3]: "X2_Pupuk_NPK",      # Pupuk NPK (ton)
        df_raw.columns[4]: "X3_Suhu",           # Rata-rata suhu (°C)
        df_raw.columns[5]: "X4_CurahHujan",     # Curah hujan (mm)
        df_raw.columns[6]: "X5_Alsintan",       # Bantuan Alsintan (unit)
    }
    df_raw = df_raw.rename(columns=col_map)

    # ── Hapus baris header kosong (row pertama NaN semua) — index tetap ───
    df = df_raw[df_raw["Provinsi"].notna()].copy()

    # ── Bersihkan kolom X3_Suhu: datetime artefak Excel → float ───────────
    def safe_float(v):
        """Konversi nilai ke float; kembalikan NaN jika tidak bisa."""
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            v = v.strip().replace(",", "")
            if v in ("-", "", "–", "—"):
                return np.nan
            try:
                return float(v)
            except ValueError:
                return np.nan
        try:
            import datetime
            if isinstance(v, (pd.Timestamp, datetime.datetime)):
                return np.nan
        except Exception:
            pass
        return np.nan

    num_cols = ["Y_Produktivitas", "X1_OPT", "X2_Pupuk_NPK",
                "X3_Suhu", "X4_CurahHujan", "X5_Alsintan"]

    for col in num_cols:
        df[col] = df[col].apply(safe_float)

    # ── Isi NaN dengan median kolom (robust terhadap outlier) — index tetap ──
    for col in num_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    # ── Hapus baris duplikat Provinsi — index tetap ────────────────────────
    df = df[~df["Provinsi"].duplicated(keep="first")]

    # ── Load GeoJSON dan Merge ─────────────────────────────────────────────
    geo_path = os.path.join(current_dir, "GeoIndonesia.json")
    geo_df = gpd.read_file(geo_path)
    
    prov_map = {
        "D.I Yogyakarta": "Daerah Istimewa Yogyakarta",
        "Kep. Bangka Belitung": "Kepulauan Bangka Belitung"
    }
    df["PROVINSI"] = df["Provinsi"].replace(prov_map)
    
    gdf = geo_df.merge(df, on="PROVINSI", how="inner")
    
    # Ambil titik tengah (centroid) untuk peletakan marker label jika diperlukan
    centroids = gdf.geometry.centroid
    gdf["Longitude"] = centroids.x
    gdf["Latitude"] = centroids.y

    return gdf


# ─────────────────────────────────────────────────────────────────────────────
# 2. PEMODELAN GWR & EKSTRAKSI KOEFISIEN LOKAL (XAI)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_gwr(_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Membangun model Geographically Weighted Regression (GWR) menggunakan
    library `mgwr` dengan kernel Gaussian fixed-bandwidth.

    Strategi bandwidth:
      - Dilakukan grid-search manual atas 5 kandidat bandwidth (dalam derajat)
      - Kriteria pemilihan: AICc terkecil
      - Gaussian kernel dipilih karena stabil untuk n < 100;
        penurunan bobot halus di semua arah dari titik observasi

    Langkah:
      1. Siapkan matriks koordinat dan variabel
      2. Log Transformasi pada variabel bervolume tinggi (X1, X2, X5)
      3. Standarisasi X (z-score per kolom)
      4. Grid-search bandwidth optimal
      5. Fit model GWR final
      6. Ekstrak koefisien lokal per Provinsi
      7. Tentukan Faktor_Paling_Signifikan
    """
    gdf = _gdf.copy()

    # --- Koordinat spasial ---
    coords = list(zip(gdf["Longitude"].values, gdf["Latitude"].values))

    # --- Variabel ---
    feature_cols = ["X1_OPT", "X2_Pupuk_NPK", "X3_Suhu",
                    "X4_CurahHujan", "X5_Alsintan"]
    y = gdf["Y_Produktivitas"].values.reshape(-1, 1)
    
    # --- Transformasi Log (Log1p) untuk variabel skew ---
    X_df = gdf[feature_cols].copy()
    cols_to_log = ["X1_OPT", "X2_Pupuk_NPK", "X5_Alsintan"]
    for c in cols_to_log:
        X_df[c] = np.log1p(X_df[c].values.astype(float))
        
    X = X_df.values.astype(float)

    # Standarisasi X agar skala tidak mendominasi koefisien
    X_mean = X.mean(axis=0)
    X_std  = X.std(axis=0)
    X_std[X_std == 0] = 1   # hindari pembagian nol
    X_scaled = (X - X_mean) / X_std

    # --- Grid-search bandwidth optimal (Gaussian fixed kernel) ---
    # Kandidat dalam satuan derajat koordinat (~111 km per derajat)
    bw_candidates = [3.0, 5.0, 8.0, 12.0, 20.0]
    best_bw, best_aicc = None, np.inf

    for bw in bw_candidates:
        try:
            m_tmp = GWR(coords, y, X_scaled, bw=bw,
                        kernel="gaussian", fixed=True)
            r_tmp = m_tmp.fit()
            if r_tmp.aicc < best_aicc:
                best_aicc = r_tmp.aicc
                best_bw   = bw
        except Exception:
            continue

    if best_bw is None:
        best_bw = 8.0  # fallback aman

    # --- Fit model GWR final ---
    model   = GWR(coords, y, X_scaled, bw=best_bw,
                  kernel="gaussian", fixed=True)
    results = model.fit()
    bw_opt  = best_bw

    # --- Ekstrak koefisien lokal (tanpa intercept; kolom 1–5) ---
    coef_matrix = results.params[:, 1:]  # shape: (n_obs, 5)

    coef_labels = {
        "X1_OPT"       : "X1 (OPT)",
        "X2_Pupuk_NPK" : "X2 (Pupuk NPK)",
        "X3_Suhu"      : "X3 (Suhu)",
        "X4_CurahHujan": "X4 (Curah Hujan)",
        "X5_Alsintan"  : "X5 (Alsintan)",
    }

    for i, col in enumerate(feature_cols):
        gdf[f"coef_{col}"] = coef_matrix[:, i]

    # --- Faktor paling signifikan per baris ---
    coef_cols    = [f"coef_{c}" for c in feature_cols]
    abs_coef     = gdf[coef_cols].abs()
    dominant_idx = abs_coef.idxmax(axis=1)

    col_to_label = {f"coef_{c}": lbl for c, lbl in coef_labels.items()}
    gdf["Faktor_Paling_Signifikan"] = dominant_idx.map(col_to_label)

    # --- Simpan metadata model ---
    gdf.attrs["bw_opt"]    = bw_opt
    gdf.attrs["r2"]        = results.R2
    gdf.attrs["adj_r2"]    = results.adj_R2
    gdf.attrs["aicc"]      = results.aicc
    gdf.attrs["coef_cols"] = coef_cols
    gdf.attrs["feat_lbls"] = list(coef_labels.values())

    return gdf


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Palet warna per variabel
# ─────────────────────────────────────────────────────────────────────────────
COLOR_MAP = {
    "X1 (OPT)"         : "#ef4444",
    "X2 (Pupuk NPK)"   : "#22c55e",
    "X3 (Suhu)"        : "#f59e0b",
    "X4 (Curah Hujan)" : "#38bdf8",
    "X5 (Alsintan)"    : "#818cf8",
}


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISASI: Peta Folium
# ─────────────────────────────────────────────────────────────────────────────
def build_folium_map(gdf: gpd.GeoDataFrame, selected_prov: str) -> folium.Map:
    """
    Peta interaktif Folium:
      - CircleMarker warna gradasi hijau sesuai produktivitas
      - Marker khusus biru untuk provinsi terpilih
      - Popup tabel informatif
      - Legend produksi di pojok kiri bawah
    Tidak ada stroke/highlight oranye; semua warna solid modern.
    """
    center_lat = gdf["Latitude"].mean()
    center_lon = gdf["Longitude"].mean()

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=5,
        tiles="CartoDB Dark_Matter",
        prefer_canvas=True,
    )

    y_min = gdf["Y_Produktivitas"].min()
    y_max = gdf["Y_Produktivitas"].max()

    def prod_to_color(val):
        """Gradasi dari hijau tua (rendah) ke hijau cerah (tinggi)."""
        t = (val - y_min) / (y_max - y_min + 1e-9)
        r = int(15  + (34  - 15)  * t)
        g = int(80  + (197 - 80)  * t)
        b = int(50  + (94  - 50)  * t)
        return f"#{r:02x}{g:02x}{b:02x}"

    for _, row in gdf.iterrows():
        is_sel = row["Provinsi"] == selected_prov
        color  = "#38bdf8" if is_sel else prod_to_color(row["Y_Produktivitas"])
        radius = 16 if is_sel else 11
        weight = 2.5 if is_sel else 1
        border = "#0ea5e9" if is_sel else "#1e293b"

        popup_html = f"""
        <div style="font-family:Inter,sans-serif;min-width:200px;color:#1e293b;">
          <h4 style="margin:0 0 6px;color:#0369a1;">{row['Provinsi']}</h4>
          <table style="font-size:12px;width:100%;border-collapse:collapse;">
            <tr><td>🌾 Produktivitas</td><td><b>{row['Y_Produktivitas']:.2f} ku/ha</b></td></tr>
            <tr><td>🐛 OPT</td><td>{row['X1_OPT']:,.0f} ha</td></tr>
            <tr><td>🧪 Pupuk NPK</td><td>{row['X2_Pupuk_NPK']:,.0f} ton</td></tr>
            <tr><td>🌡️ Suhu</td><td>{row['X3_Suhu']:.2f} °C</td></tr>
            <tr><td>🌧️ Curah Hujan</td><td>{row['X4_CurahHujan']:.1f} mm</td></tr>
            <tr><td>🚜 Alsintan</td><td>{row['X5_Alsintan']:,.0f} unit</td></tr>
            <tr><td>⭐ Faktor Utama</td><td><b>{row['Faktor_Paling_Signifikan']}</b></td></tr>
          </table>
        </div>
        """

        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=radius,
            color=border,
            weight=weight,
            fill=True,
            fill_color=color,
            fill_opacity=0.88,
            popup=folium.Popup(popup_html, max_width=260),
            tooltip=f"{row['Provinsi']} · {row['Y_Produktivitas']:.2f} ku/ha",
        ).add_to(m)

    # Legend overlay
    legend_html = f"""
    <div style="
        position:fixed; bottom:30px; left:30px; z-index:1000;
        background:rgba(15,17,23,0.90); border:1px solid #334155;
        border-radius:10px; padding:10px 14px; font-family:Inter,sans-serif;
        color:#e2e8f0; font-size:12px; min-width:160px;">
      <b style="color:#38bdf8;">Produktivitas Padi</b><br>
      <span style="color:#22c55e;">●</span> Tertinggi ({y_max:.2f} ku/ha)<br>
      <span style="color:#0f5032;">●</span> Terendah  ({y_min:.2f} ku/ha)<br>
      <span style="color:#38bdf8;">●</span> Dipilih
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    return m


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISASI: Bar Chart Koefisien Lokal GWR
# ─────────────────────────────────────────────────────────────────────────────
def build_coef_chart(row: pd.Series, feat_lbls: list) -> go.Figure:
    """
    Horizontal bar chart koefisien GWR lokal untuk satu Provinsi.
    Warna per variabel sesuai COLOR_MAP; tidak ada oranye.
    """
    coef_vals = [
        row["coef_X1_OPT"],
        row["coef_X2_Pupuk_NPK"],
        row["coef_X3_Suhu"],
        row["coef_X4_CurahHujan"],
        row["coef_X5_Alsintan"],
    ]
    colors = [COLOR_MAP[lbl] for lbl in feat_lbls]

    fig = go.Figure(go.Bar(
        x=coef_vals,
        y=feat_lbls,
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:+.3f}" for v in coef_vals],
        textposition="outside",
        textfont=dict(color="#94a3b8", size=11),
    ))

    fig.add_vline(x=0, line_width=1.5, line_dash="dot", line_color="#475569")

    fig.update_layout(
        title=dict(
            text=f"Pengaruh Lokal GWR — {row['Provinsi']}",
            font=dict(color="#e2e8f0", size=13), x=0,
        ),
        plot_bgcolor="#0f1117",
        paper_bgcolor="#0f1117",
        font=dict(color="#94a3b8", family="Inter"),
        xaxis=dict(
            title="Koefisien GWR (terstandarisasi)",
            gridcolor="#1e293b", zeroline=False,
            tickfont=dict(color="#64748b"),
        ),
        yaxis=dict(automargin=True, tickfont=dict(color="#cbd5e1", size=12)),
        margin=dict(l=10, r=30, t=50, b=40),
        height=320,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISASI: Scatter Overview (semua Provinsi)
# ─────────────────────────────────────────────────────────────────────────────
def build_overview_scatter(gdf: gpd.GeoDataFrame) -> go.Figure:
    """
    Bubble chart: X = Pupuk NPK, Y = Produktivitas, ukuran bubble = Alsintan.
    Warna titik berdasarkan Faktor_Paling_Signifikan.
    """
    color_seq = [COLOR_MAP.get(f, "#64748b") for f in gdf["Faktor_Paling_Signifikan"]]
    alsin_max = gdf["X5_Alsintan"].max()

    fig = go.Figure(go.Scatter(
        x=gdf["X2_Pupuk_NPK"],
        y=gdf["Y_Produktivitas"],
        mode="markers+text",
        marker=dict(
            size=gdf["X5_Alsintan"] / (alsin_max + 1e-9) * 35 + 8,
            color=color_seq,
            opacity=0.82,
            line=dict(width=1, color="#1e293b"),
        ),
        text=gdf["Provinsi"].str.replace("Kalimantan ", "Kal. ").str.replace("Sulawesi ", "Sul. ").str.replace("Nusa Tenggara ", "NT ").str.replace("Sumatera ", "Smt. "),
        textposition="top center",
        textfont=dict(color="#94a3b8", size=8),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Produktivitas: %{y:.2f} ku/ha<br>"
            "Pupuk NPK: %{x:,.0f} ton<br>"
            "Alsintan: %{customdata[1]:,.0f} unit<br>"
            "Faktor Utama: %{customdata[2]}<extra></extra>"
        ),
        customdata=gdf[["Provinsi", "X5_Alsintan", "Faktor_Paling_Signifikan"]].values,
    ))

    fig.update_layout(
        title=dict(
            text="Produktivitas Padi vs. Pupuk NPK (ukuran = Bantuan Alsintan)",
            font=dict(color="#e2e8f0", size=13), x=0,
        ),
        plot_bgcolor="#0f1117",
        paper_bgcolor="#0f1117",
        font=dict(color="#94a3b8", family="Inter"),
        xaxis=dict(title="Realisasi Pupuk NPK (ton)", gridcolor="#1e293b",
                   zeroline=False, tickfont=dict(color="#64748b")),
        yaxis=dict(title="Produktivitas Padi (ku/ha)", gridcolor="#1e293b",
                   zeroline=False, tickfont=dict(color="#64748b")),
        margin=dict(l=10, r=10, t=55, b=40),
        height=400,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISASI: Bar Chart Faktor Dominan per Provinsi (rangkuman XAI)
# ─────────────────────────────────────────────────────────────────────────────
def build_dominant_factor_chart(gdf: gpd.GeoDataFrame) -> go.Figure:
    """
    Bar chart horizontal: Top 5 Tertinggi dan Top 5 Terendah produktivitas,
    diwarnai berdasarkan faktor paling signifikan.
    """
    df_sorted = gdf.sort_values("Y_Produktivitas", ascending=True)
    
    if len(df_sorted) > 10:
        df_bottom5 = df_sorted.head(5)
        df_top5 = df_sorted.tail(5)
        dummy = pd.DataFrame([{"Provinsi": "  ...  ", "Y_Produktivitas": 0, "Faktor_Paling_Signifikan": ""}])
        df_filtered = pd.concat([df_bottom5, dummy, df_top5], ignore_index=True)
    else:
        df_filtered = df_sorted

    colors = [COLOR_MAP.get(f, "rgba(0,0,0,0)") if f != "" else "rgba(0,0,0,0)" for f in df_filtered["Faktor_Paling_Signifikan"]]

    fig = go.Figure(go.Bar(
        x=df_filtered["Y_Produktivitas"],
        y=df_filtered["Provinsi"],
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f if f != "" else "" for f in df_filtered["Faktor_Paling_Signifikan"]],
        textposition="outside",
        textfont=dict(color="#64748b", size=9),
        showlegend=False,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Produktivitas: %{x:.2f} ku/ha<br>"
            "Faktor Utama: %{text}<extra></extra>"
        ),
    ))

    fig.update_layout(
        title=dict(
            text="Produktivitas per Provinsi & Faktor Paling Signifikan",
            font=dict(color="#e2e8f0", size=13), x=0,
        ),
        plot_bgcolor="#0f1117",
        paper_bgcolor="#0f1117",
        font=dict(color="#94a3b8", family="Inter"),
        xaxis=dict(title="Produktivitas Padi (ku/ha)", gridcolor="#1e293b",
                   zeroline=False, tickfont=dict(color="#64748b")),
        yaxis=dict(automargin=True, tickfont=dict(color="#cbd5e1", size=10)),
        showlegend=False,
        margin=dict(l=10, r=60, t=55, b=40),
        height=400,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISASI: Bar Chart Distribusi Variabel
# ─────────────────────────────────────────────────────────────────────────────
def build_variable_chart(gdf: gpd.GeoDataFrame, col_name: str, title_label: str) -> go.Figure:
    """Bar chart untuk satu variabel spesifik di semua provinsi."""
    df_sorted = gdf.sort_values(col_name, ascending=True)
    
    color_map = {
        "Y_Produktivitas": "#10b981", 
        "X1_OPT": "#ef4444",          
        "X2_Pupuk_NPK": "#22c55e",    
        "X3_Suhu": "#f59e0b",         
        "X4_CurahHujan": "#38bdf8",   
        "X5_Alsintan": "#818cf8",     
    }
    color = color_map.get(col_name, "#38bdf8")

    fig = go.Figure(go.Bar(
        x=df_sorted[col_name],
        y=df_sorted["Provinsi"],
        orientation="h",
        marker=dict(color=color, line=dict(width=0)),
        text=[f"{v:,.2f}" if "Suhu" in col_name or "Curah" in col_name or "Produktivitas" in col_name else f"{v:,.0f}" for v in df_sorted[col_name]],
        textposition="outside",
        textfont=dict(color="#64748b", size=10),
    ))

    fig.update_layout(
        title=dict(
            text=f"Distribusi {title_label} per Provinsi",
            font=dict(color="#e2e8f0", size=13), x=0,
        ),
        plot_bgcolor="#0f1117",
        paper_bgcolor="#0f1117",
        font=dict(color="#94a3b8", family="Inter"),
        xaxis=dict(title=title_label, gridcolor="#1e293b",
                   zeroline=False, tickfont=dict(color="#64748b")),
        yaxis=dict(automargin=True, tickfont=dict(color="#cbd5e1", size=10)),
        margin=dict(l=10, r=60, t=55, b=40),
        height=max(450, len(gdf) * 20),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. MAIN APP — LAYOUT STREAMLIT
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # ── Load & proses data ──────────────────────────────────────────────────
    with st.spinner("📂 Membaca & membersihkan data AgriData.csv..."):
        gdf_clean = load_and_preprocess("AgriData.csv")

    with st.spinner("⚙️ Menjalankan model GWR (grid-search bandwidth)..."):
        gdf = run_gwr(gdf_clean)

    feat_lbls = gdf.attrs["feat_lbls"]

    # ── SIDEBAR ─────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🌾 AgriXplan")
        
        # Tes koneksi Azure ML
        st.success(connect_azure())
        st.markdown("---")

        st.markdown("### 📍 Pilih Provinsi")
        prov_list = sorted(gdf["Provinsi"].tolist())
        selected_prov = st.selectbox("Provinsi", options=prov_list, index=0)

        st.markdown("---")
        st.markdown("### 📊 Statistik Model GWR")
        st.metric("R² Global",      f"{gdf.attrs['r2']:.4f}")
        st.metric("Adj. R²",        f"{gdf.attrs['adj_r2']:.4f}")
        st.metric("AICc",           f"{gdf.attrs['aicc']:.2f}")
        st.metric("Bandwidth Opt.", f"{gdf.attrs['bw_opt']:.1f}° (Gaussian)")
        st.metric("Jumlah Provinsi", f"{len(gdf)}")

        st.markdown("---")
        st.caption(
            "**Variabel Model**\n"
            "- Y : Produktivitas padi (ku/ha)\n"
            "- X1: Lahan OPT (ha)\n"
            "- X2: Pupuk NPK bersubsidi (ton)\n"
            "- X3: Rata-rata suhu (°C)\n"
            "- X4: Curah Hujan (mm)\n"
            "- X5: Bantuan Alsintan (unit)"
        )
        st.markdown("---")
        st.caption("Microsoft Datathon · Ketahanan Pangan & Agrikultur Modern")

    # ── HEADER ──────────────────────────────────────────────────────────────
    st.markdown(
        "<h1 style='color:#38bdf8;margin-bottom:0;'>"
        "🌾 AgriXplan </h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h3 style='color:#64748b;margin-top:4px;'>"
        " Sistem Pemetaan Produktivitas Padi Indonesia</h3>",
        unsafe_allow_html=True,
    )

    # ── TABS ────────────────────────────────────────────────────────────────
    tab_dash, tab_data, tab_insight = st.tabs(["Dashboard", "Dataset", "Insight"])

    with tab_dash:
        # ── METRIC CARDS ────────────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Total Provinsi",        f"{len(gdf)}")
        with c2: st.metric("Rata-rata Produktivitas", f"{gdf['Y_Produktivitas'].mean():.2f} ku/ha")
        with c3: st.metric("Tertinggi", f"{gdf['Y_Produktivitas'].max():.2f} ku/ha",
                            delta=gdf.loc[gdf['Y_Produktivitas'].idxmax(), 'Provinsi'])
        with c4: st.metric("Terendah",  f"{gdf['Y_Produktivitas'].min():.2f} ku/ha",
                            delta=gdf.loc[gdf['Y_Produktivitas'].idxmin(), 'Provinsi'])

        st.markdown("---")

        # ── PETA + BAR CHART ────────────────────────────────────────────────────
        map_col, chart_col = st.columns([1.4, 1], gap="large")

        with map_col:
            st.markdown(
                "<div class='section-header'>🗺️ Peta Interaktif Produktivitas Padi</div>",
                unsafe_allow_html=True,
            )
            m = build_folium_map(gdf, selected_prov)
            st_folium(m, width=None, height=450, returned_objects=[])

        with chart_col:
            st.markdown(
                "<div class='section-header'>📊 Koefisien GWR Lokal per Variabel</div>",
                unsafe_allow_html=True,
            )
            sel_row = gdf[gdf["Provinsi"] == selected_prov].iloc[0]
            fig_bar = build_coef_chart(sel_row, feat_lbls)
            st.plotly_chart(fig_bar, use_container_width=True,
                            config={"displayModeBar": False})

            st.metric("🌾 Produktivitas", f"{sel_row['Y_Produktivitas']:.2f} ku/ha")

        st.markdown("---")

        # ── CHART PERINGKAT PRODUKTIVITAS SEMUA PROVINSI ────────────────────────
        st.markdown(
            "<div class='section-header'>Peringkat Produktivitas per Provinsi</div>",
            unsafe_allow_html=True,
        )
        fig_dom = build_dominant_factor_chart(gdf)
        st.plotly_chart(fig_dom, use_container_width=True,
                        config={"displayModeBar": False})

        st.markdown("---")

        # ── OVERVIEW SCATTER ─────────────────────────────────────────────────────
        st.markdown(
            "<div class='section-header'>Produktivitas vs. Pupuk NPK</div>",
            unsafe_allow_html=True,
        )
        fig_sc = build_overview_scatter(gdf)
        st.plotly_chart(fig_sc, use_container_width=True,
                        config={"displayModeBar": False})

        st.markdown("---")

        # ── VISUALISASI TIAP VARIABEL ──────────────────────────────────────────
        st.markdown(
            "<div class='section-header'>📈 Analisis Distribusi Tiap Variabel</div>",
            unsafe_allow_html=True,
        )
        
        var_choices = {
            "Produktivitas Padi (ku/ha)": "Y_Produktivitas",
            "Luas Lahan Terkena OPT (ha)": "X1_OPT",
            "Penggunaan Pupuk NPK (ton)": "X2_Pupuk_NPK",
            "Rata-rata Suhu (°C)": "X3_Suhu",
            "Curah Hujan (mm)": "X4_CurahHujan",
            "Bantuan Alsintan (unit)": "X5_Alsintan"
        }
        
        selected_var_name = st.selectbox("Pilih Variabel untuk Divisualisasikan:", options=list(var_choices.keys()))
        selected_col = var_choices[selected_var_name]
        
        fig_var = build_variable_chart(gdf, selected_col, selected_var_name)
        st.plotly_chart(fig_var, use_container_width=True, config={"displayModeBar": False})

    with tab_data:

        display_cols = [
            "Provinsi",
            "Y_Produktivitas", "X1_OPT", "X2_Pupuk_NPK",
            "X3_Suhu", "X4_CurahHujan", "X5_Alsintan",
            "coef_X1_OPT", "coef_X2_Pupuk_NPK", "coef_X3_Suhu",
            "coef_X4_CurahHujan", "coef_X5_Alsintan",
            "Faktor_Paling_Signifikan"
        ]
        display_df = gdf[display_cols].copy()
        display_df.columns = [
            "Provinsi",
            "Produktivitas (ku/ha)", "X1 OPT (ha)", "X2 Pupuk NPK (ton)",
            "X3 Suhu (°C)", "X4 Curah Hujan (mm)", "X5 Alsintan (unit)",
            "Coef X1", "Coef X2", "Coef X3", "Coef X4", "Coef X5",
            "Kategori Faktor Dominan"
        ]
        
        st.markdown("#### 🔎 Filter & Pengurutan Data")
        col_f1, col_f2 = st.columns(2)
            
        with col_f1:
            sort_mapping = {
                "Default": None,
                "Produktivitas (ku/ha)": "Produktivitas (ku/ha)",
                "Pengaruh OPT (Coef X1)": "Coef X1",
                "Pengaruh Pupuk NPK (Coef X2)": "Coef X2",
                "Pengaruh Suhu (Coef X3)": "Coef X3",
                "Pengaruh Curah Hujan (Coef X4)": "Coef X4",
                "Pengaruh Alsintan (Coef X5)": "Coef X5",
                "Nilai Aktual OPT": "X1 OPT (ha)",
                "Nilai Aktual Pupuk NPK": "X2 Pupuk NPK (ton)",
                "Nilai Aktual Suhu": "X3 Suhu (°C)",
                "Nilai Aktual Curah Hujan": "X4 Curah Hujan (mm)",
                "Nilai Aktual Alsintan": "X5 Alsintan (unit)",
            }
            sort_by_label = st.selectbox("Urutkan Berdasarkan:", options=list(sort_mapping.keys()))
            
        with col_f2:
            sort_order = st.selectbox("Arah Urutan:", options=["Tertinggi ke Terendah", "Terendah ke Tertinggi"])
            
        # Terapkan Sorting
        if sort_mapping[sort_by_label]:
            is_ascending = (sort_order == "Terendah ke Tertinggi")
            display_df = display_df.sort_values(by=sort_mapping[sort_by_label], ascending=is_ascending)

        def highlight_selected(s):
            is_sel = s["Provinsi"] == selected_prov
            return [
                "background-color:#0c1a2e;color:#38bdf8;font-weight:600;" if is_sel else ""
                for _ in s
            ]

        fmt = {
            "Produktivitas (ku/ha)" : "{:.2f}",
            "X1 OPT (ha)"           : "{:,.0f}",
            "X2 Pupuk NPK (ton)"    : "{:,.0f}",
            "X3 Suhu (°C)"          : "{:.2f}",
            "X4 Curah Hujan (mm)"   : "{:.1f}",
            "X5 Alsintan (unit)"    : "{:,.0f}",
            "Coef X1" : "{:+.4f}",
            "Coef X2" : "{:+.4f}",
            "Coef X3" : "{:+.4f}",
            "Coef X4" : "{:+.4f}",
            "Coef X5" : "{:+.4f}",
        }

        styled = (
            display_df.style
            .apply(highlight_selected, axis=1)
            .format(fmt)
            .set_table_styles([
                {"selector": "thead th",
                 "props": [("background-color", "#1e293b"), ("color", "#94a3b8"),
                           ("font-size", "11px"), ("text-transform", "uppercase")]},
                {"selector": "tbody tr:hover td",
                 "props": [("background-color", "#1e293b")]},
            ])
        )
        st.dataframe(styled, use_container_width=True, height=480)

        # Download CSV
        csv = display_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Unduh Dataset sebagai CSV",
            data=csv,
            file_name="gwr_padi_indonesia.csv",
            mime="text/csv",
        )

    with tab_insight:
        st.markdown(
            "<div class='section-header'>💡 Insight</div>",
            unsafe_allow_html=True,
        )
        st.info("Coming soon...")

    st.markdown("---")
    st.caption(
        "© 2026 · AgriXplan" 
    )


if __name__ == "__main__":
    main()
