# -------- nomes bonitos para exibição --------
def pretty_model_name(raw: str) -> str:
    """
    Converte rótulos do CSV para os nomes desejados para exibição:
      GPT-4o • Deep-Seek V-3 • Sabia 3.1 • Pluralidade • Borda • Gemini 1.5 Flash • GPT 4o-Mini
    """
    m = _norm(raw).replace("\u2011", "-")  # normaliza

    # Agregados primeiro
    if "borda" in m: return "Borda"
    if "plural" in m: return "Pluralidade"

    # Modelos (com as novas regras)
    if "4o-mini" in m or "gpt-4o-mini" in m or "gpt 4o-mini" in m or "gpt-mini" in m:
        return "GPT 4o-Mini"
    if "gpt-4o" in m or "gpt 4o" in m:
        return "GPT-4o"
    if "deep" in m and "seek" in m:
        return "Deep-Seek V-3"
    if "sabi" in m or "maritalk" in m:  # cobre "sabia", "sabiá" e agora "maritalk"
        return "Sabia 3.1"
    if "gemini" in m:  # captura "gemini", "gemini 1.5 flash", etc.
        return "Gemini 1.5 Flash"

    # fallback: mantém original
    return str(raw).strip()# -*- coding: utf-8 -*-
import os
import re
import json
import unicodedata
import pandas as pd
import streamlit as st

# ================== CONFIG DA PÁGINA ==================
st.set_page_config(
    page_title="CID-10 • Ensemble",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",   # evita “respiro” do sidebar
)

# --- Injeta PWA sem adicionar altura/espaço visual ---
import streamlit.components.v1 as components
components.html(
    """
    <link rel="manifest" href="manifest.json">
    <script>
      if ('serviceWorker' in navigator) {
        navigator.serviceWorker.register('service-worker.js');
      }
    </script>
    """,
    height=0
)

# ================== ESTILOS (reduzir respiro no topo) ==================
st.markdown("""
<style>
/* remove espaços padrão do topo do app */
[data-testid="stAppViewContainer"] { padding-top: 0 !important; }
section.main > div.block-container { padding-top: .25rem !important; }

/* mobile/tablet ainda mais compacto */
@media (max-width: 820px){
  section.main > div.block-container { padding-top: .15rem !important; }
}

/* esconde cabeçalho/rodapé do Streamlit */
#MainMenu, header, footer {visibility: hidden;}

/* título e subtítulo com margem mínima */
h1 { margin-top: .15rem !important; }
h2 { margin-top: .15rem !important; }

/* reduz margem vertical padrão criada por <p> gerados pelo markdown */
.block-container p { margin-top: .25rem; }

/* melhora espaçamento de controles para toque sem aumentar topo */
.stRadio > div, .stSelectbox > div { padding: .35rem 0; }
</style>
""", unsafe_allow_html=True)


# Paleta (cores bem distintas por MÉTRICA)
# Paleta (cores VIVAS e bem distintas por MÉTRICA)
COLOR_PREC = "#F97316"    # Precisão   (Laranja Vibrante)
COLOR_REC  = "#22C55E"    # Recall     (Verde Esmeralda)
COLOR_F1   = "#3B82F6"    # F1         (Azul Royal)
PRIMARY    = "#0F3D7A"   # títulos e chips
SECOND     = "#246BCE"

# ================== HELPERS ==================
def _norm(s: str) -> str:
    """normaliza string (minúsculas, sem acento) para casar nomes/labels."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    return s.lower().strip()

@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # normaliza algumas colunas comuns
    if "Modelo" in df.columns:
        df["Modelo"] = (
            df["Modelo"].astype(str)
            .str.replace("\u2011", "-", regex=False)  # hífen não separável
            .str.replace("\u00A0", " ", regex=False)  # espaço duro
            .str.strip()
        )
    if "Tipo" in df.columns:
        df["Tipo"] = df["Tipo"].astype(str).str.strip()
    if "k" in df.columns:
        try:
            df["k"] = df["k"].astype(int)
        except Exception:
            pass
    return df

@st.cache_data(show_spinner=False)
def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def resolve_metric_columns(df: pd.DataFrame, agg_choice: str):
    """
    Encontra as colunas de Precisão/Recall/F1 para 'Micro' ou 'Macro',
    mesmo com variações de nomes (port/inglês; com/sem acento).
    Retorna: {'precision': <col>, 'recall': <col>, 'f1': <col>}
    """
    agg_norm = _norm(agg_choice)  # 'micro' ou 'macro'
    cols = {"precision": None, "recall": None, "f1": None}

    patterns = {
        "precision": [rf"^{agg_norm}[_\s-]*prec(isao|ision|is|ao)?"],
        "recall":    [rf"^{agg_norm}[_\s-]*(recall|sensibilidade)"],
        "f1":        [rf"^{agg_norm}[_\s-]*f1"]
    }

    for c in df.columns:
        cn = _norm(c)
        for key, pats in patterns.items():
            if cols[key] is None and any(re.match(p, cn) for p in pats):
                cols[key] = c

    # fallbacks canônicos
    for key, canon in {
        "precision": f"{agg_choice}_Precision",
        "recall":    f"{agg_choice}_Recall",
        "f1":        f"{agg_choice}_F1",
    }.items():
        if cols[key] is None and canon in df.columns:
            cols[key] = canon

    missing = [k for k, v in cols.items() if v is None]
    if missing:
        st.error(
            f"Colunas de métricas não encontradas para {agg_choice}: faltando {missing}. "
            f"Verifique nomes no resultados.csv (ex.: {agg_choice}_Precision, {agg_choice}_Recall, {agg_choice}_F1)."
        )
        st.stop()
    return cols

def is_aggregate(modelo: str) -> bool:
    """Identifica agregados de forma robusta (apenas se começar com 'Agregado_' ou contiver 'borda'/'plural')."""
    m = _norm(modelo).replace("\u2011", "-")
    return m.startswith("agregado_") or ("borda" in m) or ("plural" in m)

def best_individual_delta(df_slice: pd.DataFrame, f1_col: str):
    """
    Calcula Δ dos agregados vs melhor individual (pela F1 da agregação escolhida).
    Retorna: (best_name, best_val, val_borda, val_plural, delta_borda, delta_plural)
    """
    if f1_col not in df_slice.columns:
        return None
    ind = df_slice[~df_slice["Modelo"].apply(is_aggregate)].copy()
    agg = df_slice[df_slice["Modelo"].apply(is_aggregate)].copy()
    if ind.empty:
        return None

    best_row = ind.sort_values(f1_col, ascending=False).iloc[0]
    best_name = str(best_row["Modelo"])
    best_val  = float(best_row[f1_col])

    val_borda = None
    val_plural = None
    for _, r in agg.iterrows():
        name = _norm(r["Modelo"])
        if "borda" in name:
            val_borda = float(r[f1_col])
        if "plural" in name:
            val_plural = float(r[f1_col])

    delta_borda = (None if val_borda is None else (val_borda - best_val))
    delta_plural = (None if val_plural is None else (val_plural - best_val))
    return (best_name, best_val, val_borda, val_plural, delta_borda, delta_plural)

# -------- nomes bonitos para exibição --------
# -------- nomes bonitos para exibição --------
def pretty_model_name(raw: str) -> str:
    """
    Converte rótulos do CSV para os nomes desejados para exibição:
      GPT-4o • Deep-Seek V-3 • Sabia 3.1 • Pluralidade • Borda • Gemini 1.5 Flash • GPT 4o-Mini
    """
    m = _norm(raw).replace("\u2011", "-")  # normaliza

    # Agregados primeiro
    if "borda" in m: return "Borda"
    if "plural" in m: return "Pluralidade"

    # Modelos (com as novas regras)
    if "4o-mini" in m or "gpt-4o-mini" in m or "gpt 4o-mini" in m or "gpt-mini" in m:
        return "GPT 4o-Mini"
    if "gpt-4o" in m or "gpt 4o" in m:
        return "GPT-4o"
    if "deep" in m and "seek" in m:
        return "Deep-Seek V-3"
    if "sabi" in m or "maritalk" in m:  # cobre "sabia", "sabiá" e agora "maritalk"
        return "Sabia 3.1"
    if "gemini" in m:  # captura "gemini", "gemini 1.5 flash", etc.
        return "Gemini 1.5 Flash"

    # fallback: mantém original
    return str(raw).strip()

def style_title(txt: str):
    st.markdown(f"<h2 style='margin-top:0.25rem;color:{PRIMARY};'>{txt}</h2>", unsafe_allow_html=True)

def style_subtitle(txt: str):
    st.markdown(f"<h4 style='margin-top:0.25rem;color:#222;'>{txt}</h4>", unsafe_allow_html=True)

def chip(label: str, value: str, color="#fff", bg=PRIMARY):
    st.markdown(
        f"""
        <div style="display:inline-block;margin:6px 10px 0 0;padding:8px 12px;border-radius:999px;
                    background:{bg};color:{color};font-weight:600;font-size:0.95rem;">
            {label}: {value}
        </div>
        """,
        unsafe_allow_html=True
    )

# ================== CARREGA DADOS ==================
DATA_DIR = "."

csv_path  = os.path.join(DATA_DIR, "resultados.csv")
json_path = os.path.join(DATA_DIR, "stats.json")

if not os.path.isfile(csv_path):
    st.error(f"Arquivo não encontrado: {csv_path}")
    st.stop()
if not os.path.isfile(json_path):
    st.error(f"Arquivo não encontrado: {json_path}")
    st.stop()

df = load_csv(csv_path)
stats = load_json(json_path)

# ================== HEADER ==================
# ================== HEADER ==================
st.markdown(
    f"""
    <div style="padding:4px 0 0 0; text-align:left;">
        <h1 style="margin:0;color:{PRIMARY}; font-size:28px; font-weight:700; line-height:1.2;">
            Agregação de predições de grandes modelos de linguagem via métodos de decisão em grupo para a codificação automática de diagnósticos clínicos
        </h1>
        <p style="margin:4px 0 0 0; color:#1f77b4; font-size:16px; font-weight:600; line-height:1.4;">
            Ricardo da Silva Santos (UNICAMP) • 
            Murilo Gleyson Gazzola (MACKENZIE) • 
            Renato Teixeira Souza (UNICAMP) • 
            Rodolfo de Carvalho Pacagnella (UNICAMP) • 
            Cristiano Torezan (UNICAMP)
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ================== ABAS ==================
tabs = st.tabs(["🗂️ Dashboard", "📊 Estatísticas", "🧪 Exemplos"])


# ================== DASHBOARD ==================

with tabs[0]:
    style_title("Dashboard")

    # Seletor para escolher o tipo de análise
    modo_analise = st.radio(
        "Selecione o modo de análise:",
        ("Comparar Modelos", "Analisar 'k' por Modelo"),
        horizontal=True,
        label_visibility="collapsed"
    )

    # --- MODO 1: COMPARAR DIFERENTES MODELOS ---
    if modo_analise == "Comparar Modelos":
        # Filtros: Tipo, k e Agregação
        colA, colB, colC = st.columns([1,1,1], gap="medium")
        tipos = sorted(df["Tipo"].dropna().unique().tolist()) if "Tipo" in df.columns else ["Full"]
        ks    = sorted(df["k"].dropna().unique().tolist()) if "k" in df.columns else [3]
        with colA:
            tipo = st.selectbox("Tipo", tipos, index=0)
        with colB:
            k = st.selectbox("k", ks, index=0)
        with colC:
            agg_choice = st.radio("Agregação", ["Micro", "Macro"], index=0, horizontal=True)

        # Filtra (Tipo, k)
        mask = (df["Tipo"].astype(str) == str(tipo)) & (df["k"] == int(k))
        view = df.loc[mask].copy()
        if view.empty:
            st.warning("Sem dados para esse filtro.")
            st.stop()

        # Resolve nomes das colunas de métricas
        metric_cols = resolve_metric_columns(view, agg_choice)

        # Ordena por F1
        view = view.sort_values(metric_cols["f1"], ascending=False)

        # Tabela (opcional) com nomes bonitos
        with st.expander("Ver tabela de resultados (ordenada)", expanded=False):
            cols_to_show = ["Modelo", metric_cols["precision"], metric_cols["recall"], metric_cols["f1"]]
            for extra in ["TP","FP","FN"]:
                if extra in view.columns:
                    cols_to_show.append(extra)
            df_show = view[cols_to_show].copy()
            df_show["Modelo"] = df_show["Modelo"].apply(pretty_model_name)
            df_show = df_show.rename(columns={
                metric_cols["precision"]: f"{agg_choice}_Precisão",
                metric_cols["recall"]:    f"{agg_choice}_Recall",
                metric_cols["f1"]:        f"{agg_choice}_F1"
            })
            st.dataframe(df_show, use_container_width=True, hide_index=True)

        # Prepara dados no formato tidy para o gráfico
        plot_rows = []
        for _, r in view.iterrows():
            modelo = pretty_model_name(str(r["Modelo"]))
            plot_rows.append({"Modelo": modelo, "Métrica": "Precisão", "Valor": float(r[metric_cols["precision"]])})
            plot_rows.append({"Modelo": modelo, "Métrica": "Recall",   "Valor": float(r[metric_cols["recall"]])})
            plot_rows.append({"Modelo": modelo, "Métrica": "F1",       "Valor": float(r[metric_cols["f1"]])})
        df_plot = pd.DataFrame(plot_rows)
        df_plot = df_plot.groupby(["Modelo", "Métrica"], as_index=False, sort=False)["Valor"].max()

        HIGHLIGHT = {"Borda", "Pluralidade"}
        df_plot["Agregado"] = df_plot["Modelo"].isin(HIGHLIGHT)

        df_f1_scores = df_plot[df_plot["Métrica"] == "F1"].sort_values("Valor", ascending=False)
        order_domain = df_f1_scores["Modelo"].tolist()

        # --- Gráfico de Barras Agrupadas ---
        try:
            import altair as alt

            METRIC_COLORS = {"Precisão": COLOR_PREC, "Recall": COLOR_REC, "F1": COLOR_F1}

            base = (
                alt.Chart(df_plot).mark_bar().encode(
                    x=alt.X("Modelo:N", scale=alt.Scale(domain=order_domain), axis=alt.Axis(title=None, labelAngle=0, labelFontSize=14, labelColor="#222")),
                    y=alt.Y("Valor:Q", title=f"{agg_choice} (valor)"),
                    color=alt.Color("Métrica:N", scale=alt.Scale(domain=list(METRIC_COLORS.keys()), range=[METRIC_COLORS[m] for m in METRIC_COLORS]), legend=alt.Legend(title="Métrica")),
                    xOffset="Métrica:N",
                    tooltip=["Modelo", "Métrica", alt.Tooltip("Valor:Q", format=".4f")]
                ).properties(height=460)
            )

            overlay = (
                alt.Chart(df_plot[df_plot["Agregado"]]).mark_bar(stroke="#111", strokeWidth=2, filled=False).encode(
                    x=alt.X("Modelo:N", scale=alt.Scale(domain=order_domain), axis=alt.Axis(title=None)),
                    y=alt.Y("Valor:Q"),
                    xOffset="Métrica:N",
                )
            )

            text = base.mark_text(
                align='center', baseline='bottom', dx=2, dy=0, fontSize=8
            ).encode(
                text=alt.Text('Valor:Q', format='.2f'),
                color=alt.value('black'),

