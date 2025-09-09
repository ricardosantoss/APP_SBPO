# -*- coding: utf-8 -*-
import json
import os
import unicodedata
import pandas as pd
import streamlit as st

# ============== CONFIGURAÇÃO BÁSICA ==============
st.set_page_config(
    page_title="CID-10 • Ensemble",
    page_icon="🧠",
    layout="wide"
)

PRIMARY = "#0F3D7A"   # Borda
SECOND  = "#246BCE"   # Pluralidade
GRAY    = "#B0B0B0"   # individuais (cinza claro)
BG      = "#FFFFFF"

# ============== HELPERS ==============
@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normaliza nomes com hífen não separável e espaços estranhos, se houver
    if "Modelo" in df.columns:
        df["Modelo"] = (
            df["Modelo"]
            .astype(str)
            .str.replace("\u2011", "-", regex=False)
            .str.replace("\u00A0", " ", regex=False)
            .str.strip()
        )
    if "Tipo" in df.columns:
        df["Tipo"] = df["Tipo"].astype(str).str.strip()
    if "k" in df.columns:
        # Algumas vezes vem como float; garanta int
        try:
            df["k"] = df["k"].astype(int)
        except Exception:
            pass
    return df

@st.cache_data(show_spinner=False)
def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def is_aggregate(modelo: str) -> bool:
    m = modelo.lower()
    # Ajuste aqui se seus nomes tiverem outro padrão (ex.: "Agregado_Borda", "Borda", "Pluralidade")
    return ("agregado" in m) or ("borda" in m) or ("pluralidade" in m)

def color_for_model(modelo: str) -> str:
    """
    Define cor por tipo de modelo:
    - Agregado_Borda → azul escuro
    - Agregado_Pluralidade → azul médio
    - Outros agregados → azul médio
    - Individuais (API GPT, Maritalk, Gemini etc.) → cinza claro
    """
    m = (
        str(modelo)
        .lower()
        .replace("\u2011", "-")   # hífen não separável
        .strip()
    )

    if m.startswith("agregado_borda"):
        return PRIMARY
    if m.startswith("agregado_plural"):
        return SECOND
    if m.startswith("agregado"):
        return SECOND
    return GRAY

def best_individual_delta(df_slice: pd.DataFrame, metric_col: str = "Micro_F1"):
    """Retorna (melhor_individual, valor_melhor, valor_borda, valor_plural, delta_borda, delta_plural)."""
    # separa agregados vs individuais
    ind = df_slice[~df_slice["Modelo"].apply(is_aggregate)].copy()
    agg = df_slice[df_slice["Modelo"].apply(is_aggregate)].copy()
    if ind.empty:
        return None

    best_ind_row = ind.sort_values(metric_col, ascending=False).iloc[0]
    best_name = best_ind_row["Modelo"]
    best_val  = float(best_ind_row[metric_col])

    # pega valores dos agregados, se existirem
    val_borda = None
    val_plural = None
    for _, r in agg.iterrows():
        name = r["Modelo"].lower()
        if "borda" in name:
            val_borda = float(r[metric_col])
        if ("plural" in name) or ("pluralidade" in name):
            val_plural = float(r[metric_col])

    delta_borda = (None if val_borda is None else (val_borda - best_val))
    delta_plural = (None if val_plural is None else (val_plural - best_val))

    return (best_name, best_val, val_borda, val_plural, delta_borda, delta_plural)

def pct(x):
    return f"{100.0*x:.2f}%"

def style_title(txt: str):
    st.markdown(f"<h2 style='margin-top:0.25rem;color:{PRIMARY};'>{txt}</h2>", unsafe_allow_html=True)

def style_subtitle(txt: str):
    st.markdown(f"<h4 style='margin-top:0.25rem;color:#222;'>{txt}</h4>", unsafe_allow_html=True)

def as_small_chip(label: str, value: str, color="#222", bg="#f4f6f8"):
    st.markdown(
        f"""
        <div style="display:inline-block;margin:4px 8px 4px 0;padding:6px 10px;border-radius:999px;
                    background:{bg};color:{color};font-weight:600;font-size:0.9rem;">
            {label}: {value}
        </div>
        """,
        unsafe_allow_html=True
    )

# ============== CARREGA DADOS (RESULTADOS + STATS) ==============
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

# ============== HEADER ==============
st.markdown(
    f"""
    <div style="padding:8px 0 0 0">
        <h1 style="margin:0;color:{PRIMARY};">Ensemble simples, ganho real</h1>
        <p style="margin:4px 0 0 0;color:#333;">
            Comparação de modelos individuais vs. agregação (Pluralidade / Borda) para codificação CID-10.
            App offline com dados pré-computados.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ============== "BOTÕES" PRINCIPAIS ==============
# Para uma UX estável em Streamlit, tabs dão menos fricção do que st.button para "páginas".
tabs = st.tabs(["🗂️ Dashboard", "📊 Estatísticas"])

# ============== DASHBOARD ==============
with tabs[0]:
    style_title("Dashboard")

    # Filtros essenciais
    colA, colB, colC = st.columns([1,1,1], gap="medium")
    tipos = sorted(df["Tipo"].dropna().unique().tolist()) if "Tipo" in df.columns else ["Full"]
    ks = sorted(df["k"].dropna().unique().tolist()) if "k" in df.columns else [3]
    metrica_opts = ["Micro_F1", "Macro_F1"]
    with colA:
        tipo = st.selectbox("Tipo", tipos, index=0)
    with colB:
        k = st.selectbox("k", ks, index=0)
    with colC:
        metrica = st.selectbox("Métrica", metrica_opts, index=0)

    # Filtra a visão para (Tipo, k)
    mask = (df["Tipo"].astype(str) == str(tipo)) & (df["k"] == int(k))
    view = df.loc[mask].copy()
    if view.empty:
        st.warning("Sem dados para esse filtro.")
        st.stop()

    # Ordena por métrica
    view = view.sort_values(metrica, ascending=False)

    # Tabela compacta opcional
    with st.expander("Ver tabela de resultados (ordenada)", expanded=False):
        st.dataframe(
            view[["Modelo", "Micro_F1", "Macro_F1", "TP", "FP", "FN"]],
            use_container_width=True,
            hide_index=True
        )

    # Cores por modelo
    colors = [color_for_model(m) for m in view["Modelo"]]
    # Render gráfico de barras com altair (leve e nativo)
    try:
        import altair as alt
        chart = (
            alt.Chart(view)
            .mark_bar()
            .encode(
                x=alt.X("Modelo:N", sort=None, axis=alt.Axis(labelAngle=-20, title=None)),
                y=alt.Y(f"{metrica}:Q", title=metrica.replace("_", " ")),
                color=alt.Color("Modelo:N",
                                scale=alt.Scale(range=colors),
                                legend=None),
                tooltip=["Modelo", "Micro_F1", "Macro_F1", "TP", "FP", "FN"]
            )
            .properties(height=380)
        )
        st.altair_chart(chart, use_container_width=True)
    except Exception:
        # Fallback de tabela colorida (se altair não estiver no requirements.txt)
        st.info("Para o gráfico, inclua 'altair' no requirements.txt. Exibindo tabela como fallback.")
        st.dataframe(view, use_container_width=True)

    # Deltas vs melhor individual
    info = best_individual_delta(view, metric_col=metrica)
    if info:
        best_name, best_val, val_borda, val_plural, d_borda, d_plural = info
        style_subtitle("Δ vs. melhor modelo individual")
        as_small_chip("Melhor Individual", f"{best_name} ({best_val:.4f})", color="#fff", bg=PRIMARY)
        if val_borda is not None:
            as_small_chip("Borda", f"{val_borda:.4f} ({'+' if d_borda>=0 else ''}{d_borda:.4f})", color="#fff", bg=PRIMARY)
        if val_plural is not None:
            as_small_chip("Pluralidade", f"{val_plural:.4f} ({'+' if d_plural>=0 else ''}{d_plural:.4f})", color="#fff", bg=SECOND)

# ============== ESTATÍSTICAS ==============
with tabs[1]:
    style_title("Estatísticas")

    # — CIDs do ouro
    ouro = stats.get("CIDs_ouro", {})
    cols = st.columns(4)
    cols[0].metric("Média CIDs (ouro)", f"{ouro.get('media', 0):.2f}")
    cols[1].metric("Mediana (ouro)", f"{ouro.get('mediana', 0):.2f}")
    cols[2].metric("Desvio (ouro)", f"{ouro.get('desvio', 0):.2f}")
    if "p90" in ouro:
        cols[3].metric("P90 (ouro)", f"{ouro.get('p90', 0):.2f}")

    st.divider()

    # — Distribuição Top-10 3-char do ouro (se existir)
    dist3 = stats.get("Distribuicao_3char", {})
    if dist3:
        style_subtitle("Top-10 categorias 3-char do ouro")
        df_top = pd.DataFrame({"Categoria": list(dist3.keys()), "Contagem": list(dist3.values())})
        df_top = df_top.sort_values("Contagem", ascending=True)

        try:
            import altair as alt
            chart2 = (
                alt.Chart(df_top)
                .mark_bar(color=PRIMARY)
                .encode(
                    x=alt.X("Contagem:Q"),
                    y=alt.Y("Categoria:N", sort=None),
                    tooltip=["Categoria", "Contagem"]
                )
                .properties(height=320)
            )
            st.altair_chart(chart2, use_container_width=True)
        except Exception:
            st.dataframe(df_top.sort_values("Contagem", ascending=False), use_container_width=True)

    st.divider()

    # — Estatísticas por modelo (médias, etc.)
    pred_por_modelo = stats.get("Predicoes_por_modelo", {})
    if pred_por_modelo:
        style_subtitle("Predições por modelo (estatísticas)")
        rows = []
        for modelo, d in pred_por_modelo.items():
            modelo_norm = (
                str(modelo)
                .replace("\u2011", "-")
                .replace("\u00A0", " ")
                .strip()
            )
            rows.append({
                "Modelo": modelo_norm,
                "Média (len preds)": d.get("media", 0),
                "Mediana": d.get("mediana", 0),
                "Desvio": d.get("desvio", 0),
                "P10": d.get("p10", None),
                "P90": d.get("p90", None),
                "Únicos preditos": d.get("n_unicos_preditos", None),
                "Taxa repetição média": d.get("taxa_repeticao_media", None),
            })
        df_stats_modelo = pd.DataFrame(rows).sort_values("Modelo")
        st.dataframe(df_stats_modelo, use_container_width=True, hide_index=True)

    st.divider()

    # — Cobertura agregada (se você adicionou no stats.json)
    cobertura = stats.get("Cobertura", {})
    if cobertura:
        style_subtitle("Cobertura (ouro)")
        c1, c2 = st.columns(2)
        c1.metric("Códigos folha distintos", f"{cobertura.get('Codigos_folha_distintos', 0)}")
        c2.metric("Categorias 3-char distintas", f"{cobertura.get('Categorias_3char_distintas', 0)}")

        top10 = cobertura.get("Top_10_categorias", {})
        if top10:
            df_cov = pd.DataFrame({"Categoria": list(top10.keys()), "Contagem": list(top10.values())})
            df_cov = df_cov.sort_values("Contagem", ascending=True)
            try:
                import altair as alt
                chart3 = (
                    alt.Chart(df_cov)
                    .mark_bar(color=SECOND)
                    .encode(
                        x=alt.X("Contagem:Q"),
                        y=alt.Y("Categoria:N", sort=None),
                        tooltip=["Categoria", "Contagem"]
                    )
                    .properties(height=320)
                )
                st.altair_chart(chart3, use_container_width=True)
            except Exception:
                st.dataframe(df_cov.sort_values("Contagem", ascending=False), use_container_width=True)

    # — Rodapé/meta (se existir)
    meta = stats.get("meta", {})
    if meta:
        st.caption(
            f"n_amostras={meta.get('n_amostras','?')} • "
            f"export={meta.get('data_export','?')} • "
            f"versões={meta.get('versoes',{})}"
        )
