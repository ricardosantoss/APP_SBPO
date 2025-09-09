# -*- coding: utf-8 -*-
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
    layout="wide"
)

# Paleta (cores bem distintas por MÉTRICA)
COLOR_PREC = "#F59E0B"   # Precisão  (laranja)
COLOR_REC  = "#10B981"   # Recall    (verde)
COLOR_F1   = "#3B82F6"   # F1        (azul)
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
def pretty_model_name(raw: str) -> str:
    """
    Converte rótulos do CSV para os nomes desejados para exibição:
      GPT-4o • Deep-Seek V-3 • Sabia 3.1 • Pluralidade • Borda • Gemini 1.5 Flash • GPT 4o-Mini
    """
    m = _norm(raw).replace("\u2011", "-")  # normaliza
    # Agregados primeiro
    if "borda" in m: return "Borda"
    if "plural" in m: return "Pluralidade"
    # Modelos
    if "4o-mini" in m or "gpt-4o-mini" in m or "gpt 4o-mini" in m:
        return "GPT 4o-Mini"
    if "gpt-4o" in m or "gpt 4o" in m:
        return "GPT-4o"
    if "deep" in m and "seek" in m:
        return "Deep-Seek V-3"
    if "sabi" in m:   # cobre "sabia" e "sabiá"
        return "Sabia 3.1"
    if "gemini" in m and "flash" in m:
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

# ================== ABAS ==================
tabs = st.tabs(["🗂️ Dashboard", "📊 Estatísticas"])

# ================== DASHBOARD ==================
with tabs[0]:
    style_title("Dashboard")

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

    # Resolve nomes das colunas de métricas (para a agregação escolhida)
    metric_cols = resolve_metric_columns(view, agg_choice)  # {'precision','recall','f1'}

    # Ordena por F1 para ficar confortável
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

    # ================== GRÁFICO AGRUPADO ==================
    # Prepara dados no formato tidy: 3 linhas por modelo
    plot_rows = []
    for _, r in view.iterrows():
        modelo = pretty_model_name(str(r["Modelo"]))
        plot_rows.append({"Modelo": modelo, "Métrica": "Precisão", "Valor": float(r[metric_cols["precision"]])})
        plot_rows.append({"Modelo": modelo, "Métrica": "Recall",   "Valor": float(r[metric_cols["recall"]])})
        plot_rows.append({"Modelo": modelo, "Métrica": "F1",       "Valor": float(r[metric_cols["f1"]])})
    df_plot = pd.DataFrame(plot_rows)

    # Desduplicação pós-pretty: mantém MAIOR valor por (Modelo, Métrica)
    df_plot = (
        df_plot
        .groupby(["Modelo", "Métrica"], as_index=False, sort=False)["Valor"]
        .max()
    )

    # Destacar agregados
    HIGHLIGHT = {"Borda", "Pluralidade"}
    df_plot["Agregado"] = df_plot["Modelo"].isin(HIGHLIGHT)

    # Domínio DINÂMICO no eixo X:
    preferred = [
        "GPT-4o",
        "Deep-Seek V-3",
        "Sabia 3.1",
        "Pluralidade",
        "Borda",
        "Gemini 1.5 Flash",
        "GPT 4o-Mini",
    ]
    present = [m for m in preferred if m in df_plot["Modelo"].unique().tolist()]
    extras  = sorted([m for m in df_plot["Modelo"].unique().tolist() if m not in present])
    order_domain = present + extras

    # --- Barras agrupadas com destaque ---
    try:
        import altair as alt

        METRIC_COLORS = {"Precisão": COLOR_PREC, "Recall": COLOR_REC, "F1": COLOR_F1}

        # Base: todos os modelos com opacidade menor
        base = (
            alt.Chart(df_plot)
            .mark_bar()
            .encode(
                x=alt.X(
                    "Modelo:N",
                    scale=alt.Scale(domain=order_domain),
                    axis=alt.Axis(title=None, labelAngle=0, labelFontSize=14, labelColor="#222")
                ),
                y=alt.Y("Valor:Q", title=f"{agg_choice} (valor)"),
                color=alt.Color(
                    "Métrica:N",
                    scale=alt.Scale(
                        domain=list(METRIC_COLORS.keys()),
                        range=[METRIC_COLORS[m] for m in METRIC_COLORS]
                    ),
                    legend=alt.Legend(title="Métrica")
                ),
                xOffset="Métrica:N",
                opacity=alt.condition(alt.datum.Agregado == True, alt.value(1.0), alt.value(0.45)),
                tooltip=["Modelo", "Métrica", alt.Tooltip("Valor:Q", format=".4f")]
            )
            .properties(height=460)
        )

        # Overlay: só Borda/Pluralidade com contorno
        overlay = (
            alt.Chart(df_plot[df_plot["Agregado"]])
            .mark_bar(stroke="#111", strokeWidth=2)
            .encode(
                x=alt.X("Modelo:N", scale=alt.Scale(domain=order_domain), axis=alt.Axis(title=None)),
                y=alt.Y("Valor:Q"),
                color=alt.Color(
                    "Métrica:N",
                    scale=alt.Scale(
                        domain=list(METRIC_COLORS.keys()),
                        range=[METRIC_COLORS[m] for m in METRIC_COLORS]
                    ),
                    legend=None
                ),
                xOffset="Métrica:N",
            )
        )

        chart = (base + overlay)
        st.altair_chart(chart, use_container_width=True)

    except Exception:
        st.info("Para o gráfico, inclua 'altair' no requirements.txt. Exibindo tabela como fallback.")
        st.dataframe(
            df_plot.pivot(index="Modelo", columns="Métrica", values="Valor").reindex(order_domain),
            use_container_width=True
        )

    # Δ vs. melhor modelo individual (pela F1 da agregação escolhida)
    info = best_individual_delta(view, metric_cols["f1"])
    if info:
        best_name, best_val, val_borda, val_plural, d_borda, d_plural = info
        style_subtitle("Δ vs. melhor modelo individual (pela F1)")
        chip("Melhor Individual", f"{pretty_model_name(best_name)} ({best_val:.4f})", bg=PRIMARY)
        if val_borda is not None:
            chip("Borda", f"{val_borda:.4f} ({'+' if d_borda>=0 else ''}{d_borda:.4f})", bg=PRIMARY)
        if val_plural is not None:
            chip("Pluralidade", f"{val_plural:.4f} ({'+' if d_plural>=0 else ''}{d_plural:.4f})", bg=SECOND)

# ================== ESTATÍSTICAS ==================
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

    # — Top-10 3-char do ouro (se existir)
    dist3 = stats.get("Distribuicao_3char", {})
    if dist3:
        style_subtitle("Top-10 categorias 3-char do ouro")
        df_top = pd.DataFrame({"Categoria": list(dist3.keys()), "Contagem": list(dist3.values())})
        df_top = df_top.sort_values("Contagem", ascending=True)

        try:
            import altair as alt
            chart2 = (
                alt.Chart(df_top)
                .mark_bar(color=SECOND)
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

    # — Estatísticas por modelo (se você enriqueceu o stats.json)
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
                "Modelo": pretty_model_name(modelo_norm),
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
cobertura_pm = stats.get("Cobertura_por_modelo", {})
    if not cobertura_pm:
        st.info("Para ver cobertura por modelo (full e 3-char, %Top10 e %Top20), gere 'Cobertura_por_modelo' no stats.json.")
    else:
        style_subtitle("Cobertura por modelo (full e 3-char)")

        # Mapear chaves do JSON -> nomes bonitos
        items = []
        for raw_name, d in cobertura_pm.items():
            pretty = pretty_model_name(raw_name)
            items.append((pretty, raw_name))
        # ordenar alfabeticamente pelos nomes bonitos
        items = sorted(items, key=lambda x: x[0])

        col_sel, _ = st.columns([2,1])
        with col_sel:
            pretty_names = [p for p, _ in items]
            choice = st.selectbox("Modelo", pretty_names, index=0)

        # recuperar o dicionário do modelo escolhido
        raw_key = dict(items)[choice]
        d = cobertura_pm.get(raw_key, {})

        # Extrair métricas com defaults seguros
        n_leaf = int(d.get("Codigos_folha_distintos", 0))
        n_3ch  = int(d.get("Categorias_3char_distintas", 0))

        cov_full   = d.get("Cobertura_full", {}) or {}
        cov_3char  = d.get("Cobertura_3char", {}) or {}

        pct10_full = float(cov_full.get("%Top10", 0.0))
        pct20_full = float(cov_full.get("%Top20", 0.0))
        pct10_3ch  = float(cov_3char.get("%Top10", 0.0))
        pct20_3ch  = float(cov_3char.get("%Top20", 0.0))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Únicos preditos (full)", f"{n_leaf}")
        c2.metric("Únicos preditos (3-char)", f"{n_3ch}")
        c3.metric("%Top-10 (full)", f"{pct10_full:.1f}%")
        c4.metric("%Top-20 (full)", f"{pct20_full:.1f}%")

        c5, c6 = st.columns(2)
        with c5:
            st.metric("%Top-10 (3-char)", f"{pct10_3ch:.1f}%")
        with c6:
            st.metric("%Top-20 (3-char)", f"{pct20_3ch:.1f}%")

        st.caption("“%Top-10/20” = fração das predições coberta pelos 10/20 códigos mais frequentes.")

        st.divider()

        # (Opcional) Mostrar Top-10 como barras para inspeção rápida
        import altair as alt
        top10_full  = cov_full.get("Top10", {}) or cov_full.get("Top_10", {}) or cov_full.get("Top-10", {}) or cov_full.get("Top10", {})
        top10_3char = cov_3char.get("Top10", {}) or cov_3char.get("Top_10", {}) or cov_3char.get("Top-10", {}) or cov_3char.get("Top10", {})

        colL, colR = st.columns(2)
        if top10_full:
            df_f = pd.DataFrame({"Código (full)": list(top10_full.keys()), "Contagem": list(top10_full.values())})
            df_f = df_f.sort_values("Contagem", ascending=True)
            chart_f = (
                alt.Chart(df_f)
                .mark_bar(color="#3B82F6")
                .encode(
                    x=alt.X("Contagem:Q"),
                    y=alt.Y("Código (full):N", sort=None),
                    tooltip=["Código (full)", "Contagem"]
                )
                .properties(height=320, title=f"Top-10 (full) — {choice}")
            )
            colL.altair_chart(chart_f, use_container_width=True)
        if top10_3char:
            df_c = pd.DataFrame({"Categoria (3-char)": list(top10_3char.keys()), "Contagem": list(top10_3char.values())})
            df_c = df_c.sort_values("Contagem", ascending=True)
            chart_c = (
                alt.Chart(df_c)
                .mark_bar(color="#10B981")
                .encode(
                    x=alt.X("Contagem:Q"),
                    y=alt.Y("Categoria (3-char):N", sort=None),
                    tooltip=["Categoria (3-char)", "Contagem"]
                )
                .properties(height=320, title=f"Top-10 (3-char) — {choice}")
            )
            colR.altair_chart(chart_c, use_container_width=True)

        st.divider()

        # Tabela-resumo com TODOS os modelos (para comparar lado a lado)
        rows = []
        for raw_name, dmodel in cobertura_pm.items():
            pretty = pretty_model_name(raw_name)
            r = {
                "Modelo": pretty,
                "Únicos (full)": int(dmodel.get("Codigos_folha_distintos", 0)),
                "Únicos (3-char)": int(dmodel.get("Categorias_3char_distintas", 0)),
                "%Top-10 (full)": float((dmodel.get("Cobertura_full", {}) or {}).get("%Top10", 0.0)),
                "%Top-20 (full)": float((dmodel.get("Cobertura_full", {}) or {}).get("%Top20", 0.0)),
                "%Top-10 (3-char)": float((dmodel.get("Cobertura_3char", {}) or {}).get("%Top10", 0.0)),
                "%Top-20 (3-char)": float((dmodel.get("Cobertura_3char", {}) or {}).get("%Top20", 0.0)),
            }
            rows.append(r)
        df_resumo = pd.DataFrame(rows).sort_values("Modelo")
        st.dataframe(df_resumo, use_container_width=True, hide_index=True)
    # — Rodapé/meta (se existir)
    meta = stats.get("meta", {})
    if meta:
        st.caption(
            f"n_amostras={meta.get('n_amostras','?')} • "
            f"export={meta.get('data_export','?')} • "
            f"versões={meta.get('versoes',{})}"
        )
