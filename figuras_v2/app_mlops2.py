# =============================================================================
#  SISTEMA MLOPS · BRECHA DIGITAL Y EQUIPAMIENTO DEL HOGAR · CPV 2024
#  Aplicación web local.  Ejecutar:   streamlit run app_mlops.py
# =============================================================================
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import joblib

st.set_page_config(page_title="MLOps · Brecha Digital Bolivia",
                   page_icon="◆", layout="wide",
                   initial_sidebar_state="expanded")

DIR = "mlops"
AZUL, AZUL2, CIAN = "#1E3A8A", "#3B82F6", "#06B6D4"
LILA, MENTA, CORAL = "#6366F1", "#2DD4BF", "#FB7185"
AMBAR, TINTA, GRIS = "#FBBF24", "#0F172A", "#64748B"
SEC = [AZUL2, LILA, MENTA, CORAL, AMBAR, CIAN, "#818CF8", "#5EEAD4"]
ESCALA = [[0, "#DBEAFE"], [.35, "#60A5FA"], [.7, "#3B82F6"], [1, "#1E3A8A"]]
ESCALA_R = [[0, "#CCFBF1"], [.5, "#FBBF24"], [1, "#BE123C"]]

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;500;700&family=Inter:wght@300;400;600&display=swap');
.stApp {{ background: linear-gradient(170deg,#F0F6FF 0%,#E8F0FE 40%,#EEF2FF 100%); }}
html, body, [class*="css"] {{ font-family:'Inter',sans-serif; color:{TINTA}; }}
h1,h2,h3,h4 {{ font-family:'Outfit',sans-serif; color:{TINTA}; letter-spacing:-.5px; }}
section[data-testid="stSidebar"] {{ background:linear-gradient(185deg,#0F172A 0%,#1E3A8A 60%,#312E81 100%); }}
section[data-testid="stSidebar"] * {{ color:#E0E7FF !important; }}
.hero {{ background:linear-gradient(115deg,{AZUL} 0%,{AZUL2} 45%,{CIAN} 100%);
  padding:2.2rem 2.5rem; border-radius:22px; color:#fff;
  box-shadow:0 18px 45px rgba(30,58,138,.32); margin-bottom:1.4rem; }}
.hero h1 {{ color:#fff !important; font-size:2.3rem; margin:0 0 .4rem 0; }}
.hero p {{ color:rgba(255,255,255,.94); margin:0; font-size:1.02rem; }}
.tag {{ display:inline-block; background:rgba(255,255,255,.22); padding:.3rem .95rem;
  border-radius:30px; font-size:.75rem; letter-spacing:1.5px; text-transform:uppercase;
  margin-bottom:.8rem; }}
.kpi {{ background:#fff; border-radius:18px; padding:1.25rem 1.4rem;
  box-shadow:0 6px 22px rgba(15,23,42,.08); border-left:5px solid {AZUL2};
  transition:transform .18s ease, box-shadow .18s ease; height:100%; }}
.kpi:hover {{ transform:translateY(-4px); box-shadow:0 14px 32px rgba(30,58,138,.18); }}
.kpi .e {{ font-size:.73rem; color:{GRIS}; text-transform:uppercase; letter-spacing:1.1px; font-weight:600; }}
.kpi .v {{ font-family:'Outfit'; font-size:2rem; font-weight:700; color:{TINTA}; line-height:1.15; margin:.2rem 0; }}
.kpi .s {{ font-size:.82rem; color:{GRIS}; }}
.lila {{ border-left-color:{LILA} !important; }} .menta {{ border-left-color:{MENTA} !important; }}
.coral {{ border-left-color:{CORAL} !important; }} .ambar {{ border-left-color:{AMBAR} !important; }}
.nota {{ background:#fff; border-radius:16px; padding:1.1rem 1.35rem; border-left:5px solid {MENTA};
  box-shadow:0 4px 16px rgba(15,23,42,.06); margin:.8rem 0; font-size:.93rem; line-height:1.62; }}
.alerta {{ border-left-color:{AMBAR} !important; }} .dato {{ border-left-color:{LILA} !important; }}
div.stButton>button {{ background:linear-gradient(120deg,{AZUL2},{LILA});
  color:#fff; border:none; border-radius:12px; padding:.6rem 1.5rem; font-weight:600;
  box-shadow:0 6px 18px rgba(59,130,246,.35); transition:all .2s ease; }}
div.stButton>button:hover {{ transform:translateY(-2px); box-shadow:0 12px 26px rgba(99,102,241,.45); }}
.stTabs [data-baseweb="tab-list"] {{ gap:.4rem; }}
.stTabs [data-baseweb="tab"] {{ background:#fff; border-radius:12px 12px 0 0; padding:.55rem 1.15rem; font-weight:600; }}
.stTabs [aria-selected="true"] {{ background:linear-gradient(120deg,{AZUL2},{LILA}); color:#fff !important; }}
#MainMenu, footer {{ visibility:hidden; }}
</style>""", unsafe_allow_html=True)


def base(fig, alto=430, titulo=None):
    fig.update_layout(height=alto, title=titulo,
                      paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(255,255,255,.75)",
                      font=dict(family="Inter", size=12, color=TINTA),
                      title_font=dict(family="Outfit", size=16),
                      margin=dict(l=55, r=25, t=55 if titulo else 25, b=50),
                      legend=dict(bgcolor="rgba(255,255,255,.8)",
                                  bordercolor="#DBEAFE", borderwidth=1),
                      colorway=SEC)
    fig.update_xaxes(gridcolor="#E0E9F9"); fig.update_yaxes(gridcolor="#E0E9F9")
    return fig


def kpi(e, v, s="", c=""):
    return f'<div class="kpi {c}"><div class="e">{e}</div><div class="v">{v}</div><div class="s">{s}</div></div>'


@st.cache_data
def cargar():
    def js(n):
        p = os.path.join(DIR, n)
        return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None
    d = {'reg': js("registro.json"), 'pan': js("panorama.json"),
         'cur': js("curvas.json"), 'mat': js("matrices.json"),
         'imp': js("importancias.json"), 'dis': js("dispersion.json")}
    for k, f in [('mapa', "mapa_departamentos.csv"), ('der', "deriva.csv"),
                 ('mue', "muestra.csv")]:
        p = os.path.join(DIR, f)
        d[k] = pd.read_csv(p) if os.path.exists(p) else None
    return d


@st.cache_resource
def modelo(ruta):
    return joblib.load(ruta)


D = cargar()
if D['reg'] is None:
    st.error("Faltan los artefactos. Ejecuta primero  python J_preparar_mlops.py")
    st.stop()

REG, PAN = D['reg'], D['pan']
NOM = PAN['nombres']
ETIQ = {k: {int(a): b for a, b in v.items()}
        for k, v in PAN['etiquetas'].items()}
COLS, CATS, NUMS = PAN['columnas'], PAN['categoricas'], PAN['numericas']
VERS = list(REG['versiones'].keys())
ETQ = lambda c: NOM.get(c, c)

with st.sidebar:
    st.markdown("# ◆ MLOps CPV 2024")
    st.caption("Brecha digital y equipamiento del hogar en Bolivia")
    st.markdown("---")
    pagina = st.radio("Secciones", [
        "Inicio", "Panorama del dataset", "Mapa de Bolivia",
        "Clasificación", "Regresión", "Predicción en vivo",
        "Objetivos adicionales", "MLOps y versionado"],
        label_visibility="collapsed")
    st.markdown("---")
    VER = st.selectbox("Versión del sistema", VERS, index=len(VERS)-1)
    st.caption(f"Entrenamiento: {REG['versiones'][VER]['n_entrenamiento']:,}"
               f"  ·  Prueba: {REG['versiones'][VER]['n_prueba']:,}")
    st.markdown("---")
    st.caption(f"Generado {REG['generado']} · semilla {REG['semilla']}")

VD = REG['versiones'][VER]
MC = list(VD['clasificacion'].keys())
MR = list(VD['regresion'].keys())

# =============================================================================
if pagina == "Inicio":
    st.markdown(f"""<div class="hero"><div class="tag">Sistema MLOps · Censo 2024</div>
    <h1>Brecha digital y equipamiento del hogar</h1>
    <p>Modelos supervisados entrenados sobre {PAN['n_total']:,} viviendas particulares
    ocupadas de Bolivia, con versionado, control de deriva y predicción en vivo.</p></div>""",
                unsafe_allow_html=True)
    c = st.columns(5)
    c[0].markdown(kpi("Viviendas", f"{PAN['n_total']:,}", "dataset limpio"),
                  unsafe_allow_html=True)
    c[1].markdown(kpi("Con internet", f"{PAN['pct_internet']:.2f} %",
                      "oficial INE 76,30 %", "lila"), unsafe_allow_html=True)
    c[2].markdown(kpi("Equipamiento", f"{PAN['ieh_medio']:.2f}",
                      "bienes de 13", "menta"), unsafe_allow_html=True)
    c[3].markdown(kpi("Hogares rurales", f"{PAN['pct_rural']:.1f} %",
                      "del total", "ambar"), unsafe_allow_html=True)
    c[4].markdown(kpi("Carencia servicios", f"{PAN['pct_sin_serv']:.1f} %",
                      "agua, energía o baño", "coral"), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    i, d = st.columns([1.1, 1])
    with i:
        st.markdown("### El problema")
        st.markdown(f"""<div class="nota dato">
        Tres de cada cuatro viviendas bolivianas tienen internet, pero el promedio
        esconde una brecha territorial profunda. Este sistema estima, a partir de
        las características físicas de una vivienda, su probabilidad de acceso y
        su nivel de equipamiento, para identificar hogares en riesgo de exclusión.
        </div>""", unsafe_allow_html=True)
        st.markdown("### Los objetivos modelados")
        st.markdown(f"""<div class="nota">
        <b>internet</b> · clasificación binaria · ¿la vivienda tiene acceso?<br>
        <b>ieh</b> · regresión · ¿cuántos de 13 bienes posee el hogar?<br>
        <b>hacinamiento y servicios</b> · dos objetivos binarios adicionales
        </div>""", unsafe_allow_html=True)
    with d:
        mj = max(VD['clasificacion'].items(),
                 key=lambda x: x[1]['metricas']['ROC_AUC'])
        rj = max(VD['regresion'].items(),
                 key=lambda x: x[1]['metricas']['R2'])
        fig = go.Figure(go.Pie(
            labels=["Con internet", "Sin internet"],
            values=[PAN['pct_internet'], 100-PAN['pct_internet']], hole=.62,
            marker=dict(colors=[AZUL2, CORAL],
                        line=dict(color="white", width=3)),
            textinfo="label+percent", textfont=dict(size=14, family="Outfit")))
        fig.add_annotation(text=f"<b>{PAN['pct_internet']:.1f} %</b><br>conectados",
                           showarrow=False,
                           font=dict(size=19, family="Outfit", color=TINTA))
        st.plotly_chart(base(fig, 330, "Cobertura nacional"),
                        use_container_width=True)
        st.markdown(f"""<div class="nota alerta">
        <b>Mejores modelos en {VER}</b><br>
        Clasificación · {mj[0]} · AUC {mj[1]['metricas']['ROC_AUC']:.4f}<br>
        Regresión · {rj[0]} · R² {rj[1]['metricas']['R2']:.4f}
        </div>""", unsafe_allow_html=True)

# =============================================================================
elif pagina == "Panorama del dataset":
    st.markdown("# Panorama del dataset")
    st.markdown(f"""<div class="nota dato">Explora cualquiera de las
    {len(COLS)} variables del censo: su distribución y cómo se relaciona con
    los cuatro indicadores del proyecto.</div>""", unsafe_allow_html=True)
    t1, t2, t3 = st.tabs(["Explorador de variables", "Correlaciones",
                          "Vista previa"])
    with t1:
        c1, c2 = st.columns([1, 1])
        var = c1.selectbox("Variable", COLS, format_func=ETQ,
                           index=COLS.index('v06_piso') if 'v06_piso' in COLS else 0)
        ind = c2.selectbox("Indicador a cruzar",
                           ["internet", "ieh", "hacin", "serv"],
                           format_func=lambda k: {
                               'internet': "% con internet",
                               'ieh': "Bienes promedio",
                               'hacin': "% hacinamiento crítico",
                               'serv': "% carencia de servicios"}[k])
        d = PAN['distribuciones'][var]
        a, b = st.columns(2)
        if 'valores' in d:
            et = [ETIQ.get(var, {}).get(int(v), str(int(v)))
                  for v in d['valores']]
            fig = go.Figure(go.Bar(x=et, y=d['n'],
                                   marker=dict(color=d['n'], colorscale=ESCALA),
                                   hovertemplate="%{x}<br>%{y:,} viviendas<extra></extra>"))
            fig.update_yaxes(title="Viviendas")
            a.plotly_chart(base(fig, 420, f"Distribución · {ETQ(var)}"),
                           use_container_width=True)
            med = {'internet': PAN['pct_internet'], 'ieh': PAN['ieh_medio'],
                   'hacin': PAN['pct_hacin'], 'serv': PAN['pct_sin_serv']}[ind]
            col = [CORAL if (v < med if ind in ('internet', 'ieh') else v > med)
                   else MENTA for v in d[ind]]
            fig = go.Figure(go.Bar(x=et, y=d[ind], marker_color=col,
                                   text=[f"{v:.1f}" for v in d[ind]],
                                   textposition="outside"))
            fig.add_hline(y=med, line_dash="dash", line_color=AZUL,
                          annotation_text=f"Media {med:.2f}")
            b.plotly_chart(base(fig, 420, f"Indicador según {ETQ(var)}"),
                           use_container_width=True)
        else:
            ctr = [(d['bins'][i]+d['bins'][i+1])/2 for i in range(len(d['hist']))]
            fig = go.Figure(go.Bar(x=ctr, y=d['hist'],
                                   marker=dict(color=d['hist'], colorscale=ESCALA)))
            a.plotly_chart(base(fig, 420, f"Histograma · {ETQ(var)}"),
                           use_container_width=True)
            b.info("Variable continua: sin desglose por categoría.")
    with t2:
        m = np.array(PAN['correlacion']['matriz'])
        vs = [ETQ(v) for v in PAN['correlacion']['vars']]
        fig = go.Figure(go.Heatmap(z=m, x=vs, y=vs, colorscale="RdBu",
                                   zmid=0, zmin=-1, zmax=1,
                                   text=np.round(m, 2),
                                   texttemplate="%{text}",
                                   textfont=dict(size=9)))
        st.plotly_chart(base(fig, 620, "Matriz de correlación"),
                        use_container_width=True)
    with t3:
        if D['mue'] is not None:
            st.dataframe(D['mue'].head(st.slider("Filas", 10, 200, 40)),
                         use_container_width=True, height=460)

# =============================================================================
elif pagina == "Mapa de Bolivia":
    st.markdown("# Mapa territorial")
    mp = D['mapa'].copy()
    ops = {'sin_internet': "% SIN internet", 'internet': "% con internet",
           'ieh': "Bienes promedio (0-13)",
           'hacin_critico': "% hacinamiento crítico",
           'sin_servicios': "% carencia de servicios",
           'rural': "% población rural"}
    c1, c2 = st.columns([1.4, 1])
    met = c1.selectbox("Indicador a mapear", list(ops), format_func=lambda k: ops[k])
    modo = c2.radio("Vista", ["Burbujas", "Calor"], horizontal=True)
    peor = met in ('sin_internet', 'hacin_critico', 'sin_servicios', 'rural')
    esc = ESCALA_R if peor else ESCALA
    k = st.columns(4)
    top = mp.loc[mp[met].idxmax()]; bot = mp.loc[mp[met].idxmin()]
    k[0].markdown(kpi("Valor más alto", f"{top[met]:.2f}",
                      str(top['departamento']), "coral" if peor else "menta"),
                  unsafe_allow_html=True)
    k[1].markdown(kpi("Valor más bajo", f"{bot[met]:.2f}",
                      str(bot['departamento']), "menta" if peor else "coral"),
                  unsafe_allow_html=True)
    k[2].markdown(kpi("Brecha", f"{top[met]-bot[met]:.2f}",
                      "entre extremos", "lila"), unsafe_allow_html=True)
    k[3].markdown(kpi("Promedio", f"{mp[met].mean():.2f}",
                      "de los 9 departamentos", "ambar"), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    a, b = st.columns([1.25, 1])
    with a:
        if modo == "Burbujas":
            fig = go.Figure(go.Scattergeo(
                lat=mp['lat'], lon=mp['lon'], text=mp['departamento'],
                mode="markers+text", textposition="top center",
                textfont=dict(size=11, family="Outfit", color=TINTA),
                marker=dict(size=mp['viviendas']/mp['viviendas'].max()*55+16,
                            color=mp[met], colorscale=esc, showscale=True,
                            line=dict(width=2.5, color="white"),
                            colorbar=dict(title=ops[met], thickness=14)),
                customdata=np.stack([mp['viviendas'], mp['internet'],
                                     mp['ieh'], mp['rural']], axis=-1),
                hovertemplate="<b>%{text}</b><br>" + ops[met] +
                              ": %{marker.color:.2f}<br>Viviendas: %{customdata[0]:,}"
                              "<br>Internet: %{customdata[1]:.1f} %"
                              "<br>Bienes: %{customdata[2]:.2f}"
                              "<br>Rural: %{customdata[3]:.1f} %<extra></extra>"))
        else:
            fig = go.Figure(go.Densmapbox(
                lat=mp['lat'], lon=mp['lon'], z=mp[met], radius=75,
                colorscale=esc, showscale=True,
                hovertext=mp['departamento'],
                colorbar=dict(title=ops[met], thickness=14)))
            fig.update_layout(mapbox_style="carto-positron",
                              mapbox=dict(center=dict(lat=-16.6, lon=-64.8),
                                          zoom=4.1))
        if modo == "Burbujas":
            fig.update_geos(scope="south america", showcountries=True,
                            countrycolor="#94A3B8", showland=True,
                            landcolor="#F8FAFF", showocean=True,
                            oceancolor="#E0F2FE", lataxis_range=[-23.5, -9],
                            lonaxis_range=[-70, -57])
        st.plotly_chart(base(fig, 560, f"Bolivia · {ops[met]}"),
                        use_container_width=True)
    with b:
        s = mp.sort_values(met)
        fig = go.Figure(go.Bar(
            x=s[met], y=s['departamento'], orientation='h',
            marker=dict(color=s[met], colorscale=esc),
            text=[f"{v:.2f}" for v in s[met]], textposition="outside"))
        fig.add_vline(x=mp[met].mean(), line_dash="dash", line_color=AZUL,
                      annotation_text="Media")
        st.plotly_chart(base(fig, 560, "Ranking departamental"),
                        use_container_width=True)
    st.markdown(f"""<div class="nota alerta"><b>Nota metodológica.</b>
    El mapa usa los valores <b>observados</b> en el censo, no las predicciones.
    Los modelos están calibrados con <code>class_weight='balanced'</code>, lo
    que los hace excelentes para <b>ordenar por riesgo</b> pero desplaza sus
    probabilidades: no deben leerse como proporciones poblacionales.
    </div>""", unsafe_allow_html=True)
    with st.expander("Ver tabla completa"):
        st.dataframe(mp.drop(columns=['lat', 'lon']), use_container_width=True)

# =============================================================================
elif pagina == "Clasificación":
    st.markdown("# Clasificación · ¿la vivienda tiene internet?")
    sel = st.radio("Modelo", MC, horizontal=True)
    m = VD['clasificacion'][sel]['metricas']
    c = st.columns(6)
    for col, (k_, v_, cl) in zip(c, [
            ("ROC AUC", m['ROC_AUC'], ""), ("PR AUC", m['PR_AUC'], "lila"),
            ("Exactitud", m['Accuracy'], "menta"),
            ("Precisión", m['Precision'], "ambar"),
            ("Recall", m['Recall'], ""),
            ("Recall SIN internet", m['Recall_min'], "coral")]):
        col.markdown(kpi(k_, f"{v_:.4f}", "", cl), unsafe_allow_html=True)
    st.markdown(f"""<div class="nota coral" style="border-left-color:{CORAL}">
    <b>El recall de la clase SIN internet ({m['Recall_min']:.4f}) es la métrica
    que manda.</b> Mide qué proporción de hogares desconectados detecta el
    modelo: son exactamente la población que el proyecto busca identificar.
    </div>""", unsafe_allow_html=True)
    a, b = st.columns(2)
    with a:
        fig = go.Figure()
        for i, n_ in enumerate(MC):
            cu = D['cur'].get(f"{VER}|clf|{n_}")
            if cu:
                fig.add_trace(go.Scatter(
                    x=cu['fpr'], y=cu['tpr'], name=n_,
                    line=dict(width=4 if n_ == sel else 1.8,
                              color=SEC[i % len(SEC)]),
                    opacity=1 if n_ == sel else .45))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Azar",
                                 line=dict(dash="dash", color=GRIS)))
        fig.update_xaxes(title="Tasa de falsos positivos")
        fig.update_yaxes(title="Tasa de verdaderos positivos")
        st.plotly_chart(base(fig, 430, "Curvas ROC · el seleccionado resalta"),
                        use_container_width=True)
    with b:
        fig = go.Figure()
        for i, n_ in enumerate(MC):
            cu = D['cur'].get(f"{VER}|clf|{n_}")
            if cu:
                fig.add_trace(go.Scatter(
                    x=cu['rec'], y=cu['prec'], name=n_,
                    line=dict(width=4 if n_ == sel else 1.8,
                              color=SEC[i % len(SEC)]),
                    opacity=1 if n_ == sel else .45))
        fig.update_xaxes(title="Recall"); fig.update_yaxes(title="Precision")
        st.plotly_chart(base(fig, 430, "Curvas Precision-Recall"),
                        use_container_width=True)
    a, b = st.columns([1, 1.3])
    with a:
        mc = np.array(D['mat'][f"{VER}|clf|{sel}"])
        fig = go.Figure(go.Heatmap(
            z=mc, x=["Predice: sin", "Predice: con"],
            y=["Real: sin", "Real: con"], colorscale=ESCALA,
            text=[[f"{v:,}" for v in r] for r in mc],
            texttemplate="%{text}", textfont=dict(size=20, family="Outfit"),
            showscale=False))
        st.plotly_chart(base(fig, 380, f"Matriz de confusión · {sel}"),
                        use_container_width=True)
    with b:
        im = D['imp'].get(f"{VER}|clf|{sel}", {})
        if im:
            s = pd.Series(im).sort_values().tail(12)
            fig = go.Figure(go.Bar(x=s.values, y=[ETQ(i) for i in s.index],
                                   orientation='h',
                                   marker=dict(color=s.values, colorscale=ESCALA)))
            st.plotly_chart(base(fig, 380, "Variables más influyentes"),
                            use_container_width=True)
    tb = pd.DataFrame({n_: VD['clasificacion'][n_]['metricas'] for n_ in MC}).T
    tb['segundos'] = [VD['clasificacion'][n_]['segundos'] for n_ in MC]
    st.markdown("### Comparación de todos los modelos")
    st.dataframe(tb.style.format("{:.4f}").background_gradient(
        cmap="Blues", subset=['ROC_AUC', 'PR_AUC', 'Recall_min']),
        use_container_width=True)

# =============================================================================
elif pagina == "Regresión":
    st.markdown("# Regresión · índice de equipamiento del hogar")
    sel = st.radio("Modelo", MR, horizontal=True)
    m = VD['regresion'][sel]['metricas']
    c = st.columns(5)
    for col, (k_, v_, s_, cl) in zip(c, [
            ("R²", f"{m['R2']:.4f}", "variación explicada", "menta"),
            ("RMSE", f"{m['RMSE']:.4f}", "bienes de 13", ""),
            ("MAE", f"{m['MAE']:.4f}", "error absoluto", "lila"),
            ("MSE", f"{m['MSE']:.4f}", "error cuadrático", "ambar"),
            ("MAPE", f"{m['MAPE']:.2f} %", "excluye ieh = 0", "coral")]):
        col.markdown(kpi(k_, v_, s_, cl), unsafe_allow_html=True)
    a, b = st.columns(2)
    with a:
        dd = D['dis'].get(f"{VER}|reg|{sel}")
        if dd:
            r = np.array(dd['real'], dtype=float)
            p = np.array(dd['pred'], dtype=float)
            j = np.random.default_rng(42).uniform(-.28, .28, len(r))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=r+j, y=p, mode="markers", name="Hogares",
                                     marker=dict(size=5, color=AZUL2, opacity=.3)))
            fig.add_trace(go.Scatter(x=[0, 13], y=[0, 13], name="Perfecta",
                                     line=dict(dash="dash", color=CORAL, width=3)))
            fig.update_xaxes(title="Bienes reales")
            fig.update_yaxes(title="Bienes predichos")
            st.plotly_chart(base(fig, 430, "Real frente a predicho"),
                            use_container_width=True)
    with b:
        if dd:
            res = r - p
            fig = go.Figure(go.Histogram(x=res, nbinsx=45,
                                         marker=dict(color=MENTA)))
            fig.add_vline(x=0, line_color=TINTA, line_width=2)
            fig.update_xaxes(title="Residuo (real − predicho)")
            st.plotly_chart(base(fig, 430,
                                 f"Errores · media {res.mean():+.3f}"),
                            use_container_width=True)
    a, b = st.columns([1.2, 1])
    with a:
        im = D['imp'].get(f"{VER}|reg|{sel}", {})
        if im:
            s = pd.Series(im).sort_values().tail(12)
            fig = go.Figure(go.Bar(x=s.values, y=[ETQ(i) for i in s.index],
                                   orientation='h',
                                   marker=dict(color=s.values,
                                               colorscale=[[0, "#E0E7FF"],
                                                           [1, LILA]])))
            st.plotly_chart(base(fig, 400, "Variables más influyentes"),
                            use_container_width=True)
    with b:
        s = pd.Series({n_: VD['regresion'][n_]['metricas']['R2']
                       for n_ in MR}).sort_values()
        fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation='h',
                               marker_color=[MENTA if i == sel else AZUL2
                                             for i in s.index],
                               text=[f"{v:.4f}" for v in s.values],
                               textposition="outside"))
        fig.update_xaxes(title="R²")
        st.plotly_chart(base(fig, 400, "R² por modelo"),
                        use_container_width=True)
    tb = pd.DataFrame({n_: VD['regresion'][n_]['metricas'] for n_ in MR}).T
    st.dataframe(tb.style.format("{:.4f}").background_gradient(
        cmap="Purples", subset=['R2']), use_container_width=True)

# =============================================================================
elif pagina == "Predicción en vivo":
    st.markdown("# Predicción en vivo")
    st.markdown("""<div class="nota dato">Describe una vivienda y los modelos
    estimarán su probabilidad de acceso a internet y su nivel de equipamiento.
    Puedes elegir qué algoritmo usa cada uno.</div>""", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    mc_sel = c1.selectbox("Modelo de clasificación", MC)
    mr_sel = c2.selectbox("Modelo de regresión", MR)
    VIS = ['urbrur', 'v06_piso', 'v03_pared', 'v10_combus', 'v11_basura',
           'v09_energia', 'v07_aguapro', 'v15_servsan', 'v16_desague',
           'tip_hog', 'tot_pers', 'v13_habitac', 'v14_dormit']
    VIS = [v for v in VIS if v in COLS]
    vals = dict(REG['valores_tipicos'])
    cols3 = st.columns(3)
    for i, v in enumerate(VIS):
        with cols3[i % 3]:
            if v in ETIQ:
                et = ETIQ[v]
                ks = sorted(et)
                dflt = ks.index(int(vals.get(v, ks[0]))) if int(
                    vals.get(v, ks[0])) in ks else 0
                vals[v] = st.selectbox(ETQ(v), ks, index=dflt,
                                       format_func=lambda k, e=et: e[k],
                                       key=f"i{v}")
            else:
                lo, hi = (1, 15) if v == 'tot_pers' else (
                    (1, 8) if v == 'v13_habitac' else (0, 8))
                vals[v] = st.slider(ETQ(v), lo, hi,
                                    int(min(max(vals.get(v, lo), lo), hi)),
                                    key=f"i{v}")
    if st.button("◆  Estimar", use_container_width=True):
        f = dict(vals)
        if f.get('v15_servsan') == 3:
            f['v16_desague'] = 0
        f['hacinamiento'] = f['tot_pers'] / max(f.get('v14_dormit', 1), 1)
        f['pers_por_habitac'] = f['tot_pers'] / max(f.get('v13_habitac', 1), 1)
        f['hacin_critico'] = int(f['hacinamiento'] > 3)
        f['sin_servicios'] = int(f.get('v07_aguapro', 1) >= 5 or
                                 f.get('v09_energia', 1) == 5 or
                                 f.get('v15_servsan', 1) == 3)
        X = pd.DataFrame([{c: float(f.get(c, 0)) for c in COLS}])[COLS]
        rc = os.path.join(DIR, "modelos",
                          f"{VER}_clf_{mc_sel}.joblib".replace(" ", "_"))
        rr = os.path.join(DIR, "modelos",
                          f"{VER}_reg_{mr_sel}.joblib".replace(" ", "_"))
        p = float(modelo(rc).predict_proba(X)[0, 1])
        ieh = float(modelo(rr).predict(X)[0])
        niv, col = ("BAJO riesgo", MENTA) if p >= .70 else (
            ("Riesgo MEDIO", AMBAR) if p >= .45 else ("ALTO riesgo", CORAL))
        a, b, c3 = st.columns(3)
        a.markdown(kpi("Probabilidad de internet", f"{p*100:.1f} %", niv,
                       "menta" if p >= .7 else ("ambar" if p >= .45 else "coral")),
                   unsafe_allow_html=True)
        b.markdown(kpi("Bienes estimados", f"{ieh:.1f}",
                       f"de 13 · {'Bajo' if ieh<=3 else ('Medio' if ieh<=6 else 'Alto')}",
                       "lila"), unsafe_allow_html=True)
        c3.markdown(kpi("Personas por dormitorio", f"{f['hacinamiento']:.2f}",
                        "crítico" if f['hacin_critico'] else "normal",
                        "coral" if f['hacin_critico'] else ""),
                    unsafe_allow_html=True)
        g1, g2 = st.columns(2)
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=p*100,
            number={'suffix': " %", 'font': {'size': 44, 'family': 'Outfit'}},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': col, 'thickness': .75},
                   'steps': [{'range': [0, 45], 'color': "#FEE2E2"},
                             {'range': [45, 70], 'color': "#FEF3C7"},
                             {'range': [70, 100], 'color': "#CCFBF1"}],
                   'threshold': {'line': {'color': AZUL, 'width': 4},
                                 'value': PAN['pct_internet']}}))
        g1.plotly_chart(base(fig, 340, "Probabilidad de acceso a internet"),
                        use_container_width=True)
        ref = pd.Series({'Esta vivienda': ieh,
                         'Promedio nacional': PAN['ieh_medio']})
        fig = go.Figure(go.Bar(x=ref.values, y=ref.index, orientation='h',
                               marker_color=[col, GRIS],
                               text=[f"{v:.2f}" for v in ref.values],
                               textposition="outside"))
        fig.update_xaxes(range=[0, 13], title="Bienes de 13")
        g2.plotly_chart(base(fig, 340, "Equipamiento estimado"),
                        use_container_width=True)

# =============================================================================
elif pagina == "Objetivos adicionales":
    st.markdown("# Objetivos adicionales")
    st.markdown("""<div class="nota dato">El dataset permite modelar otras
    dimensiones de vulnerabilidad además de la conectividad.</div>""",
                unsafe_allow_html=True)
    ex = REG.get('extra', {})
    for obj, info in ex.items():
        st.markdown(f"### {ETQ(obj)}")
        st.caption(info['descripcion'])
        c = st.columns(5)
        c[0].markdown(kpi("Prevalencia", f"{info['prevalencia']:.2f} %",
                          "de los hogares", "coral"), unsafe_allow_html=True)
        for col, (k_, v_) in zip(c[1:], info['metricas'].items()):
            col.markdown(kpi(k_, f"{v_:.4f}"), unsafe_allow_html=True)
        a, b = st.columns([1, 1.25])
        cu = D['cur'].get(f"extra|{obj}")
        if cu:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=cu['fpr'], y=cu['tpr'], name="Modelo",
                                     line=dict(color=AZUL2, width=3),
                                     fill='tozeroy'))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Azar",
                                     line=dict(dash="dash", color=GRIS)))
            a.plotly_chart(base(fig, 330, "Curva ROC"),
                           use_container_width=True)
        s = pd.Series(info['importancias']).sort_values().tail(10)
        fig = go.Figure(go.Bar(x=s.values, y=[ETQ(i) for i in s.index],
                               orientation='h',
                               marker=dict(color=s.values, colorscale=ESCALA)))
        b.plotly_chart(base(fig, 330, "Variables más influyentes"),
                       use_container_width=True)
        st.markdown(f"""<div class="nota alerta"><b>Control de fuga.</b>
        Se excluyeron {len(info.get('excluidas', []))} variables que componen
        este indicador: <code>{', '.join(info.get('excluidas', []))}</code>.
        Sin esa exclusión el AUC sería 1,000 por construcción, no por
        aprendizaje.</div>""", unsafe_allow_html=True)
        st.markdown("---")

# =============================================================================
elif pagina == "MLOps y versionado":
    st.markdown("# MLOps · versionado y control de deriva")
    t1, t2, t3 = st.tabs(["Registro de versiones", "Comparación",
                          "Deriva geográfica"])
    with t1:
        st.markdown("""<div class="nota dato">Cada versión del sistema queda
        registrada con su tamaño de entrenamiento, sus algoritmos, sus métricas
        y sus modelos serializados. Cambiar la versión en la barra lateral
        recalcula toda la aplicación.</div>""", unsafe_allow_html=True)
        filas = []
        for v, d in REG['versiones'].items():
            for t, dd in [('Clasificación', 'clasificacion'),
                          ('Regresión', 'regresion')]:
                for n_, i_ in d[dd].items():
                    filas.append({'Versión': v, 'Tipo': t, 'Modelo': n_,
                                  'Entrenamiento': d['n_entrenamiento'],
                                  'Métrica': 'ROC_AUC' if dd == 'clasificacion' else 'R2',
                                  'Valor': i_['metricas']['ROC_AUC'] if dd == 'clasificacion'
                                  else i_['metricas']['R2'],
                                  'Segundos': i_['segundos']})
        reg = pd.DataFrame(filas)
        st.dataframe(reg.style.format({'Valor': "{:.4f}",
                                       'Segundos': "{:.1f}",
                                       'Entrenamiento': "{:,}"})
                     .background_gradient(cmap="Blues", subset=['Valor']),
                     use_container_width=True, height=430)
    with t2:
        a, b = st.columns(2)
        for col, (tipo, met, tit) in zip([a, b], [
                ('clasificacion', 'ROC_AUC', "Clasificación · AUC por versión"),
                ('regresion', 'R2', "Regresión · R² por versión")]):
            fig = go.Figure()
            for i, v in enumerate(VERS):
                d = REG['versiones'][v][tipo]
                fig.add_trace(go.Bar(name=f"{v} ({REG['versiones'][v]['n_entrenamiento']:,})",
                                     x=list(d), y=[d[k]['metricas'][met] for k in d],
                                     marker_color=SEC[i], text=[f"{d[k]['metricas'][met]:.4f}"
                                                                for k in d],
                                     textposition="outside"))
            fig.update_layout(barmode="group")
            vals = [REG['versiones'][v][tipo][k]['metricas'][met]
                    for v in VERS for k in REG['versiones'][v][tipo]]
            fig.update_yaxes(range=[min(vals)*.97, max(vals)*1.02], title=met)
            col.plotly_chart(base(fig, 440, tit), use_container_width=True)
        if len(VERS) >= 2:
            v1, v2 = VERS[0], VERS[-1]
            d1 = REG['versiones'][v1]['clasificacion']
            d2 = REG['versiones'][v2]['clasificacion']
            comunes = [k for k in d1 if k in d2]
            dif = {k: d2[k]['metricas']['ROC_AUC'] - d1[k]['metricas']['ROC_AUC']
                   for k in comunes}
            s = pd.Series(dif).sort_values()
            fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation='h',
                                   marker_color=[MENTA if v > 0 else CORAL
                                                 for v in s.values],
                                   text=[f"{v:+.4f}" for v in s.values],
                                   textposition="outside"))
            fig.add_vline(x=0, line_color=TINTA)
            fig.update_xaxes(title=f"Cambio de AUC de {v1} a {v2}")
            st.plotly_chart(base(fig, 380,
                                 f"Efecto de ampliar el entrenamiento"),
                            use_container_width=True)
    with t3:
        de = D['der']
        dr = REG.get('deriva', {})
        if de is not None:
            c = st.columns(3)
            c[0].markdown(kpi("AUC donde entrenó",
                              f"{dr.get('auc_entrenamiento', 0):.4f}",
                              ", ".join(dr.get('departamentos_entrenamiento', [])),
                              "menta"), unsafe_allow_html=True)
            c[1].markdown(kpi("AUC en departamentos no vistos",
                              f"{dr.get('auc_no_visto', 0):.4f}",
                              "generalización", "ambar"),
                          unsafe_allow_html=True)
            c[2].markdown(kpi("Caída", f"{dr.get('caida', 0):+.4f}",
                              "deriva geográfica", "coral"),
                          unsafe_allow_html=True)
            s = de.sort_values('auc')
            fig = go.Figure(go.Bar(
                x=s['auc'], y=s['departamento'], orientation='h',
                marker_color=[MENTA if g == 'Entrenamiento' else CORAL
                              for g in s['grupo']],
                text=[f"{v:.4f}" for v in s['auc']], textposition="outside"))
            fig.add_vline(x=dr.get('auc_entrenamiento', 0), line_dash="dash",
                          line_color=AZUL, annotation_text="Media entrenamiento")
            fig.update_xaxes(title="AUC", range=[max(0, s['auc'].min()-.06),
                                                 s['auc'].max()+.03])
            st.plotly_chart(base(fig, 460,
                                 "Verde: donde entrenó · Rojo: no visto"),
                            use_container_width=True)
            st.markdown(f"""<div class="nota alerta">
            <b>Qué significa.</b> El modelo se entrenó solo con
            {", ".join(dr.get('departamentos_entrenamiento', []))} y se evaluó
            en los seis departamentos restantes. La caída de
            <b>{dr.get('caida', 0):+.4f}</b> en AUC es <b>deriva geográfica</b>:
            el modelo no generaliza igual en todo el territorio. En producción
            esto obligaría a reentrenar con cobertura nacional o a entrenar
            modelos regionales.</div>""", unsafe_allow_html=True)
            st.dataframe(de, use_container_width=True)