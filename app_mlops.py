# =============================================================================
#  SISTEMA MLOPS · BRECHA DIGITAL Y EQUIPAMIENTO DEL HOGAR · CPV 2024
#  Universidad Mayor de San Andrés · Carrera de Informática
#  Ejecutar desde la terminal:   streamlit run app_mlops.py
# =============================================================================
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import joblib

st.set_page_config(page_title="MLOps · Brecha Digital Bolivia",
                   page_icon="◆", layout="wide",
                   initial_sidebar_state="expanded")

DIR = "mlops"
AZUL, AZUL2, CIAN = "#1E3A8A", "#3B82F6", "#06B6D4"
LILA, MENTA, CORAL = "#6366F1", "#2DD4BF", "#FB7185"
AMBAR, TINTA, GRIS = "#FBBF24", "#0F172A", "#64748B"
ORO = "#D97706"
SEC = [AZUL2, LILA, MENTA, CORAL, AMBAR, CIAN, "#818CF8", "#5EEAD4",
       "#F472B6", "#38BDF8"]
PALETAS = {
    "Océano": [[0, "#E0F2FE"], [.35, "#38BDF8"], [.7, "#2563EB"], [1, "#1E3A8A"]],
    "Aurora": [[0, "#EDE9FE"], [.35, "#A78BFA"], [.7, "#7C3AED"], [1, "#4C1D95"]],
    "Turquesa": [[0, "#CCFBF1"], [.4, "#2DD4BF"], [.75, "#0D9488"], [1, "#134E4A"]],
    "Atardecer": [[0, "#FEF3C7"], [.4, "#FBBF24"], [.75, "#F97316"], [1, "#9A3412"]],
}
ALERTA = [[0, "#CCFBF1"], [.5, "#FBBF24"], [1, "#BE123C"]]

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;500;700;800&family=Inter:wght@300;400;600&display=swap');
.stApp {{ background:
   radial-gradient(1200px 600px at 10% -10%, #DBEAFE 0%, transparent 55%),
   radial-gradient(900px 500px at 95% 5%, #E0E7FF 0%, transparent 50%),
   linear-gradient(170deg,#F2F7FF 0%,#EAF1FE 45%,#F1F0FE 100%); }}
html, body, [class*="css"] {{ font-family:'Inter',sans-serif; color:{TINTA}; }}
h1,h2,h3,h4 {{ font-family:'Outfit',sans-serif; color:{TINTA}; letter-spacing:-.5px; }}
section[data-testid="stSidebar"] {{ background:linear-gradient(185deg,#0B1220 0%,#15265C 55%,#2E1D6B 100%); }}
section[data-testid="stSidebar"] * {{ color:#E0E7FF !important; }}
.hero {{ background:linear-gradient(115deg,{AZUL} 0%,{AZUL2} 42%,{CIAN} 100%);
  padding:2.1rem 2.4rem; border-radius:24px; color:#fff; position:relative; overflow:hidden;
  box-shadow:0 20px 50px rgba(30,58,138,.33); margin-bottom:1.3rem; }}
.hero:after {{ content:""; position:absolute; right:-70px; top:-70px; width:260px; height:260px;
  background:radial-gradient(circle,rgba(255,255,255,.22),transparent 70%); border-radius:50%; }}
.hero h1 {{ color:#fff !important; font-size:2.3rem; margin:0 0 .4rem 0; font-weight:800; }}
.hero p {{ color:rgba(255,255,255,.95); margin:0; font-size:1.01rem; }}
.tag {{ display:inline-block; background:rgba(255,255,255,.22); padding:.3rem .95rem;
  border-radius:30px; font-size:.74rem; letter-spacing:1.6px; text-transform:uppercase; margin-bottom:.7rem; }}
.kpi {{ background:rgba(255,255,255,.93); border-radius:18px; padding:1.15rem 1.3rem;
  box-shadow:0 6px 24px rgba(15,23,42,.09); border-left:5px solid {AZUL2};
  transition:transform .2s ease, box-shadow .2s ease; height:100%; }}
.kpi:hover {{ transform:translateY(-5px) scale(1.01); box-shadow:0 16px 36px rgba(30,58,138,.2); }}
.kpi .e {{ font-size:.71rem; color:{GRIS}; text-transform:uppercase; letter-spacing:1.1px; font-weight:600; }}
.kpi .v {{ font-family:'Outfit'; font-size:1.9rem; font-weight:700; line-height:1.15; margin:.2rem 0;
  background:linear-gradient(120deg,{AZUL},{LILA}); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }}
.kpi .s {{ font-size:.8rem; color:{GRIS}; }}
.lila {{ border-left-color:{LILA} !important; }} .menta {{ border-left-color:{MENTA} !important; }}
.coral {{ border-left-color:{CORAL} !important; }} .ambar {{ border-left-color:{AMBAR} !important; }}
.nota {{ background:rgba(255,255,255,.94); border-radius:16px; padding:1rem 1.3rem;
  border-left:5px solid {MENTA}; box-shadow:0 4px 18px rgba(15,23,42,.07); margin:.8rem 0;
  font-size:.92rem; line-height:1.6; }}
.alerta {{ border-left-color:{AMBAR} !important; }} .dato {{ border-left-color:{LILA} !important; }}
.rojo {{ border-left-color:{CORAL} !important; }}
.gana {{ background:linear-gradient(120deg,#FFFBEB,#FEF9C3); border:2px solid {AMBAR};
  border-radius:18px; padding:1.15rem 1.4rem; box-shadow:0 8px 24px rgba(251,191,36,.28); height:100%; }}
.gana .c {{ font-size:.72rem; letter-spacing:1.4px; text-transform:uppercase;
  color:{ORO}; font-weight:700; }}
.gana .m {{ font-family:'Outfit'; font-size:1.65rem; font-weight:800; color:{AZUL};
  line-height:1.2; margin:.15rem 0; }}
.gana .d {{ font-size:.86rem; color:#78350F; }}
div.stButton>button {{ background:linear-gradient(120deg,{AZUL2},{LILA}); color:#fff; border:none;
  border-radius:12px; padding:.6rem 1.5rem; font-weight:600;
  box-shadow:0 8px 20px rgba(59,130,246,.38); transition:all .22s ease; }}
div.stButton>button:hover {{ transform:translateY(-3px) scale(1.02); box-shadow:0 14px 30px rgba(99,102,241,.5); }}
.stTabs [data-baseweb="tab-list"] {{ gap:.4rem; }}
.stTabs [data-baseweb="tab"] {{ background:rgba(255,255,255,.85); border-radius:12px 12px 0 0;
  padding:.5rem 1.1rem; font-weight:600; }}
.stTabs [aria-selected="true"] {{ background:linear-gradient(120deg,{AZUL2},{LILA}); color:#fff !important; }}
#MainMenu, footer {{ visibility:hidden; }}
</style>""", unsafe_allow_html=True)


def base(fig, alto=430, titulo=None, leyenda=True):
    fig.update_layout(height=alto, title=titulo,
                      paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(255,255,255,.72)",
                      font=dict(family="Inter", size=12, color=TINTA),
                      title_font=dict(family="Outfit", size=16),
                      margin=dict(l=55, r=25, t=58 if titulo else 25, b=50),
                      showlegend=leyenda,
                      legend=dict(bgcolor="rgba(255,255,255,.82)",
                                  bordercolor="#DBEAFE", borderwidth=1),
                      colorway=SEC)
    fig.update_xaxes(gridcolor="#E2EAF8")
    fig.update_yaxes(gridcolor="#E2EAF8")
    return fig


def kpi(e, v, s="", c=""):
    return (f'<div class="kpi {c}"><div class="e">{e}</div>'
            f'<div class="v">{v}</div><div class="s">{s}</div></div>')


def tarjeta_ganador(cat, g):
    d = f"{g['metrica']} {g['valor']:.4f}"
    if g.get('desv'):
        d += f" ± {g['desv']:.4f}"
    return (f'<div class="gana"><div class="c">◆ Ganador · {cat}</div>'
            f'<div class="m">{g["modelo"]}</div>'
            f'<div class="d">{d}<br>{g["extra"]}</div></div>')


@st.cache_data
def cargar():
    def js(n):
        p = os.path.join(DIR, n)
        if not os.path.exists(p):
            return None
        # Algunos archivos pueden haberse escrito con la codificación del
        # sistema en lugar de UTF-8. Se intentan ambas antes de fallar.
        for cod in ("utf-8", "cp1252", "latin-1"):
            try:
                return json.load(open(p, encoding=cod))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
        return None
    d = {'reg': js("registro.json"), 'pan': js("panorama.json"),
         'cur': js("curvas.json"), 'mat': js("matrices.json"),
         'imp': js("importancias.json"), 'dis': js("dispersion.json"),
         'cru': js("cruces.json"), 'rank': js("ranking.json")}
    for k, f in [('mapa', "mapa_departamentos.csv"), ('der', "deriva.csv"),
                 ('mue', "muestra.csv")]:
        p = os.path.join(DIR, f)
        if os.path.exists(p):
            try:
                d[k] = pd.read_csv(p)
            except UnicodeDecodeError:
                d[k] = pd.read_csv(p, encoding='latin-1')
        else:
            d[k] = None
    return d


@st.cache_resource
def modelo(ruta):
    try:
        return joblib.load(ruta)
    except ModuleNotFoundError as e:
        st.error(f"No se pudo cargar el modelo porque falta la librería "
                 f"'{e.name}'. Instálela con:  pip install {e.name}")
        st.stop()
    except FileNotFoundError:
        st.error(f"No se encuentra el archivo del modelo: {ruta}")
        st.stop()


D = cargar()
if D['reg'] is None:
    st.error("Faltan los artefactos. Ejecute primero: python J_preparar_mlops.py")
    st.stop()

REG, PAN = D['reg'], D['pan']
NOM = PAN['nombres']
ETIQ = {k: {int(a): b for a, b in v.items()}
        for k, v in PAN['etiquetas'].items()}
COLS, CATS, NUMS = PAN['columnas'], PAN['categoricas'], PAN['numericas']
VERS = list(REG['versiones'].keys())
CV = PAN.get('cv', {})
GAN = PAN.get('ganadores', {})
RANK = D['rank'] or {}
ETQ = lambda c: NOM.get(c, c)
MAPA = D['mapa'].copy() if D['mapa'] is not None else None

# Librerías necesarias para deserializar cada modelo. Si alguna no está
# instalada en el entorno que ejecuta la aplicación, sus modelos no pueden
# cargarse y se retiran de los selectores para evitar un error en pantalla.
import importlib.util
LIBRERIA = {'XGBoost': 'xgboost', 'LightGBM': 'lightgbm',
            'CatBoost': 'catboost'}
NO_DISPONIBLES = [m for m, lib in LIBRERIA.items()
                  if importlib.util.find_spec(lib) is None]

# Referencia del informe para cada modelo (validación cruzada, Fase H)
REF_CV = {
    'clasificacion': {'Random Forest': (0.8098, 0.0023), 'CatBoost': (0.8096, 0.0025),
                      'Regresión Logística': (0.8069, 0.0025),
                      'XGBoost': (0.8048, 0.0019), 'LightGBM': (0.7997, 0.0026)},
    'regresion': {'CatBoost': (0.5581, 0.0083), 'LightGBM': (0.5571, 0.0088),
                  'XGBoost': (0.5525, 0.0086), 'Random Forest': (0.5378, 0.0112),
                  'Ridge': (0.5254, 0.0064)}}


def ordenar(nombres, tipo):
    r = REF_CV.get(tipo, {})
    return sorted(nombres, key=lambda n: -r.get(n, (0, 0))[0])


def etiqueta_modelo(n, tipo):
    g = GAN.get(tipo, {})
    return ("◆ " + n + "  (ganador)") if n == g.get('modelo') else n


with st.sidebar:
    st.markdown("# ◆ MLOps CPV 2024")
    st.caption("Brecha digital y equipamiento del hogar en Bolivia")
    st.markdown("---")
    pagina = st.radio("Secciones", [
        "Inicio", "Panorama del dataset", "Explorador territorial",
        "Mapa de Bolivia", "Clasificación", "Regresión",
        "Ranking del proyecto", "Predicción en vivo",
        "Objetivos adicionales", "MLOps y versionado"],
        label_visibility="collapsed")
    st.markdown("---")
    VER = st.selectbox("Versión del sistema", VERS, index=len(VERS) - 1)
    PAL = st.selectbox("Paleta de color", list(PALETAS), index=0)
    ESC = PALETAS[PAL]
    ESC_R = ALERTA
    st.caption(f"Entrenamiento {REG['versiones'][VER]['n_entrenamiento']:,}"
               f" · Prueba {REG['versiones'][VER]['n_prueba']:,}")
    st.markdown("---")
    st.caption(f"Generado {REG['generado']} · semilla {REG['semilla']}")

VD = REG['versiones'][VER]
MC = [m for m in ordenar(list(VD['clasificacion']), 'clasificacion')
      if m not in NO_DISPONIBLES]
MR = [m for m in ordenar(list(VD['regresion']), 'regresion')
      if m not in NO_DISPONIBLES]

if NO_DISPONIBLES:
    AVISO_LIBS = (
        '<div class="nota alerta"><b>Librerías no instaladas en este '
        'entorno.</b> No se encontraron: <code>'
        + ", ".join(LIBRERIA[m] for m in NO_DISPONIBLES) +
        '</code>. Sus modelos existen en la carpeta mlops/modelos pero no '
        'pueden cargarse sin esas librerías, de modo que se han retirado de '
        'los selectores. Para habilitarlos, ejecute en la terminal: '
        '<code>pip install ' + " ".join(LIBRERIA[m] for m in NO_DISPONIBLES) +
        '</code> y reinicie la aplicación. El resto del sistema funciona con '
        'normalidad.</div>')
else:
    AVISO_LIBS = ""

def aviso_libs():
    if AVISO_LIBS:
        st.markdown(AVISO_LIBS, unsafe_allow_html=True)

AVISO_VERSION = """<div class="nota alerta"><b>Cómo leer estas cifras.</b>
Las métricas de esta página corresponden a la versión seleccionada del
sistema, reentrenada con el volumen que indica la barra lateral. No son las
del informe: allí los modelos se evaluaron sobre la muestra común de 120.000
registros y se validaron con cinco particiones. La columna de referencia
muestra el valor del informe para que ambos puedan compararse.</div>"""

# =============================================================================
if pagina == "Inicio":
    st.markdown(f"""<div class="hero"><div class="tag">Sistema MLOps · Censo 2024</div>
    <h1>Brecha digital y equipamiento del hogar</h1>
    <p>Modelos supervisados sobre {PAN['n_total']:,} viviendas particulares ocupadas,
    con versionado, control de deriva territorial y predicción en línea.</p></div>""",
                unsafe_allow_html=True)

    if GAN:
        st.markdown("### Modelos ganadores del proyecto")
        cg = st.columns(len(GAN))
        nombres = {'clasificacion': 'Clasificación · acceso a internet',
                   'regresion': 'Regresión · equipamiento del hogar',
                   'multiclase': 'Multiclase · nivel de equipamiento'}
        for col, (k, g) in zip(cg, GAN.items()):
            col.markdown(tarjeta_ganador(nombres.get(k, k), g),
                         unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("Ver el criterio de selección de cada ganador"):
            for k, g in GAN.items():
                st.markdown(f"**{nombres.get(k, k)} · {g['modelo']}**")
                st.caption(g['criterio'])
                st.caption("Modelo final: " + g['final'])

    c = st.columns(5)
    for col, (e, v, s, cl) in zip(c, [
            ("Viviendas", f"{PAN['n_total']:,}", "dataset depurado", ""),
            ("Con internet", f"{PAN['pct_internet']:.2f} %",
             f"oficial INE {PAN.get('validacion', {}).get('oficial_ine', 76.3)} %", "lila"),
            ("Equipamiento", f"{PAN['ieh_medio']:.2f}", "bienes de 13", "menta"),
            ("Hogares rurales", f"{PAN['pct_rural']:.1f} %", "del total", "ambar"),
            ("Carencia servicios", f"{PAN['pct_sin_serv']:.1f} %",
             "agua, energía o baño", "coral")]):
        col.markdown(kpi(e, v, s, cl), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    a, b = st.columns([1.15, 1])
    with a:
        lim = PAN.get('limpieza', [])
        if not lim:
            st.markdown('''<div class="nota alerta"><b>Falta el gráfico de
            trazabilidad.</b> El archivo mlops/panorama.json fue generado con
            una versión anterior de la fase de preparación. Vuelva a ejecutar
            <code>python FASE_J_preparar_mlops.py</code> con la versión
            actual.</div>''', unsafe_allow_html=True)
        if lim:
            fig = go.Figure(go.Waterfall(
                orientation="v",
                measure=["absolute"] + ["relative"] * (len(lim) - 1) + ["total"],
                x=[l['etapa'] for l in lim] + ["Dataset limpio"],
                y=[lim[0]['n']] + [-l['quita'] for l in lim[1:]] + [0],
                connector=dict(line=dict(color=GRIS)),
                increasing=dict(marker_color=MENTA),
                decreasing=dict(marker_color=CORAL),
                totals=dict(marker_color=AZUL2),
                text=[f"{lim[0]['n']:,}"] + [f"−{l['quita']:,}" for l in lim[1:]]
                     + [f"{lim[-1]['n']:,}"], textposition="outside"))
            fig.update_yaxes(title="Viviendas")
            st.plotly_chart(base(fig, 430, "Trazabilidad de la depuración", False),
                            use_container_width=True)
    with b:
        fig = go.Figure(go.Pie(
            labels=["Con internet", "Sin internet"],
            values=[PAN['pct_internet'], 100 - PAN['pct_internet']], hole=.64,
            marker=dict(colors=[AZUL2, CORAL], line=dict(color="white", width=4)),
            textinfo="label+percent", textfont=dict(size=14, family="Outfit"),
            pull=[0, .06]))
        fig.add_annotation(text=f"<b>{PAN['pct_internet']:.2f} %</b><br>conectados",
                           showarrow=False,
                           font=dict(size=20, family="Outfit", color=TINTA))
        st.plotly_chart(base(fig, 430, "Cobertura nacional", False),
                        use_container_width=True)

    val = PAN.get('validacion', {})
    st.markdown(f"""<div class="nota dato"><b>Validación externa.</b>
    La cobertura calculada sobre el universo depurado es de
    {val.get('propio', 76.27)} %, frente al {val.get('oficial_ine', 76.3)} %
    publicado por el Instituto Nacional de Estadística: una diferencia de
    {abs(val.get('propio', 76.27) - val.get('oficial_ine', 76.3)):.2f} puntos
    porcentuales. El universo construido equivale al oficial.</div>""",
                unsafe_allow_html=True)

# =============================================================================
elif pagina == "Panorama del dataset":
    st.markdown("# Panorama del dataset")
    t1, t2, t3, t4 = st.tabs(["Explorador de variables", "Composición",
                              "Correlaciones", "Vista previa"])
    with t1:
        c1, c2 = st.columns([1, 1])
        var = c1.selectbox("Variable", COLS, format_func=ETQ,
                           index=COLS.index('v06_piso') if 'v06_piso' in COLS else 0)
        ind = c2.selectbox("Indicador a cruzar", ["internet", "ieh", "hacin", "serv"],
                           format_func=lambda k: {'internet': "% con internet",
                                                  'ieh': "Bienes promedio",
                                                  'hacin': "% hacinamiento crítico",
                                                  'serv': "% carencia de servicios"}[k])
        d = PAN['distribuciones'][var]
        a, b = st.columns(2)
        if 'valores' in d:
            et = [ETIQ.get(var, {}).get(int(v), str(int(v))) for v in d['valores']]
            fig = go.Figure(go.Bar(x=et, y=d['n'],
                                   marker=dict(color=d['n'], colorscale=ESC),
                                   hovertemplate="%{x}<br>%{y:,}<extra></extra>"))
            fig.update_yaxes(title="Viviendas")
            a.plotly_chart(base(fig, 420, f"Distribución · {ETQ(var)}", False),
                           use_container_width=True)
            med = {'internet': PAN['pct_internet'], 'ieh': PAN['ieh_medio'],
                   'hacin': PAN['pct_hacin'], 'serv': PAN['pct_sin_serv']}[ind]
            peor = ind in ('hacin', 'serv')
            col = [CORAL if ((v > med) if peor else (v < med)) else MENTA
                   for v in d[ind]]
            fig = go.Figure(go.Bar(x=et, y=d[ind], marker_color=col,
                                   text=[f"{v:.1f}" for v in d[ind]],
                                   textposition="outside"))
            fig.add_hline(y=med, line_dash="dash", line_color=AZUL,
                          annotation_text=f"Media {med:.2f}")
            b.plotly_chart(base(fig, 420, f"Indicador según {ETQ(var)}", False),
                           use_container_width=True)
        else:
            ctr = [(d['bins'][i] + d['bins'][i + 1]) / 2
                   for i in range(len(d['hist']))]
            fig = go.Figure(go.Bar(x=ctr, y=d['hist'],
                                   marker=dict(color=d['hist'], colorscale=ESC)))
            a.plotly_chart(base(fig, 420, f"Histograma · {ETQ(var)}", False),
                           use_container_width=True)
            b.info("Variable continua: sin desglose por categoría.")
    with t2:
        sel = st.multiselect("Variables", CATS,
                             default=[x for x in ['v06_piso', 'v10_combus',
                                                  'v11_basura', 'v15_servsan']
                                      if x in CATS], format_func=ETQ)
        if sel:
            cols = st.columns(min(len(sel), 2))
            for i, v in enumerate(sel):
                d = PAN['distribuciones'][v]
                if 'valores' not in d:
                    continue
                et = [ETIQ.get(v, {}).get(int(x), str(int(x)))
                      for x in d['valores']]
                fig = go.Figure(go.Pie(labels=et, values=d['n'], hole=.5,
                                       marker=dict(line=dict(color="white",
                                                             width=3)),
                                       textinfo="percent",
                                       textfont=dict(size=11)))
                cols[i % len(cols)].plotly_chart(base(fig, 360, ETQ(v)),
                                                 use_container_width=True)
        ne = PAN.get('nivel_equip', {})
        if ne:
            orden = [k for k in ['Bajo', 'Medio', 'Alto'] if k in ne]
            fig = go.Figure(go.Funnel(y=orden, x=[ne[k] for k in orden],
                                      textinfo="label+value+percent initial",
                                      marker=dict(color=[CORAL, AMBAR, MENTA])))
            st.plotly_chart(base(fig, 340, "Nivel de equipamiento del hogar",
                                 False), use_container_width=True)
    with t3:
        m = np.array(PAN['correlacion']['matriz'])
        vs = [ETQ(v) for v in PAN['correlacion']['vars']]
        fig = go.Figure(go.Heatmap(z=m, x=vs, y=vs, colorscale="RdBu", zmid=0,
                                   zmin=-1, zmax=1, text=np.round(m, 2),
                                   texttemplate="%{text}",
                                   textfont=dict(size=9)))
        st.plotly_chart(base(fig, 640, "Matriz de correlación", False),
                        use_container_width=True)
    with t4:
        if D['mue'] is not None:
            st.dataframe(D['mue'].head(st.slider("Filas", 10, 200, 40)),
                         use_container_width=True, height=440)

# =============================================================================
elif pagina == "Explorador territorial":
    st.markdown("# Explorador territorial")
    CRU = D['cru'] or {}
    if not CRU:
        st.warning("Falta cruces.json: vuelva a ejecutar J_preparar_mlops.py")
    else:
        v = st.selectbox("Variable a cruzar con el departamento", list(CRU),
                         format_func=ETQ)
        cc = pd.DataFrame(CRU[v])
        dep = {int(k): b for k, b in PAN['etiquetas']['idep'].items()}
        cc['departamento'] = cc['idep'].map(dep)
        cc['categoria'] = cc['cat'].map(
            lambda x: ETIQ.get(v, {}).get(int(x), str(int(x))))
        a, b = st.columns([1.1, 1])
        with a:
            fig = go.Figure(go.Treemap(
                labels=list(cc['categoria'] + " · " + cc['departamento']),
                parents=list(cc['departamento']), values=list(cc['n']),
                marker=dict(colors=list(cc['internet']), colorscale=ESC,
                            showscale=True,
                            colorbar=dict(title="% internet", thickness=13)),
                hovertemplate="<b>%{label}</b><br>%{value:,} viviendas<extra></extra>"))
            st.plotly_chart(base(fig, 520,
                                 "Tamaño = viviendas · color = % con internet",
                                 False), use_container_width=True)
        with b:
            piv = cc.pivot_table(index='departamento', columns='categoria',
                                 values='internet')
            fig = go.Figure(go.Heatmap(z=piv.values, x=list(piv.columns),
                                       y=list(piv.index), colorscale=ESC,
                                       text=np.round(piv.values, 1),
                                       texttemplate="%{text}",
                                       textfont=dict(size=9),
                                       colorbar=dict(title="% internet")))
            st.plotly_chart(base(fig, 520,
                                 f"% con internet · departamento × {ETQ(v)}",
                                 False), use_container_width=True)
        piv2 = cc.pivot_table(index='departamento', columns='categoria',
                              values='ieh')
        fig = go.Figure(go.Heatmap(z=piv2.values, x=list(piv2.columns),
                                   y=list(piv2.index), colorscale=ESC,
                                   text=np.round(piv2.values, 2),
                                   texttemplate="%{text}",
                                   textfont=dict(size=9),
                                   colorbar=dict(title="Bienes")))
        st.plotly_chart(base(fig, 460,
                             f"Bienes promedio · departamento × {ETQ(v)}",
                             False), use_container_width=True)

# =============================================================================
elif pagina == "Mapa de Bolivia":
    st.markdown("# Mapa territorial")
    mp = MAPA
    ops = {'sin_internet': "% SIN internet", 'internet': "% con internet",
           'ieh': "Bienes promedio (0-13)",
           'hacin_critico': "% hacinamiento crítico",
           'sin_servicios': "% carencia de servicios",
           'rural': "% población rural", 'personas': "Personas por hogar"}
    c1, c2 = st.columns([1.4, 1])
    met = c1.selectbox("Indicador", list(ops), format_func=lambda k: ops[k])
    modo = c2.radio("Vista", ["Burbujas", "Mapa de calor", "3D"],
                    horizontal=True)
    peor = met in ('sin_internet', 'hacin_critico', 'sin_servicios', 'rural')
    esc = ESC_R if peor else ESC
    k = st.columns(4)
    top, bot = mp.loc[mp[met].idxmax()], mp.loc[mp[met].idxmin()]
    for col, (e, v, s, cl) in zip(k, [
            ("Valor más alto", f"{top[met]:.2f}", str(top['departamento']),
             "coral" if peor else "menta"),
            ("Valor más bajo", f"{bot[met]:.2f}", str(bot['departamento']),
             "menta" if peor else "coral"),
            ("Brecha", f"{top[met] - bot[met]:.2f}", "entre extremos", "lila"),
            ("Promedio", f"{mp[met].mean():.2f}", "9 departamentos", "ambar")]):
        col.markdown(kpi(e, v, s, cl), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    a, b = st.columns([1.3, 1])
    with a:
        if modo == "3D":
            fig = go.Figure(go.Scatter3d(
                x=mp['internet'], y=mp['ieh'], z=mp['sin_servicios'],
                mode="markers+text", text=mp['departamento'],
                textposition="top center", textfont=dict(size=10),
                marker=dict(size=mp['viviendas'] / mp['viviendas'].max() * 34 + 10,
                            color=mp[met], colorscale=esc, opacity=.9,
                            line=dict(width=2, color="white"), showscale=True,
                            colorbar=dict(title=ops[met], thickness=13)),
                hovertemplate="<b>%{text}</b><br>Internet %{x:.1f} %"
                              "<br>Bienes %{y:.2f}<br>Sin servicios %{z:.1f} %"
                              "<extra></extra>"))
            fig.update_layout(scene=dict(
                xaxis_title="% internet", yaxis_title="Bienes",
                zaxis_title="% sin servicios"))
            st.plotly_chart(base(fig, 580, "Espacio tridimensional de carencia",
                                 False), use_container_width=True)
        elif modo == "Mapa de calor":
            try:
                traza = go.Densitymap(
                    lat=mp['lat'], lon=mp['lon'], z=mp[met], radius=75,
                    colorscale=esc, showscale=True,
                    hovertext=mp['departamento'],
                    colorbar=dict(title=ops[met], thickness=14))
                fig = go.Figure(traza)
                fig.update_layout(map_style="carto-positron",
                                  map=dict(center=dict(lat=-16.6, lon=-64.8),
                                           zoom=4.1))
            except AttributeError:
                fig = go.Figure(go.Densmapbox(
                    lat=mp['lat'], lon=mp['lon'], z=mp[met], radius=75,
                    colorscale=esc, showscale=True,
                    hovertext=mp['departamento'],
                    colorbar=dict(title=ops[met], thickness=14)))
                fig.update_layout(mapbox_style="carto-positron",
                                  mapbox=dict(center=dict(lat=-16.6, lon=-64.8),
                                              zoom=4.1))
            st.plotly_chart(base(fig, 580, f"Bolivia · {ops[met]}", False),
                            use_container_width=True)
        else:
            fig = go.Figure(go.Scattergeo(
                lat=mp['lat'], lon=mp['lon'], text=mp['departamento'],
                mode="markers+text", textposition="top center",
                textfont=dict(size=11, family="Outfit", color=TINTA),
                marker=dict(size=mp['viviendas'] / mp['viviendas'].max() * 52 + 16,
                            color=mp[met], colorscale=esc, showscale=True,
                            line=dict(width=2.5, color="white"),
                            colorbar=dict(title=ops[met], thickness=14)),
                customdata=np.stack([mp['viviendas'], mp['internet'], mp['ieh'],
                                     mp['rural']], axis=-1),
                hovertemplate="<b>%{text}</b><br>" + ops[met] +
                              ": %{marker.color:.2f}<br>Viviendas %{customdata[0]:,}"
                              "<br>Internet %{customdata[1]:.1f} %"
                              "<br>Bienes %{customdata[2]:.2f}"
                              "<br>Rural %{customdata[3]:.1f} %<extra></extra>"))
            fig.update_geos(scope="south america", showcountries=True,
                            countrycolor="#94A3B8", showland=True,
                            landcolor="#F8FAFF", showocean=True,
                            oceancolor="#E0F2FE", showlakes=True,
                            lakecolor="#DBEAFE", lataxis_range=[-23.5, -9],
                            lonaxis_range=[-70, -57])
            st.plotly_chart(base(fig, 580, f"Bolivia · {ops[met]}", False),
                            use_container_width=True)
    with b:
        s = mp.sort_values(met)
        fig = go.Figure(go.Bar(x=s[met], y=s['departamento'], orientation='h',
                               marker=dict(color=s[met], colorscale=esc),
                               text=[f"{v:.2f}" for v in s[met]],
                               textposition="outside"))
        fig.add_vline(x=mp[met].mean(), line_dash="dash", line_color=AZUL,
                      annotation_text="Media")
        st.plotly_chart(base(fig, 300, "Ranking departamental", False),
                        use_container_width=True)
        rad = mp.copy()
        ejes = ['internet', 'ieh', 'hacin_critico', 'sin_servicios', 'rural']
        for e in ejes:
            rng = rad[e].max() - rad[e].min()
            rad[e + '_n'] = (rad[e] - rad[e].min()) / (rng if rng else 1) * 100
        dsel = st.multiselect("Comparar departamentos",
                              list(mp['departamento']),
                              default=list(mp.nlargest(3, 'sin_servicios')
                                           ['departamento']))
        fig = go.Figure()
        for i, dd in enumerate(dsel):
            r = rad[rad['departamento'] == dd].iloc[0]
            fig.add_trace(go.Scatterpolar(
                r=[r[e + '_n'] for e in ejes] + [r[ejes[0] + '_n']],
                theta=[ops.get(e, e) for e in ejes] + [ops[ejes[0]]],
                fill='toself', name=dd, opacity=.55,
                line=dict(color=SEC[i % len(SEC)], width=2.5)))
        fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100],
                                                     showticklabels=False)))
        st.plotly_chart(base(fig, 400, "Perfil comparado (escala 0-100)"),
                        use_container_width=True)

    st.markdown(f"""<div class="nota rojo"><b>Nota metodológica.</b>
    El mapa usa los valores <b>observados</b> en el censo, no las predicciones
    de los modelos. Los clasificadores se entrenaron con ponderación de clases
    balanceada, lo que desplaza sus probabilidades hacia abajo entre 15 y 22
    puntos porcentuales. Son adecuados para <b>ordenar hogares por riesgo</b>
    pero no para estimar proporciones poblacionales.</div>""",
                unsafe_allow_html=True)
    with st.expander("Tabla completa y coordenadas paralelas"):
        st.dataframe(mp.drop(columns=['lat', 'lon']), use_container_width=True)
        fig = go.Figure(go.Parcoords(
            line=dict(color=mp['internet'], colorscale=ESC, showscale=True),
            dimensions=[dict(label=ops.get(x, x), values=mp[x])
                        for x in ['internet', 'ieh', 'hacin_critico',
                                  'sin_servicios', 'rural']]))
        st.plotly_chart(base(fig, 420, "Coordenadas paralelas", False),
                        use_container_width=True)

# =============================================================================
elif pagina == "Clasificación":
    g = GAN.get('clasificacion', {})
    st.markdown("# Clasificación · ¿la vivienda tiene internet?")
    aviso_libs()
    if g:
        st.markdown(f"""<div class="nota" style="border-left-color:{AMBAR}">
        <b>◆ Modelo ganador del proyecto: {g['modelo']}</b> · {g['metrica']}
        {g['valor']:.4f} ± {g['desv']:.4f} en validación cruzada de cinco
        particiones, con {g['extra']}. {g['final']}.
        {'<br><i>Este modelo no aparece abajo porque su librería no está '
         'instalada en este entorno.</i>' if g['modelo'] in NO_DISPONIBLES
         else ''}</div>""", unsafe_allow_html=True)
    sel = st.radio("Modelo", MC, horizontal=True,
                   format_func=lambda n: etiqueta_modelo(n, 'clasificacion'))
    m = VD['clasificacion'][sel]['metricas']
    ref = REF_CV['clasificacion'].get(sel)

    c = st.columns(6)
    for col, (k_, v_, cl) in zip(c, [
            ("ROC AUC", m['ROC_AUC'], ""), ("PR AUC", m['PR_AUC'], "lila"),
            ("Exactitud", m['Accuracy'], "menta"),
            ("Precisión", m['Precision'], "ambar"),
            ("Recall", m['Recall'], ""),
            ("Recall SIN internet", m['Recall_min'], "coral")]):
        col.markdown(kpi(k_, f"{v_:.4f}", "", cl), unsafe_allow_html=True)
    if ref:
        st.caption(f"Referencia del informe para {sel}: AUC {ref[0]:.4f} ± "
                   f"{ref[1]:.4f} en validación cruzada de cinco particiones "
                   f"sobre 40.000 registros.")
    st.markdown(AVISO_VERSION, unsafe_allow_html=True)
    st.markdown(f"""<div class="nota rojo">
    <b>El recall de la clase SIN internet ({m['Recall_min']:.4f}) es la métrica
    decisiva.</b> Mide qué proporción de hogares desconectados detecta el
    modelo: son exactamente la población que el proyecto busca identificar.
    </div>""", unsafe_allow_html=True)

    a, b = st.columns(2)
    for col, (clave, ejex, ejey, tit) in zip([a, b], [
            ('roc', "Tasa de falsos positivos", "Tasa de verdaderos positivos",
             "Curvas ROC · el modelo elegido resalta"),
            ('pr', "Recall", "Precision", "Curvas Precision-Recall")]):
        fig = go.Figure()
        for i, n_ in enumerate(MC):
            cu = D['cur'].get(f"{VER}|clf|{n_}")
            if not cu:
                continue
            x, y = (cu['fpr'], cu['tpr']) if clave == 'roc' else (cu['rec'],
                                                                  cu['prec'])
            fig.add_trace(go.Scatter(x=x, y=y, name=n_,
                                     line=dict(width=4.2 if n_ == sel else 1.7,
                                               color=SEC[i % len(SEC)]),
                                     opacity=1 if n_ == sel else .4,
                                     fill='tozeroy' if n_ == sel and clave == 'roc'
                                     else None))
        if clave == 'roc':
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Azar",
                                     line=dict(dash="dash", color=GRIS)))
        fig.update_xaxes(title=ejex)
        fig.update_yaxes(title=ejey)
        col.plotly_chart(base(fig, 430, tit), use_container_width=True)

    a, b, c3 = st.columns([1, 1.2, 1])
    with a:
        mc = np.array(D['mat'][f"{VER}|clf|{sel}"])
        fig = go.Figure(go.Heatmap(z=mc, x=["Predice: sin", "Predice: con"],
                                   y=["Real: sin", "Real: con"], colorscale=ESC,
                                   text=[[f"{v:,}" for v in r] for r in mc],
                                   texttemplate="%{text}",
                                   textfont=dict(size=19, family="Outfit"),
                                   showscale=False))
        st.plotly_chart(base(fig, 370, f"Matriz · {sel}", False),
                        use_container_width=True)
    with b:
        im = D['imp'].get(f"{VER}|clf|{sel}", {})
        if im:
            s = pd.Series(im).sort_values().tail(12)
            fig = go.Figure(go.Bar(x=s.values, y=[ETQ(i) for i in s.index],
                                   orientation='h',
                                   marker=dict(color=s.values, colorscale=ESC)))
            st.plotly_chart(base(fig, 370, "Variables más influyentes", False),
                            use_container_width=True)
    with c3:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=m['ROC_AUC'] * 100,
            number={'valueformat': ".2f",
                    'font': {'size': 36, 'family': 'Outfit'}},
            gauge={'axis': {'range': [50, 100]},
                   'bar': {'color': AZUL2, 'thickness': .78},
                   'steps': [{'range': [50, 70], 'color': "#FEE2E2"},
                             {'range': [70, 80], 'color': "#FEF3C7"},
                             {'range': [80, 100], 'color': "#CCFBF1"}]}))
        st.plotly_chart(base(fig, 370, "AUC × 100", False),
                        use_container_width=True)

    tb = pd.DataFrame({n_: VD['clasificacion'][n_]['metricas'] for n_ in MC}).T
    tb['segundos'] = [VD['clasificacion'][n_]['segundos'] for n_ in MC]
    tb['AUC informe (CV)'] = [REF_CV['clasificacion'].get(n_, (np.nan,))[0]
                              for n_ in MC]
    a, b = st.columns([1.35, 1])
    a.markdown("### Comparación de los cinco finalistas")
    a.caption("La última columna es el valor publicado en el informe. Las "
              "demás corresponden a esta versión del sistema.")
    a.dataframe(tb.style.format("{:.4f}").background_gradient(
        cmap="Blues", subset=['ROC_AUC', 'PR_AUC', 'Recall_min']),
        use_container_width=True)
    ejes = ['Accuracy', 'Precision', 'Recall', 'Recall_min', 'F1', 'ROC_AUC']
    fig = go.Figure()
    for i, n_ in enumerate(MC):
        v = [VD['clasificacion'][n_]['metricas'][e] for e in ejes]
        fig.add_trace(go.Scatterpolar(r=v + [v[0]], theta=ejes + [ejes[0]],
                                      fill='toself', name=n_,
                                      opacity=.9 if n_ == sel else .3,
                                      line=dict(width=3 if n_ == sel else 1.4,
                                                color=SEC[i % len(SEC)])))
    fig.update_layout(polar=dict(radialaxis=dict(range=[.3, 1])))
    b.plotly_chart(base(fig, 430, "Perfil de métricas"),
                   use_container_width=True)

# =============================================================================
elif pagina == "Regresión":
    g = GAN.get('regresion', {})
    st.markdown("# Regresión · índice de equipamiento del hogar")
    aviso_libs()
    if g:
        st.markdown(f"""<div class="nota" style="border-left-color:{AMBAR}">
        <b>◆ Modelo ganador del proyecto: {g['modelo']}</b> · {g['metrica']}
        {g['valor']:.4f} ± {g['desv']:.4f} en validación cruzada, con
        {g['extra']}. {g['final']}.
        {'<br><i>Este modelo no aparece abajo porque su librería no está '
         'instalada en este entorno.</i>' if g['modelo'] in NO_DISPONIBLES
         else ''}</div>""", unsafe_allow_html=True)
    sel = st.radio("Modelo", MR, horizontal=True,
                   format_func=lambda n: etiqueta_modelo(n, 'regresion'))
    m = VD['regresion'][sel]['metricas']
    ref = REF_CV['regresion'].get(sel)
    c = st.columns(5)
    for col, (k_, v_, s_, cl) in zip(c, [
            ("R²", f"{m['R2']:.4f}", "variación explicada", "menta"),
            ("RMSE", f"{m['RMSE']:.4f}", "bienes de 13", ""),
            ("MAE", f"{m['MAE']:.4f}", "error absoluto", "lila"),
            ("MSE", f"{m['MSE']:.4f}", "error cuadrático", "ambar"),
            ("MAPE", f"{m['MAPE']:.2f} %", "excluye ieh = 0", "coral")]):
        col.markdown(kpi(k_, v_, s_, cl), unsafe_allow_html=True)
    if ref:
        st.caption(f"Referencia del informe para {sel}: R² {ref[0]:.4f} ± "
                   f"{ref[1]:.4f} en validación cruzada de cinco particiones.")
    st.markdown(AVISO_VERSION, unsafe_allow_html=True)

    dd = D['dis'].get(f"{VER}|reg|{sel}")
    a, b = st.columns(2)
    if dd:
        r = np.array(dd['real'], dtype=float)
        p = np.array(dd['pred'], dtype=float)
        j = np.random.default_rng(42).uniform(-.28, .28, len(r))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=r + j, y=p, mode="markers", name="Hogares",
                                 marker=dict(size=5, color=p, colorscale=ESC,
                                             opacity=.45,
                                             colorbar=dict(title="Predicho",
                                                           thickness=12))))
        fig.add_trace(go.Scatter(x=[0, 13], y=[0, 13], name="Perfecta",
                                 line=dict(dash="dash", color=CORAL, width=3)))
        fig.update_xaxes(title="Bienes reales")
        fig.update_yaxes(title="Bienes predichos")
        a.plotly_chart(base(fig, 430, "Real frente a predicho"),
                       use_container_width=True)
        res = r - p
        fig = go.Figure(go.Histogram(x=res, nbinsx=45, marker_color=MENTA))
        fig.add_vline(x=0, line_color=TINTA, line_width=2)
        fig.update_xaxes(title="Residuo (real − predicho)")
        b.plotly_chart(base(fig, 430, f"Errores · media {res.mean():+.3f}",
                            False), use_container_width=True)
        fig = go.Figure(go.Histogram2d(x=r, y=p, colorscale=ESC, nbinsx=14,
                                       nbinsy=28,
                                       colorbar=dict(title="Hogares")))
        fig.add_trace(go.Scatter(x=[0, 13], y=[0, 13], mode="lines",
                                 line=dict(dash="dash", color=CORAL, width=3),
                                 showlegend=False))
        fig.update_xaxes(title="Bienes reales")
        fig.update_yaxes(title="Bienes predichos")
        st.plotly_chart(base(fig, 430, "Densidad conjunta real-predicho", False),
                        use_container_width=True)

    a, b = st.columns([1.2, 1])
    im = D['imp'].get(f"{VER}|reg|{sel}", {})
    if im:
        s = pd.Series(im).sort_values().tail(12)
        fig = go.Figure(go.Bar(x=s.values, y=[ETQ(i) for i in s.index],
                               orientation='h',
                               marker=dict(color=s.values, colorscale=ESC)))
        a.plotly_chart(base(fig, 400, "Variables más influyentes", False),
                       use_container_width=True)
    s = pd.Series({n_: VD['regresion'][n_]['metricas']['R2']
                   for n_ in MR}).sort_values()
    fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation='h',
                           marker_color=[MENTA if i == sel else AZUL2
                                         for i in s.index],
                           text=[f"{v:.4f}" for v in s.values],
                           textposition="outside"))
    fig.update_xaxes(title="R²")
    b.plotly_chart(base(fig, 400, "R² por modelo en esta versión", False),
                   use_container_width=True)

    tb = pd.DataFrame({n_: VD['regresion'][n_]['metricas'] for n_ in MR}).T
    tb['R2 informe (CV)'] = [REF_CV['regresion'].get(n_, (np.nan,))[0]
                             for n_ in MR]
    st.markdown("### Comparación de los cinco finalistas")
    st.dataframe(tb.style.format("{:.4f}").background_gradient(
        cmap="Purples", subset=['R2']), use_container_width=True)

# =============================================================================
elif pagina == "Ranking del proyecto":
    st.markdown("# Ranking consolidado del proyecto")
    st.markdown("""<div class="nota dato">Estas son las métricas publicadas en
    el informe: los modelos evaluados en las fases D a H sobre la misma
    partición de prueba de 30.000 observaciones. No se recalculan aquí, de
    modo que coinciden exactamente con el documento.</div>""",
                unsafe_allow_html=True)
    if not RANK:
        st.markdown('''<div class="nota alerta">
        <b>Falta el archivo mlops/ranking.json.</b> Esta sección lee las
        métricas consolidadas por la Fase I, que la fase de preparación copia
        desde la carpeta <code>entregables/</code>. Para generarlo:
        <br><br>
        1. Compruebe que exista la carpeta <code>entregables/</code> con los
        tres archivos <code>I_todas_metricas_*.csv</code>.<br>
        2. Compruebe que <code>FASE_J_preparar_mlops.py</code> sea la versión
        actual: debe contener la línea <code>ranking.json</code>.<br>
        3. Ejecute <code>python FASE_J_preparar_mlops.py</code>.<br>
        4. Reinicie la aplicación.
        </div>''', unsafe_allow_html=True)
    else:
        t1, t2, t3 = st.tabs(["Clasificación binaria", "Regresión",
                              "Multiclase"])
        COLOR_FASE = {'D': AZUL2, 'E': LILA, 'F': MENTA, 'G': AMBAR,
                      'H': CORAL, 'I': CIAN}
        for tab, clave, met, asc, tit in [
                (t1, 'clasificacion', 'ROC_AUC', False, "AUC"),
                (t2, 'regresion', 'R2', False, "R²"),
                (t3, 'multiclase', 'F1', False, "F1 macro")]:
            with tab:
                if clave not in RANK:
                    st.info("Sin datos para esta tarea.")
                    continue
                d = pd.DataFrame(RANK[clave]).sort_values(met, ascending=asc)
                d = d.reset_index(drop=True)
                d.index += 1
                c = st.columns(3)
                c[0].markdown(kpi("Modelos evaluados", f"{len(d)}",
                                  "en las fases D a H"), unsafe_allow_html=True)
                c[1].markdown(kpi(f"Mejor {tit}", f"{d[met].iloc[0]:.4f}",
                                  f"{d['modelo'].iloc[0]} · Fase "
                                  f"{d['fase'].iloc[0]}", "menta"),
                              unsafe_allow_html=True)
                c[2].markdown(kpi("Diferencia extremos",
                                  f"{d[met].iloc[0] - d[met].iloc[-1]:.4f}",
                                  "entre el primero y el último", "lila"),
                              unsafe_allow_html=True)
                s = d.sort_values(met)
                fig = go.Figure(go.Bar(
                    x=s[met], y=[f"{m} ({f})" for m, f in zip(s['modelo'],
                                                              s['fase'])],
                    orientation='h',
                    marker_color=[COLOR_FASE.get(str(f), GRIS)
                                  for f in s['fase']],
                    text=[f"{v:.4f}" for v in s[met]],
                    textposition="outside"))
                lo = max(0, s[met].min() - .03)
                fig.update_xaxes(title=tit, range=[lo, s[met].max() + .02])
                st.plotly_chart(base(fig, 30 * len(d) + 160,
                                     f"Los {len(d)} modelos ordenados por {tit}"
                                     " · color por fase", False),
                                use_container_width=True)
                if 'segundos' in d:
                    fig = go.Figure(go.Scatter(
                        x=d['segundos'], y=d[met], mode="markers+text",
                        text=d['modelo'], textposition="top center",
                        textfont=dict(size=8),
                        marker=dict(size=15,
                                    color=[COLOR_FASE.get(str(f), GRIS)
                                           for f in d['fase']],
                                    line=dict(width=2, color="white"))))
                    fig.update_xaxes(title="Segundos de entrenamiento",
                                     type="log")
                    fig.update_yaxes(title=tit)
                    st.plotly_chart(base(fig, 430,
                                         "Desempeño frente a costo", False),
                                    use_container_width=True)
                st.dataframe(d.style.format(
                    {c_: "{:.4f}" for c_ in d.columns
                     if d[c_].dtype.kind == 'f'}).background_gradient(
                    cmap="Blues", subset=[met]), use_container_width=True)

# =============================================================================
elif pagina == "Predicción en vivo":
    st.markdown("# Predicción en vivo")
    aviso_libs()
    st.markdown("""<div class="nota dato">Describa una vivienda y los modelos
    estimarán su acceso a internet y su nivel de equipamiento. Los modelos
    marcados con ◆ son los ganadores del proyecto.</div>""",
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    mc_sel = c1.selectbox("Modelo de clasificación", MC,
                          format_func=lambda n: etiqueta_modelo(n,
                                                                'clasificacion'))
    mr_sel = c2.selectbox("Modelo de regresión", MR,
                          format_func=lambda n: etiqueta_modelo(n, 'regresion'))
    VIS = [v for v in ['urbrur', 'v06_piso', 'v03_pared', 'v10_combus',
                       'v11_basura', 'v09_energia', 'v07_aguapro',
                       'v15_servsan', 'v16_desague', 'tip_hog', 'tot_pers',
                       'v13_habitac', 'v14_dormit'] if v in COLS]
    vals = dict(REG['valores_tipicos'])
    cols3 = st.columns(3)
    for i, v in enumerate(VIS):
        with cols3[i % 3]:
            if v in ETIQ:
                ks = sorted(ETIQ[v])
                act = int(vals.get(v, ks[0]))
                vals[v] = st.selectbox(ETQ(v), ks,
                                       index=ks.index(act) if act in ks else 0,
                                       format_func=lambda k, e=ETIQ[v]: e[k],
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
        X = pd.DataFrame([{cc: float(f.get(cc, 0)) for cc in COLS}])[COLS]
        p = float(modelo(os.path.join(
            DIR, "modelos", f"{VER}_clf_{mc_sel}.joblib".replace(" ", "_")))
            .predict_proba(X)[0, 1])
        ieh = float(modelo(os.path.join(
            DIR, "modelos", f"{VER}_reg_{mr_sel}.joblib".replace(" ", "_")))
            .predict(X)[0])
        niv, col = ("BAJO riesgo", MENTA) if p >= .70 else (
            ("Riesgo MEDIO", AMBAR) if p >= .45 else ("ALTO riesgo", CORAL))
        a, b, c3 = st.columns(3)
        a.markdown(kpi("Probabilidad de internet", f"{p * 100:.1f} %", niv,
                       "menta" if p >= .7 else ("ambar" if p >= .45 else "coral")),
                   unsafe_allow_html=True)
        b.markdown(kpi("Bienes estimados", f"{ieh:.1f}",
                       f"de 13 · {'Bajo' if ieh <= 3 else ('Medio' if ieh <= 6 else 'Alto')}",
                       "lila"), unsafe_allow_html=True)
        c3.markdown(kpi("Personas por dormitorio", f"{f['hacinamiento']:.2f}",
                        "crítico" if f['hacin_critico'] else "normal",
                        "coral" if f['hacin_critico'] else ""),
                    unsafe_allow_html=True)
        g1, g2 = st.columns(2)
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta", value=p * 100,
            delta={'reference': PAN['pct_internet'], 'suffix': " pp"},
            number={'suffix': " %", 'font': {'size': 42, 'family': 'Outfit'}},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': col, 'thickness': .76},
                   'steps': [{'range': [0, 45], 'color': "#FEE2E2"},
                             {'range': [45, 70], 'color': "#FEF3C7"},
                             {'range': [70, 100], 'color': "#CCFBF1"}],
                   'threshold': {'line': {'color': AZUL, 'width': 4},
                                 'value': PAN['pct_internet']}}))
        g1.plotly_chart(base(fig, 360, "Probabilidad frente a la media nacional",
                             False), use_container_width=True)
        ref = pd.Series({'Esta vivienda': ieh,
                         'Promedio nacional': PAN['ieh_medio']})
        fig = go.Figure(go.Bar(x=ref.values, y=ref.index, orientation='h',
                               marker_color=[col, GRIS],
                               text=[f"{v:.2f}" for v in ref.values],
                               textposition="outside"))
        fig.update_xaxes(range=[0, 13], title="Bienes de 13")
        g2.plotly_chart(base(fig, 360, "Equipamiento estimado", False),
                        use_container_width=True)
        comp = {}
        for n_ in MC:
            try:
                comp[n_] = float(modelo(os.path.join(
                    DIR, "modelos",
                    f"{VER}_clf_{n_}.joblib".replace(" ", "_")))
                    .predict_proba(X)[0, 1]) * 100
            except Exception:
                pass
        if comp:
            s = pd.Series(comp).sort_values()
            fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation='h',
                                   marker=dict(color=s.values, colorscale=ESC),
                                   text=[f"{v:.1f} %" for v in s.values],
                                   textposition="outside"))
            fig.add_vline(x=PAN['pct_internet'], line_dash="dash",
                          line_color=CORAL, annotation_text="Media nacional")
            fig.update_xaxes(title="Probabilidad estimada (%)", range=[0, 112])
            st.plotly_chart(base(fig, 330,
                                 "¿Coinciden los cinco modelos en esta vivienda?",
                                 False), use_container_width=True)
        st.markdown("""<div class="nota rojo"><b>Cómo interpretar esta
        probabilidad.</b> Los modelos están entrenados con ponderación de
        clases, lo que desplaza sus probabilidades hacia abajo. El valor sirve
        para situar a esta vivienda respecto de las demás —ordenarla por
        riesgo— y no como la probabilidad real de que tenga internet.</div>""",
                    unsafe_allow_html=True)

# =============================================================================
elif pagina == "Objetivos adicionales":
    st.markdown("# Objetivos adicionales")
    st.markdown("""<div class="nota dato">El conjunto depurado permite modelar
    otras dimensiones de vulnerabilidad además de la conectividad. Estos dos
    objetivos no forman parte del informe principal: son una extensión del
    sistema.</div>""", unsafe_allow_html=True)
    for obj, info in REG.get('extra', {}).items():
        st.markdown(f"### {ETQ(obj)}")
        st.caption(info['descripcion'])
        c = st.columns(5)
        c[0].markdown(kpi("Prevalencia", f"{info['prevalencia']:.2f} %",
                          "de los hogares", "coral"), unsafe_allow_html=True)
        for col, (k_, v_) in zip(c[1:], info['metricas'].items()):
            col.markdown(kpi(k_, f"{v_:.4f}"), unsafe_allow_html=True)
        a, b, c3 = st.columns([1, 1.2, .9])
        cu = D['cur'].get(f"extra|{obj}")
        if cu:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=cu['fpr'], y=cu['tpr'], name="Modelo",
                                     line=dict(color=AZUL2, width=3),
                                     fill='tozeroy'))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Azar",
                                     line=dict(dash="dash", color=GRIS)))
            a.plotly_chart(base(fig, 330, "Curva ROC"), use_container_width=True)
        s = pd.Series(info['importancias']).sort_values().tail(10)
        fig = go.Figure(go.Bar(x=s.values, y=[ETQ(i) for i in s.index],
                               orientation='h',
                               marker=dict(color=s.values, colorscale=ESC)))
        b.plotly_chart(base(fig, 330, "Variables más influyentes", False),
                       use_container_width=True)
        mm = D['mat'].get(f"extra|{obj}")
        if mm:
            mm = np.array(mm)
            fig = go.Figure(go.Heatmap(z=mm, x=["Pred: no", "Pred: sí"],
                                       y=["Real: no", "Real: sí"],
                                       colorscale=ESC,
                                       text=[[f"{v:,}" for v in r] for r in mm],
                                       texttemplate="%{text}",
                                       textfont=dict(size=15, family="Outfit"),
                                       showscale=False))
            c3.plotly_chart(base(fig, 330, "Matriz de confusión", False),
                            use_container_width=True)
        st.markdown(f"""<div class="nota alerta"><b>Control de fuga de
        información.</b> Se excluyeron {len(info.get('excluidas', []))}
        variables que componen este indicador:
        <code>{', '.join(info.get('excluidas', []))}</code>. Sin esa exclusión
        el AUC sería 1,000 por construcción y no por aprendizaje. El modelo
        trabaja con {info.get('n_predictoras', '—')} predictoras.</div>""",
                    unsafe_allow_html=True)
        st.markdown("---")

# =============================================================================
elif pagina == "MLOps y versionado":
    st.markdown("# MLOps · versionado y control de deriva")
    t1, t2, t3, t4 = st.tabs(["Qué se versiona", "Registro de versiones",
                              "Comparación", "Deriva geográfica"])
    with t1:
        st.markdown("""<div class="nota dato">El versionado de este proyecto
        opera en tres niveles. Cada uno responde a una pregunta distinta sobre
        la reproducibilidad del resultado.</div>""", unsafe_allow_html=True)
        a, b, c3 = st.columns(3)
        a.markdown("""#### 1 · Datos
El archivo original del censo nunca se modifica. La depuración produce un
archivo nuevo con sufijo de versión. El primer procedimiento fue la etapa
preliminar; el segundo, documentado en trece pasos, es el definitivo.""")
        b.markdown("""#### 2 · Procedimiento
Dos bitácoras automáticas. La de limpieza registra cada intervención sobre los
datos con su justificación. La maestra, en Excel, registra cada decisión
metodológica y cada modelo entrenado con todas sus métricas.""")
        c3.markdown("""#### 3 · Modelos
Cada versión del sistema guarda sus métricas, curvas, matrices, importancias y
los modelos serializados. Al mantener fija la configuración y variar solo el
volumen de entrenamiento, la comparación aísla el efecto de los datos.""")
        existe_mlflow = os.path.exists("mlflow.db")
        st.markdown(f"""<div class="nota {'menta' if existe_mlflow else 'alerta'}">
        <b>MLflow:</b> {'registrado en mlflow.db. Puede abrirse con el comando '
        'mlflow ui --backend-store-uri sqlite:///mlflow.db'
        if existe_mlflow else
        'no se encontró mlflow.db en esta carpeta. El registro en MLflow es '
        'opcional en este proyecto: si la biblioteca está instalada, la fase de '
        'preparación lo alimenta automáticamente. El versionado no depende de '
        'ella, porque se implementó con registro propio en registro.json.'}
        </div>""", unsafe_allow_html=True)
    with t2:
        filas = []
        for v, d in REG['versiones'].items():
            for t, dd in [('Clasificación', 'clasificacion'),
                          ('Regresión', 'regresion')]:
                for n_, i_ in d[dd].items():
                    filas.append({'Versión': v, 'Tipo': t, 'Modelo': n_,
                                  'Entrenamiento': d['n_entrenamiento'],
                                  'Métrica': 'ROC_AUC' if dd == 'clasificacion'
                                  else 'R2',
                                  'Valor': i_['metricas']['ROC_AUC']
                                  if dd == 'clasificacion' else i_['metricas']['R2'],
                                  'Segundos': i_['segundos']})
        reg = pd.DataFrame(filas)
        st.dataframe(reg.style.format({'Valor': "{:.4f}", 'Segundos': "{:.1f}",
                                       'Entrenamiento': "{:,}"})
                     .background_gradient(cmap="Blues", subset=['Valor']),
                     use_container_width=True, height=400)
        fig = go.Figure(go.Scatter(
            x=reg['Segundos'], y=reg['Valor'], mode="markers+text",
            text=reg['Modelo'] + " · " + reg['Versión'],
            textposition="top center", textfont=dict(size=8),
            marker=dict(size=16, color=reg['Valor'], colorscale=ESC,
                        line=dict(width=2, color="white"), showscale=True,
                        colorbar=dict(title="Métrica"))))
        fig.update_xaxes(title="Segundos de entrenamiento", type="log")
        fig.update_yaxes(title="Métrica principal")
        st.plotly_chart(base(fig, 440, "Desempeño frente a costo", False),
                        use_container_width=True)
    with t3:
        a, b = st.columns(2)
        for col, (tipo, met, tit) in zip([a, b], [
                ('clasificacion', 'ROC_AUC', "Clasificación · AUC por versión"),
                ('regresion', 'R2', "Regresión · R² por versión")]):
            fig = go.Figure()
            for i, v in enumerate(VERS):
                d = REG['versiones'][v][tipo]
                fig.add_trace(go.Bar(
                    name=f"{v} ({REG['versiones'][v]['n_entrenamiento']:,})",
                    x=list(d), y=[d[k]['metricas'][met] for k in d],
                    marker_color=SEC[i],
                    text=[f"{d[k]['metricas'][met]:.4f}" for k in d],
                    textposition="outside"))
            fig.update_layout(barmode="group")
            vals = [REG['versiones'][v][tipo][k]['metricas'][met]
                    for v in VERS for k in REG['versiones'][v][tipo]]
            fig.update_yaxes(range=[min(vals) * .97, max(vals) * 1.02],
                             title=met)
            col.plotly_chart(base(fig, 440, tit), use_container_width=True)
        if len(VERS) >= 2:
            v1, v2 = VERS[0], VERS[-1]
            d1 = REG['versiones'][v1]['clasificacion']
            d2 = REG['versiones'][v2]['clasificacion']
            dif = {k: d2[k]['metricas']['ROC_AUC'] - d1[k]['metricas']['ROC_AUC']
                   for k in d1 if k in d2}
            s = pd.Series(dif).sort_values()
            fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation='h',
                                   marker_color=[MENTA if v > 0 else CORAL
                                                 for v in s.values],
                                   text=[f"{v:+.4f}" for v in s.values],
                                   textposition="outside"))
            fig.add_vline(x=0, line_color=TINTA)
            fig.update_xaxes(title=f"Cambio de AUC de {v1} a {v2}")
            st.plotly_chart(base(fig, 380,
                                 "Efecto de ampliar el volumen de entrenamiento",
                                 False), use_container_width=True)
    with t4:
        de, dr = D['der'], REG.get('deriva', {})
        if de is not None:
            c = st.columns(3)
            c[0].markdown(kpi("AUC donde entrenó",
                              f"{dr.get('auc_entrenamiento', 0):.4f}",
                              ", ".join(dr.get('departamentos_entrenamiento', [])),
                              "menta"), unsafe_allow_html=True)
            c[1].markdown(kpi("AUC en departamentos no vistos",
                              f"{dr.get('auc_no_visto', 0):.4f}",
                              "capacidad de generalización", "ambar"),
                          unsafe_allow_html=True)
            c[2].markdown(kpi("Caída", f"{dr.get('caida', 0):+.4f}",
                              "deriva geográfica", "coral"),
                          unsafe_allow_html=True)
            a, b = st.columns([1.25, 1])
            s = de.sort_values('auc')
            fig = go.Figure(go.Bar(
                x=s['auc'], y=s['departamento'], orientation='h',
                marker_color=[MENTA if g_ == 'Entrenamiento' else CORAL
                              for g_ in s['grupo']],
                text=[f"{v:.4f}" for v in s['auc']], textposition="outside"))
            fig.add_vline(x=dr.get('auc_entrenamiento', 0), line_dash="dash",
                          line_color=AZUL, annotation_text="Media entrenamiento")
            fig.update_xaxes(title="AUC",
                             range=[max(0, s['auc'].min() - .06),
                                    s['auc'].max() + .03])
            a.plotly_chart(base(fig, 460,
                                "Verde: donde entrenó · Rojo: no visto", False),
                           use_container_width=True)
            if MAPA is not None:
                mm = MAPA.merge(de[['departamento', 'auc', 'grupo']],
                                on='departamento', how='left')
                fig = go.Figure(go.Scattergeo(
                    lat=mm['lat'], lon=mm['lon'], text=mm['departamento'],
                    mode="markers+text", textposition="top center",
                    textfont=dict(size=10),
                    marker=dict(size=26, color=mm['auc'], colorscale=ESC_R,
                                reversescale=True, showscale=True,
                                line=dict(width=2.5, color="white"),
                                colorbar=dict(title="AUC", thickness=13)),
                    hovertemplate="<b>%{text}</b><br>AUC %{marker.color:.4f}"
                                  "<extra></extra>"))
                fig.update_geos(scope="south america", showcountries=True,
                                countrycolor="#94A3B8", showland=True,
                                landcolor="#F8FAFF", showocean=True,
                                oceancolor="#E0F2FE",
                                lataxis_range=[-23.5, -9],
                                lonaxis_range=[-70, -57])
                b.plotly_chart(base(fig, 460, "Dónde se degrada el modelo",
                                    False), use_container_width=True)
            st.markdown(f"""<div class="nota alerta"><b>Qué significa.</b>
            El modelo se entrenó solo con
            {", ".join(dr.get('departamentos_entrenamiento', []))} y se evaluó
            sobre los seis departamentos restantes, que no vio durante el
            entrenamiento. La caída de <b>{dr.get('caida', 0):+.4f}</b> en AUC
            es deriva geográfica: el modelo no generaliza igual en todo el
            territorio. En producción obligaría a reentrenar con cobertura
            nacional o a construir modelos regionales.</div>""",
                        unsafe_allow_html=True)
            st.dataframe(de, use_container_width=True)
