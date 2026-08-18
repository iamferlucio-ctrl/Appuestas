import streamlit as st
import numpy as np
import pandas as pd
import requests

# Configuración de página
st.set_page_config(page_title="OddsDeconstruct AI", page_icon="⚡", layout="wide")

st.markdown("""
<style>
    .main { background-color: #0b0f19; }
    .stApp { background-color: #0b0f19; color: #f8fafc; }
    div[data-testid="metric-container"] {
        background-color: #1e293b; border: 1px solid #334155; padding: 10px; border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Barra Lateral
st.sidebar.header("🔑 Configuración")
api_key = st.sidebar.text_input("API Key:", value="", type="password")
region = st.sidebar.selectbox("Región:", ["eu", "us", "uk", "au"], index=0)
bankroll = st.sidebar.number_input("Bankroll ($):", value=1000.0, step=100.0)
kelly_fraction = st.sidebar.slider("Fracción Kelly:", 0.1, 1.0, 0.25)

@st.cache_data(ttl=600)
def fetch_live_odds(key, reg):
    if not key:
        return None
    url = f"https://api.the-odds-api.com/v4/sports/soccer/odds/?apiKey={key}&regions={reg}&markets=h2h,totals&oddsFormat=decimal"
    try:
        res = requests.get(url, timeout=10)
        return res.json() if res.status_code == 200 else None
    except Exception:
        return None

# Motor de Montecarlo Seguro
def run_montecarlo(l_h, l_a, sims=50000, phi=1.15):
    p_h, n_h = 1.0 / phi, max(0.1, l_h / (phi - 1.0))
    p_a, n_a = 1.0 / phi, max(0.1, l_a / (phi - 1.0))
    sim_h = np.random.negative_binomial(n_h, p_h, sims)
    sim_a = np.random.negative_binomial(n_a, p_a, sims)
    totals = sim_h + sim_a
    return {
        'totals': totals,
        'p_o15': np.mean(totals > 1.5), 'p_u15': np.mean(totals <= 1.5),
        'p_o25': np.mean(totals > 2.5), 'p_u25': np.mean(totals <= 2.5),
        'p_o35': np.mean(totals > 3.5), 'p_u35': np.mean(totals <= 3.5)
    }

st.title("⚡ OddsDeconstruct AI — Real-Time Quant Engine")

if not api_key:
    st.info("👈 Ingresa tu API Key en el menú lateral para cargar partidos.")
else:
    data = fetch_live_odds(api_key, region)
    if not data or not isinstance(data, list):
        st.error("No se pudieron obtener datos. Verifica tu API Key o la región seleccionada.")
    else:
        matches = [m for m in data if m.get('bookmakers')]
        if not matches:
            st.warning("No hay partidos con cuotas disponibles en este momento.")
        else:
            titles = [f"{m.get('sport_title', 'Futbol')}: {m.get('home_team')} vs {m.get('away_team')}" for m in matches]
            idx = st.selectbox("📌 Selecciona Partido en Vivo:", range(len(titles)), format_func=lambda x: titles[x])
            
            selected = matches[idx]
            home_name = selected.get('home_team', 'Local')
            away_name = selected.get('away_team', 'Visita')
            
            st.subheader(f"🏟️ {home_name} vs {away_name}")
            
            # Extracción segura de cuotas
            odd_o25, odd_u25 = 1.90, 1.90
            for bm in selected.get('bookmakers', []):
                for mkt in bm.get('markets', []):
                    if mkt.get('key') == 'totals':
                        for out in mkt.get('outcomes', []):
                            if out.get('point') == 2.5:
                                if out.get('name') == 'Over': odd_o25 = float(out.get('price', 1.90))
                                elif out.get('name') == 'Under': odd_u25 = float(out.get('price', 1.90))

            # Controles de xG
            c1, c2 = st.columns(2)
            l_h = c1.slider(f"xG {home_name}:", 0.2, 4.0, 1.3)
            l_a = c2.slider(f"xG {away_name}:", 0.2, 4.0, 1.1)

            # Ejecución
            mc = run_montecarlo(l_h, l_a)

            # Métricas
            m1, m2, m3 = st.columns(3)
            m1.metric("Goles Totales Esperados", f"{np.mean(mc['totals']):.2f}")
            m2.metric("Prob. Real Over 2.5", f"{mc['p_o25']*100:.1f}%")
            m3.metric("Cuota Casa Over 2.5", f"{odd_o25:.2f}")

            # Tabla de Decisiones
            odds_map = {
                "Over 1.5": (1.30, mc['p_o15']),
                "Under 1.5": (3.50, mc['p_u15']),
                "Over 2.5": (odd_o25, mc['p_o25']),
                "Under 2.5": (odd_u25, mc['p_u25']),
                "Over 3.5": (2.80, mc['p_o35']),
                "Under 3.5": (1.45, mc['p_u35']),
            }

            rows = []
            for name, (odd, p_real) in odds_map.items():
                ev = (p_real * odd) - 1.0
                b = max(odd - 1.0, 0.01)
                k_full = (b * p_real - (1.0 - p_real)) / b
                k_stake = max(0.0, k_full * kelly_fraction)
                
                diag = "NO APOSTAR"
                if ev > 0.05 and k_stake > 0:
                    diag = "🔥 APUESTA DE VALOR (+EV)"
                elif ev < -0.10:
                    diag = "⚠️ TRAMPA DE MERCADO"

                rows.append({
                    "Mercado": name,
                    "Cuota": f"{odd:.2f}",
                    "Prob. IA (50k)": f"{p_real*100:.1f}%",
                    "Valor Esperado (+EV)": f"{ev*100:+.2f}%",
                    "Apostar ($)": f"${k_stake * bankroll:.2f}",
                    "Diagnóstico": diag
                })

            st.subheader("📋 Matriz de Decisiones & Gestión Kelly")
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
