import streamlit as st
import numpy as np
import pandas as pd
import requests
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de página
st.set_page_config(
    page_title="OddsDeconstruct AI — Quant & Live API",
    page_icon="⚡",
    layout="wide"
)

# Estilo visual Dark Mode
st.markdown("""
<style>
    .main { background-color: #0b0f19; }
    .stApp { background-color: #0b0f19; color: #f8fafc; }
    div[data-testid="metric-container"] {
        background-color: #1e293b;
        border: 1px solid #334155;
        padding: 12px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. CONEXIÓN A API GRATUITA (THE-ODDS-API)
# ==========================================

st.sidebar.header("🔑 Configuración de API Gratuita")
api_key = st.sidebar.text_input(
    "API Key (The-Odds-API):", 
    value="", 
    type="password",
    help="Consigue tu clave 100% gratis en https://the-odds-api.com (500 solicitudes/mes free)"
)

region = st.sidebar.selectbox("Región de Casa de Apuestas:", ["eu", "us", "uk", "au"], index=0)
bankroll = st.sidebar.number_input("Capital Total ($ Bankroll):", value=1000.0, step=100.0)
kelly_fraction = st.sidebar.slider("Fracción Criterio de Kelly:", 0.1, 1.0, 0.25)

@st.cache_data(ttl=600) # Cache de 10 minutos para ahorrar llamadas a la API
def fetch_live_odds(key, reg):
    if not key:
        return None
    url = f"https://api.the-odds-api.com/v4/sports/soccer/odds/?apiKey={key}&regions={reg}&markets=h2h,totals&oddsFormat=decimal"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error en la API: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error de conexión: {e}")
        return None

# ==========================================
# 2. MOTOR QUANT: MONTECARLO (50K) & SHIN
# ==========================================

class ShinEngine:
    @staticmethod
    def deoverround(odds_list):
        odds = np.array(odds_list, dtype=float)
        recip_sum = np.sum(1.0 / odds)
        
        def shin_obj(z):
            p = (np.sqrt(z**2 + 4 * (1 - z) * (1.0 / odds) / recip_sum) - z) / (2 * (1 - z))
            return (np.sum(p) - 1.0)**2

        res = minimize(shin_obj, x0=0.02, bounds=[(0.0, 0.4)])
        z_opt = res.x[0] if res.success else 0.0
        p_clean = (np.sqrt(z_opt**2 + 4 * (1 - z_opt) * (1.0 / odds) / recip_sum) - z_opt) / (2 * (1 - z_opt))
        return p_clean / np.sum(p_clean), z_opt

class MonteCarloEngine:
    def __init__(self, simulations=50000, phi=1.15):
        self.simulations = simulations
        self.phi = phi

    def run(self, l_h, l_a):
        p_h, n_h = 1.0 / self.phi, l_h / (self.phi - 1.0)
        p_a, n_a = 1.0 / self.phi, l_a / (self.phi - 1.0)

        sim_h = np.random.negative_binomial(n_h, p_h, self.simulations)
        sim_a = np.random.negative_binomial(n_a, p_a, self.simulations)
        totals = sim_h + sim_a

        return {
            'totals': totals, 'sim_h': sim_h, 'sim_a': sim_a,
            'p_o15': np.mean(totals > 1.5), 'p_u15': np.mean(totals <= 1.5),
            'p_o25': np.mean(totals > 2.5), 'p_u25': np.mean(totals <= 2.5),
            'p_o35': np.mean(totals > 3.5), 'p_u35': np.mean(totals <= 3.5)
        }

# ==========================================
# 3. INTERFAZ Y DESPLIEGUE DE PARTIDOS
# ==========================================

st.title("⚡ OddsDeconstruct AI — Real-Time API Engine")

if not api_key:
    st.info("💡 **Instrucciones:** Ingresa tu clave gratuita de *The-Odds-API* en la barra lateral izquierda para cargar los partidos en vivo. Si no tienes una, obténla en [the-odds-api.com](https://the-odds-api.com) de forma gratuita.")
    # Datos de demostración
    live_data = None
else:
    live_data = fetch_live_odds(api_key, region)

if live_data:
    match_titles = [f"{m['sport_title']}: {m['home_team']} vs {m['away_team']}" for m in live_data if 'bookmakers' in m and len(m['bookmakers']) > 0]
    selected_idx = st.selectbox("📌 Selecciona Partido en Vivo:", range(len(match_titles)), format_func=lambda x: match_titles[x])
    
    selected_match = live_data[selected_idx]
    bookmaker = selected_match['bookmakers'][0] # Primera casa de apuestas disponible
    
    # Extracción de cuotas Totales (Over/Under 2.5)
    odd_o25, odd_u25 = 1.90, 1.90 # Valores por defecto
    for m in bookmaker['markets']:
        if m['key'] == 'totals':
            for outcome in m['outcomes']:
                if outcome['name'] == 'Over' and outcome.get('point') == 2.5:
                    odd_o25 = outcome['price']
                elif outcome['name'] == 'Under' and outcome.get('point') == 2.5:
                    odd_u25 = outcome['price']

    st.subheader(f"🏟️ {selected_match['home_team']} vs {selected_match['away_team']}")
    
    # Sliders para ajuste
