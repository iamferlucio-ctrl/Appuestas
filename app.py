import streamlit as st
import numpy as np
import pandas as pd
import requests
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="OddsDeconstruct Quant AI", page_icon="⚡", layout="wide")

st.markdown("""
<style>
    .main { background-color: #0b0f19; }
    .stApp { background-color: #0b0f19; color: #f8fafc; }
    div[data-testid="metric-container"] {
        background-color: #1e293b; border: 1px solid #334155; padding: 12px; border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. MOTOR MATEMÁTICO CUANTITATIVO
# ==========================================

class ShinEngine:
    @staticmethod
    def deoverround(odds_list):
        odds = np.array(odds_list, dtype=float)
        if len(odds) < 2 or np.any(odds <= 1.0):
            return np.ones(len(odds)) / len(odds), 0.0
        
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
        p_h, n_h = 1.0 / self.phi, max(0.1, l_h / (self.phi - 1.0))
        p_a, n_a = 1.0 / self.phi, max(0.1, l_a / (self.phi - 1.0))
        
        sim_h = np.random.negative_binomial(n_h, p_h, self.simulations)
        sim_a = np.random.negative_binomial(n_a, p_a, self.simulations)
        totals = sim_h + sim_a

        matrix = np.zeros((6, 6))
        for h, a in zip(sim_h, sim_a):
            if h < 6 and a < 6:
                matrix[h, a] += 1
        matrix = (matrix / self.simulations) * 100

        return {
            'totals': totals,
            'matrix': matrix,
            'p_over': lambda line: np.mean(totals > line),
            'p_under': lambda line: np.mean(totals < line),
            'p_exact': lambda line: np.mean(totals == line)
        }

# ==========================================
# 2. CONFIGURACIÓN Y API
# ==========================================

st.sidebar.header("🔑 Panel de Control")
api_key = st.sidebar.text_input("API Key (The-Odds-API):", value="", type="password")
region = st.sidebar.selectbox("Región de Mercado:", ["eu", "us", "uk", "au"], index=0)
bankroll = st.sidebar.number_input("Capital Total ($ Bankroll):", value=1000.0, step=100.0)
kelly_fraction = st.sidebar.slider("Fracción de Kelly:", 0.05, 0.50, 0.25, step=0.05)

@st.cache_data(ttl=300)
def fetch_data(key, reg):
    if not key: return None
    url = f"https://api.the-odds-api.com/v4/sports/soccer/odds/?apiKey={key}&regions={reg}&markets=totals&oddsFormat=decimal"
    try:
        r = requests.get(url, timeout=10)
        return r.json() if r.status_code == 200 else None
    except: return None

# ==========================================
# 3. INTERFAZ Y RENDERIZADO
# ==========================================

st.title("⚡ OddsDeconstruct AI — Quant & Institutional Engine")

if not api_key:
    st.info("👈 Ingresa tu API Key en el menú lateral.")
else:
    raw_data = fetch_data(api_key, region)
    if not raw_data or not isinstance(raw_data, list):
        st.error("Error de autenticación o datos con la API.")
    else:
        valid_matches = [m for m in raw_data if m.get('bookmakers')]
        if not valid_matches:
            st.warning("No hay mercados de Totales disponibles.")
        else:
            titles = [f"{m.get('sport_title', 'Fútbol')}: {m.get('home_team')} vs {m.get('away_team')}" for m in valid_matches]
            idx = st.selectbox("📌 Selecciona Encuentro:", range(len(titles)), format_func=lambda x: titles[x])
            
            match = valid_matches[idx]
            home_team = match.get('home_team', 'Local')
            away_team = match.get('away_team', 'Visita')

            totals_market = {}
            for bm in match.get('bookmakers', []):
                for mkt in bm.get('markets', []):
                    if mkt.get('key') == 'totals':
                        for out in mkt.get('outcomes', []):
                            point = float(out.get('point', 0.0))
                            name = out.get('name')
                            price = float(out.get('price', 0.0))
                            if point not in totals_market: totals_market[point] = {}
                            totals_market[point][name] = price

            st.subheader(f"🏟️ {home_team} vs {away_team}")

            c1, c2 = st.columns(2)
            l_h = c1.slider(f"xG Estimado {home_team}:", 0.2, 4.0, 1.35, 0.05)
            l_a = c2.slider(f"xG Estimado {away_team}:", 0.2, 4.0, 1.15, 0.05)

            mc = MonteCarloEngine()
            results = mc.run(l_h, l_a)

            # Selección de línea para Shin Z
            main_line = 2.5 if 2.5 in totals_market else (list(totals_market.keys())[0] if totals_market else None)
            z_score = 0.0
            if main_line and 'Over' in totals_market[main_line] and 'Under' in totals_market[main_line]:
                _, z_score = ShinEngine.deoverround([totals_market[main_line]['Over'], totals_market[main_line]['Under']])

            m1, m2, m3 = st.columns(3)
            m1.metric("Total Goles Esperados (xG)", f"{l_h + l_a:.2f}")
            m2.metric("Media Simulada (Montecarlo)", f"{np.mean(results['totals']):.2f}")
            m3.metric(f"Información Oculta Shin Z ({main_line if main_line else 'N/A'})", f"{z_score*100:.2f}%")

            # Matriz Multimercado con soporte para Líneas Asiáticas (Push)
            rows = []
            for point in sorted(totals_market.keys()):
                prices = totals_market[point]
                if 'Over' in prices and 'Under' in prices:
                    o_price = prices['Over']
                    u_price = prices['Under']
                    
                    p_o = results['p_over'](point)
                    p_u = results['p_under'](point)
                    p_push = results['p_exact'](point) if point.is_integer() else 0.0
                    
                    # Cálculo de EV ajustado por Devolución (Push)
                    ev_o = (p_o * o_price + p_push) - 1.0
                    ev_u = (p_u * u_price + p_push) - 1.0
                    
                    # Criterio Kelly para apuestas con tasa de devolución
                    p_effective_o = p_o / (1.0 - p_push) if p_push < 1.0 else 0.0
                    b_o = o_price - 1.0
                    k_o_full = (b_o * p_effective_o - (1.0 - p_effective_o)) / b_o if b_o > 0 else 0
                    k_o = max(0.0, k_o_full * kelly_fraction) if ev_o > 0 else 0.0

                    p_effective_u = p_u / (1.0 - p_push) if p_push < 1.0 else 0.0
                    b_u = u_price - 1.0
                    k_u_full = (b_u * p_effective_u - (1.0 - p_effective_u)) / b_u if b_u > 0 else 0
                    k_u = max(0.0, k_u_full * kelly_fraction) if ev_u > 0 else 0.0

                    rows.append({
                        "Mercado": f"Over {point}", "Cuota": f"{o_price:.2f}",
                        "Prob. Win": f"{p_o*100:.1f}%", "Push": f"{p_push*100:.1f}%",
                        "EV (+/-)": f"{ev_o*100:+.2f}%", "Stake ($)": f"${k_o * bankroll:.2f}",
                        "Diagnóstico": "🔥 VALOR (+EV)" if ev_o > 0.03 else ("⚠️ TRAMPA" if ev_o < -0.08 else "NEUTRO")
                    })
                    
                    rows.append({
                        "Mercado": f"Under {point}", "Cuota": f"{u_price:.2f}",
                        "Prob. Win": f"{p_u*100:.1f}%", "Push": f"{p_push*100:.1f}%",
                        "EV (+/-)": f"{ev_u*100:+.2f}%", "Stake ($)": f"${k_u * bankroll:.2f}",
                        "Diagnóstico": "🔥 VALOR (+EV)" if ev_u > 0.03 else ("⚠️ TRAMPA" if ev_u < -0.08 else "NEUTRO")
                    })

            st.subheader("📋 Matriz Multimercado Dinámica")
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            st.subheader("📈 Análisis Gráfico Cuantitativo")
            g1, g2 = st.columns(2)

            with g1:
                fig, ax = plt.subplots(figsize=(5, 3.8))
                fig.patch.set_facecolor('#0b0f19')
                ax.set_facecolor('#0b0f19')
                counts, bins, patches = ax.hist(results['totals'], bins=np.arange(0, 9)-0.5, rwidth=0.85, color='#06b6d4', edgecolor='#1e293b')
                ax.set_title("Distribución de Frecuencia de Goles", color="white", fontsize=10, pad=12)
                ax.tick_params(colors='white')
                ax.grid(axis='y', linestyle='--', alpha=0.2)
                st.pyplot(fig)

            with g2:
                fig2, ax2 = plt.subplots(figsize=(5, 3.8))
                fig2.patch.set_facecolor('#0b0f19')
                ax2.set_facecolor('#0b0f19')
                
                # Renderizado con contraste adaptativo automático
                sns.heatmap(results['matrix'], annot=True, fmt=".1f", cmap="mako", cbar=False, ax=ax2,
                            annot_kws={"size": 8})
                
                ax2.set_title("Marcadores Exactos Probables (%)", color="white", fontsize=10, pad=12)
                ax2.set_xlabel(f"Goles {away_team}", color="white", fontsize=9)
                ax2.set_ylabel(f"Goles {home_team}", color="white", fontsize=9)
                ax2.tick_params(colors='white')
                st.pyplot(fig2)
