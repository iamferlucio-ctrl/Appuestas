import os
import streamlit as st
import numpy as np
import pandas as pd
import requests
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="OddsDeconstruct Institutional AI", page_icon="⚡", layout="wide")

st.markdown("""
<style>
    .main { background-color: #0b0f19; }
    .stApp { background-color: #0b0f19; color: #f8fafc; }
    div[data-testid="metric-container"] {
        background-color: #1e293b; border: 1px solid #334155; padding: 12px; border-radius: 8px;
    }
    .report-box {
        background-color: #1e293b; border-left: 4px solid #06b6d4; padding: 15px; border-radius: 6px; margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. MOTOR MATEMÁTICO CUANTITATIVO
# ==========================================

class ShinEngine:
    @staticmethod
    def deoverround(odds_list):
        odds = np.array([o for o in odds_list if o > 1.0], dtype=float)
        if len(odds) < 2:
            return np.ones(len(odds)) / max(1, len(odds)), 0.0
        
        recip_sum = np.sum(1.0 / odds)
        def shin_obj(z):
            p = (np.sqrt(z**2 + 4 * (1 - z) * (1.0 / odds) / recip_sum) - z) / (2 * (1 - z))
            return (np.sum(p) - 1.0)**2

        res = minimize(shin_obj, x0=0.02, bounds=[(0.0, 0.4)])
        z_opt = res.x[0] if res.success else 0.0
        p_clean = (np.sqrt(z_opt**2 + 4 * (1 - z_opt) * (1.0 / odds) / recip_sum) - z_opt) / (2 * (1 - z_opt))
        return p_clean / np.sum(p_clean), z_opt

class InstitutionalMonteCarlo:
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
            'totals': totals, 'matrix': matrix,
            'p_home': np.mean(sim_h > sim_a),
            'p_draw': np.mean(sim_h == sim_a),
            'p_away': np.mean(sim_h < sim_a),
            'p_over': lambda line: np.mean(totals > line),
            'p_under': lambda line: np.mean(totals < line),
            'p_exact_total': lambda line: np.mean(totals == line)
        }

# ==========================================
# 2. CONFIGURACIÓN Y PERSISTENCIA DE API KEY
# ==========================================

st.sidebar.header("🔑 Control Cuantitativo")

# Detección automática de la API Key desde variables de entorno (.Renviron / Secrets)
env_api_key = os.getenv("ODDS_API_KEY") or os.getenv("API_KEY") or ""

if env_api_key:
    api_key = env_api_key
    st.sidebar.success("✅ API Key cargada automáticamente.")
else:
    api_key = st.sidebar.text_input("API Key (The-Odds-API):", value="", type="password")

region = st.sidebar.selectbox("Región Mercado:", ["eu", "us", "uk", "au"], index=0)
bankroll = st.sidebar.number_input("Capital Bankroll ($):", value=1000.0, step=100.0)
kelly_fraction = st.sidebar.slider("Fracción Kelly (Riesgo):", 0.05, 0.50, 0.25, step=0.05)

@st.cache_data(ttl=300)
def fetch_data_robust(key, reg):
    if not key: return None, "Falta API Key"
    url = f"https://api.the-odds-api.com/v4/sports/soccer/odds/?apiKey={key}&regions={reg}&markets=h2h,totals,spreads&oddsFormat=decimal"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json(), None
        else:
            return None, f"HTTP Error {r.status_code}: {r.text}"
    except Exception as e:
        return None, str(e)

# ==========================================
# 3. INTERFAZ E INFORMES DE ALTO NIVEL
# ==========================================

st.title("⚡ OddsDeconstruct AI — Quant & Institutional Platform")

if not api_key:
    st.info("👈 Ingresa tu API Key en la barra lateral para sincronizar los mercados.")
else:
    raw_data, err_msg = fetch_data_robust(api_key, region)
    if raw_data is None:
        st.error(f"Error al conectar con la API: {err_msg}")
    else:
        valid_matches = [m for m in raw_data if m.get('bookmakers')]
        if not valid_matches:
            st.warning("No hay encuentros activos con cuotas para la región seleccionada.")
        else:
            titles = [f"{m.get('sport_title', 'Fútbol')}: {m.get('home_team')} vs {m.get('away_team')}" for m in valid_matches]
            idx = st.selectbox("📌 Selecciona Evento Deportivo:", range(len(titles)), format_func=lambda x: titles[x])
            
            match = valid_matches[idx]
            home_team = match.get('home_team', 'Local')
            away_team = match.get('away_team', 'Visita')

            # Extracción estructurada de mercados
            mkt_data = {'h2h': {}, 'totals': {}, 'spreads': {}}
            for bm in match.get('bookmakers', []):
                for mkt in bm.get('markets', []):
                    k = mkt.get('key')
                    if k in mkt_data:
                        for out in mkt.get('outcomes', []):
                            name, price = out.get('name'), float(out.get('price', 0.0))
                            point = float(out.get('point', 0.0)) if 'point' in out else None
                            if k == 'totals':
                                if point not in mkt_data['totals']: mkt_data['totals'][point] = {}
                                mkt_data['totals'][point][name] = price
                            elif k == 'spreads':
                                if point not in mkt_data['spreads']: mkt_data['spreads'][point] = {}
                                mkt_data['spreads'][point][name] = price
                            else:
                                mkt_data[k][name] = price

            st.subheader(f"🏟️ {home_team} vs {away_team}")

            # Control de xG
            c1, c2 = st.columns(2)
            l_h = c1.slider(f"xG Esperado {home_team}:", 0.2, 4.0, 1.35, 0.05)
            l_a = c2.slider(f"xG Esperado {away_team}:", 0.2, 4.0, 1.15, 0.05)

            # Simulación Montecarlo 50,000 iteraciones
            mc = InstitutionalMonteCarlo()
            results = mc.run(l_h, l_a)

            # Métricas Principales
            h2h_odds = [mkt_data['h2h'].get(home_team, 0), mkt_data['h2h'].get('Draw', 0), mkt_data['h2h'].get(away_team, 0)]
            _, z_h2h = ShinEngine.deoverround(h2h_odds)

            m1, m2, m3 = st.columns(3)
            m1.metric("Goles Totales Esperados", f"{l_h + l_a:.2f}")
            m2.metric("Sesgo Información Shin Z (1X2)", f"{z_h2h*100:.2f}%")
            max_idx = np.unravel_index(np.argmax(results['matrix']), results['matrix'].shape)
            m3.metric("Marcador Más Probable (Moda)", f"{max_idx[0]} - {max_idx[1]}")

            # EVALUACIÓN ESTRICTA DE VALOR (+EV)
            eval_rows = []
            
            def add_eval(mkt_name, odd, p_win, p_push=0.0):
                if odd <= 1.0: return
                # Cálculo de Esperanza Matemática (EV)
                ev = (p_win * odd + p_push) - 1.0
                p_eff = p_win / (1.0 - p_push) if p_push < 1.0 else 0.0
                b = odd - 1.0
                k_full = (b * p_eff - (1.0 - p_eff)) / b if b > 0 else 0.0
                stake = max(0.0, k_full * kelly_fraction * bankroll) if ev > 0.0 else 0.0
                
                # Regla Estricta: Solo es VALOR si EV > 0 (independiente de la probabilidad pura)
                if ev >= 0.03:
                    diag = "🔥 VALOR (+EV)"
                elif ev <= -0.05:
                    diag = "⚠️ TRAMPA (-EV)"
                else:
                    diag = "NEUTRO"

                eval_rows.append({
                    "Mercado": mkt_name, 
                    "Cuota Casa": f"{odd:.2f}",
                    "Prob. Modelo": f"{p_win*100:.1f}%", 
                    "Push": f"{p_push*100:.1f}%",
                    "EV (+/-)": f"{ev*100:+.2f}%", 
                    "Stake Sugerido": f"${stake:.2f}",
                    "Diagnóstico": diag
                })

            # Evaluaciones 1X2
            if home_team in mkt_data['h2h']: add_eval(f"Ganador {home_team}", mkt_data['h2h'][home_team], results['p_home'])
            if 'Draw' in mkt_data['h2h']: add_eval("Empate", mkt_data['h2h']['Draw'], results['p_draw'])
            if away_team in mkt_data['h2h']: add_eval(f"Ganador {away_team}", mkt_data['h2h'][away_team], results['p_away'])

            # Evaluaciones Totales (Over/Under)
            for pt, prices in mkt_data['totals'].items():
                p_push = results['p_exact_total'](pt) if pt.is_integer() else 0.0
                if 'Over' in prices: add_eval(f"Over {pt}", prices['Over'], results['p_over'](pt), p_push)
                if 'Under' in prices: add_eval(f"Under {pt}", prices['Under'], results['p_under'](pt), p_push)

            df_eval = pd.DataFrame(eval_rows)

            # Informe de Inteligencia
            st.subheader("🧠 Informe de Inteligencia Cuantitativa")
            value_bets = [r for r in eval_rows if "🔥" in r["Diagnóstico"]]
            traps = [r for r in eval_rows if "⚠️" in r["Diagnóstico"]]
            
            st.markdown(f"""
            <div class="report-box">
                <b>AUDITORÍA DE MERCADO Y PREDICCIÓN:</b><br>
                * <b>Resultado Más Probable:</b> {max_idx[0]} - {max_idx[1]} (Basado en la moda de las simulations de Montecarlo).<br>
                * <b>Eficiencia del Mercado (Shin Z):</b> El margen de información oculta detectado es de <b>{z_h2h*100:.2f}%</b>.<br>
                * <b>Oportunidades de Valor (+EV Real):</b> Se detectaron <b>{len(value_bets)}</b> selecciones donde la cuota ofrecida por la casa supera la probabilidad real del modelo.<br>
                * <b>Nota del Auditor:</b> Una apuesta con la probabilidad más alta no necesariamente es la mejor opción si la cuota está mal pagada por la casa. Confía en las selecciones marcadas con <b>+EV</b>.
            </div>
            """, unsafe_allow_html=True)

            st.subheader("📋 Matriz Multimercado Institucional")
            st.dataframe(df_eval, use_container_width=True)

            # Gráficos
            st.subheader("📈 Mapa de Calor & Distribución de Marcadores")
            g1, g2 = st.columns(2)

            with g1:
                fig, ax = plt.subplots(figsize=(5, 3.8))
                fig.patch.set_facecolor('#0b0f19')
                ax.set_facecolor('#0b0f19')
                ax.hist(results['totals'], bins=np.arange(0, 9)-0.5, rwidth=0.85, color='#06b6d4', edgecolor='#1e293b')
                ax.set_title("Distribución de Frecuencia de Goles", color="white", fontsize=10)
                ax.tick_params(colors='white')
                ax.grid(axis='y', linestyle='--', alpha=0.2)
                st.pyplot(fig)

            with g2:
                fig2, ax2 = plt.subplots(figsize=(5, 3.8))
                fig2.patch.set_facecolor('#0b0f19')
                ax2.set_facecolor('#0b0f19')
                sns.heatmap(results['matrix'], annot=True, fmt=".1f", cmap="mako", cbar=False, ax=ax2, annot_kws={"size": 8})
                ax2.set_title("Matriz de Marcadores Exactos (%)", color="white", fontsize=10)
                ax2.set_xlabel(f"Goles {away_team}", color="white", fontsize=9)
                ax2.set_ylabel(f"Goles {home_team}", color="white", fontsize=9)
                ax2.tick_params(colors='white')
                st.pyplot(fig2)
