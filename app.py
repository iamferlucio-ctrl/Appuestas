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
# 1. MOTOR MATEMÁTICO MULTIMERCADO
# ==========================================

class ShinEngine:
    @staticmethod
    def deoverround(odds_list):
        odds = np.array(odds_list, dtype=float)
        odds = odds[odds > 1.0]
        if len(odds) < 2:
            return np.ones(len(odds)) / len(odds), 0.0
        
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
        diff = sim_h - sim_a

        # Matriz 6x6
        matrix = np.zeros((6, 6))
        for h, a in zip(sim_h, sim_a):
            if h < 6 and a < 6:
                matrix[h, a] += 1
        matrix = (matrix / self.simulations) * 100

        return {
            'totals': totals, 'diff': diff, 'matrix': matrix,
            # Probabilidades 1X2
            'p_home': np.mean(sim_h > sim_a),
            'p_draw': np.mean(sim_h == sim_a),
            'p_away': np.mean(sim_h < sim_a),
            # BTTS
            'p_btts_yes': np.mean((sim_h > 0) & (sim_a > 0)),
            'p_btts_no': np.mean((sim_h == 0) | (sim_a == 0)),
            # Totales
            'p_over': lambda line: np.mean(totals > line),
            'p_under': lambda line: np.mean(totals < line),
            'p_exact_total': lambda line: np.mean(totals == line),
            # Hándicap Asiático
            'p_ah_home': lambda hcap: np.mean(diff + hcap > 0),
            'p_ah_away': lambda hcap: np.mean(diff + hcap < 0),
            'p_ah_push': lambda hcap: np.mean(diff + hcap == 0)
        }

# ==========================================
# 2. CONFIGURACIÓN Y API MULTI-MERCADO
# ==========================================

st.sidebar.header("🔑 Control Cuantitativo")
api_key = st.sidebar.text_input("API Key (The-Odds-API):", value="", type="password")
region = st.sidebar.selectbox("Región Mercado:", ["eu", "us", "uk", "au"], index=0)
bankroll = st.sidebar.number_input("Capital Bankroll ($):", value=1000.0, step=100.0)
kelly_fraction = st.sidebar.slider("Fracción Kelly (Riesgo):", 0.05, 0.50, 0.25, step=0.05)

@st.cache_data(ttl=300)
def fetch_all_markets(key, reg):
    if not key: return None
    url = f"https://api.the-odds-api.com/v4/sports/soccer/odds/?apiKey={key}&regions={reg}&markets=h2h,totals,btts,spreads&oddsFormat=decimal"
    try:
        r = requests.get(url, timeout=10)
        return r.json() if r.status_code == 200 else None
    except: return None

# ==========================================
# 3. INTERFAZ E INFORMES DE ALTO NIVEL
# ==========================================

st.title("⚡ OddsDeconstruct AI — Quant & Institutional Platform")

if not api_key:
    st.info("👈 Ingresa tu API Key en la barra lateral para sincronizar los mercados globales.")
else:
    raw_data = fetch_all_markets(api_key, region)
    if not raw_data or not isinstance(raw_data, list):
        st.error("Error al conectar con el feed de cuotas multimercado.")
    else:
        valid_matches = [m for m in raw_data if m.get('bookmakers')]
        if not valid_matches:
            st.warning("No hay encuentros con datos multimercado activos.")
        else:
            titles = [f"{m.get('sport_title', 'Fútbol')}: {m.get('home_team')} vs {m.get('away_team')}" for m in valid_matches]
            idx = st.selectbox("📌 Selecciona Evento Deportivo:", range(len(titles)), format_func=lambda x: titles[x])
            
            match = valid_matches[idx]
            home_team = match.get('home_team', 'Local')
            away_team = match.get('away_team', 'Visita')

            # Extracción Multimercado
            mkt_data = {'h2h': {}, 'totals': {}, 'btts': {}, 'spreads': {}}
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

            # Simulación Montecarlo 50k
            mc = InstitutionalMonteCarlo()
            results = mc.run(l_h, l_a)

            # Métricas
            h2h_odds = [mkt_data['h2h'].get(home_team, 0), mkt_data['h2h'].get('Draw', 0), mkt_data['h2h'].get(away_team, 0)]
            _, z_h2h = ShinEngine.deoverround([o for o in h2h_odds if o > 1.0])

            m1, m2, m3 = st.columns(3)
            m1.metric("Goles Totales Esperados", f"{l_h + l_a:.2f}")
            m2.metric("Sesgo Información Shin Z (1X2)", f"{z_h2h*100:.2f}%")
            m3.metric("Marcador Más Probable", f"{np.unravel_index(np.argmax(results['matrix']), results['matrix'].shape)}")

            # Construcción de Opciones Multimercado
            eval_rows = []
            
            def add_eval(mkt_name, odd, p_win, p_push=0.0):
                if odd <= 1.0: return
                ev = (p_win * odd + p_push) - 1.0
                p_eff = p_win / (1.0 - p_push) if p_push < 1.0 else 0.0
                b = odd - 1.0
                k_full = (b * p_eff - (1.0 - p_eff)) / b if b > 0 else 0.0
                stake = max(0.0, k_full * kelly_fraction * bankroll) if ev > 0.0 else 0.0
                
                eval_rows.append({
                    "Mercado": mkt_name, "Cuota Casa": f"{odd:.2f}",
                    "Prob. Modelo": f"{p_win*100:.1f}%", "Push": f"{p_push*100:.1f}%",
                    "EV (+/-)": f"{ev*100:+.2f}%", "Stake Sugerido": f"${stake:.2f}",
                    "Diagnóstico": "🔥 VALOR (+EV)" if ev > 0.04 else ("⚠️ TRAMPA" if ev < -0.08 else "NEUTRO")
                })

            # Evaluaciones 1X2
            if home_team in mkt_data['h2h']: add_eval(f"Ganador {home_team}", mkt_data['h2h'][home_team], results['p_home'])
            if 'Draw' in mkt_data['h2h']: add_eval("Empate", mkt_data['h2h']['Draw'], results['p_draw'])
            if away_team in mkt_data['h2h']: add_eval(f"Ganador {away_team}", mkt_data['h2h'][away_team], results['p_away'])

            # Evaluaciones Doble Oportunidad
            if 'Draw' in mkt_data['h2h']:
                add_eval(f"1X ({home_team} o Empate)", mkt_data['h2h'].get('1X', 1.0), results['p_home'] + results['p_draw'])
                add_eval(f"X2 (Empate o {away_team})", mkt_data['h2h'].get('X2', 1.0), results['p_away'] + results['p_draw'])

            # Evaluaciones BTTS
            if 'Yes' in mkt_data['btts']: add_eval("Ambos Marcan: SÍ", mkt_data['btts']['Yes'], results['p_btts_yes'])
            if 'No' in mkt_data['btts']: add_eval("Ambos Marcan: NO", mkt_data['btts']['No'], results['p_btts_no'])

            # Evaluaciones Totales
            for pt, prices in mkt_data['totals'].items():
                p_push = results['p_exact_total'](pt) if pt.is_integer() else 0.0
                if 'Over' in prices: add_eval(f"Over {pt}", prices['Over'], results['p_over'](pt), p_push)
                if 'Under' in prices: add_eval(f"Under {pt}", prices['Under'], results['p_under'](pt), p_push)

            df_eval = pd.DataFrame(eval_rows)

            # INFORME DE INTELIGENCIA CUANTITATIVA (Interpretación Ejecutiva)
            st.subheader("🧠 Informe de Inteligencia Cuantitativa")
            
            value_bets = [r for r in eval_rows if "🔥" in r["Diagnóstico"]]
            traps = [r for r in eval_rows if "⚠️" in r["Diagnóstico"]]
            
            report_html = f"""
            <div class="report-box">
                <b>RESUMEN EJECUTIVO DE MERCADO:</b><br>
                * <b>Eficiencia del Mercado (Shin Z):</b> El margen de información oculta detectado en 1X2 es de <b>{z_h2h*100:.2f}%</b>. {"Alta eficiencia institucional." if z_h2h < 0.03 else "Inconsistencias detectadas en el Overround de la casa."}<br>
                * <b>Oportunidades de Valor (+EV):</b> Se detectaron <b>{len(value_bets)}</b> selecciones con ventaja estadística sobre la casa de apuestas.<br>
                * <b>Sesgo de Mercado:</b> La casa muestra una sobreprotección en selecciones populares. Hay <b>{len(traps)}</b> mercados catalogados como trampas de valor negativo.
            </div>
            """
            st.markdown(report_html, unsafe_allow_html=True)

            st.subheader("📋 Matriz Multimercado Institucional")
            st.dataframe(df_eval, use_container_width=True)

            # Visualizaciones
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
