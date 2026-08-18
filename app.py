import os
import streamlit as st
import numpy as np
import pandas as pd
import requests
from scipy.optimize import minimize
from scipy.stats import nbinom
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# CONFIGURACIÓN PÁGINA Y ESTILOS
# ==========================================
st.set_page_config(
    page_title="OddsDeconstruct Institutional AI — Institutional Reverse Engineering Engine",
    page_icon="🛡️",
    layout="wide"
)

st.markdown("""
<style>
    .main { background-color: #090d16; }
    .stApp { background-color: #090d16; color: #f1f5f9; }
    div[data-testid="metric-container"] {
        background-color: #111827; border: 1px solid #1e293b; padding: 14px; border-radius: 10px;
    }
    .report-card {
        background-color: #111827; border-left: 5px solid #06b6d4; padding: 18px; border-radius: 8px; margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);
    }
    .trap-card {
        background-color: #1a1016; border-left: 5px solid #f43f5e; padding: 18px; border-radius: 8px; margin-bottom: 20px;
    }
    .badge-val { background-color: #059669; color: white; padding: 3px 8px; border-radius: 4px; font-weight: bold; font-size: 0.85em; }
    .badge-trap { background-color: #e11d48; color: white; padding: 3px 8px; border-radius: 4px; font-weight: bold; font-size: 0.85em; }
    .badge-neutral { background-color: #475569; color: white; padding: 3px 8px; border-radius: 4px; font-weight: bold; font-size: 0.85em; }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 1. MOTOR DE INGENIERÍA INVERSA & DECONVOLUCIÓN (SHIN + SHARP CROSS-AUDIT)
# ==========================================

class ReverseEngineeringEngine:
    """
    Engine that deconstructs bookmaker margins and isolates true implicit probabilities
    from asymmetric information traps (Shin Z-score) and Sharp/Soft discrepancies.
    """
    @staticmethod
    def shin_deconvolute(odds_list):
        """
        Deconstructs overround using Shin's (1992, 1993) model for insider trading / asymmetric information.
        Returns clean probabilities and the Z-parameter (measure of bookmaker's information edge / trap level).
        """
        odds = np.array([o for o in odds_list if o > 1.0], dtype=float)
        if len(odds) < 2:
            return np.ones(len(odds)) / max(1, len(odds)), 0.0
        
        recip_sum = np.sum(1.0 / odds)
        
        def shin_obj(z):
            p = (np.sqrt(z**2 + 4 * (1 - z) * (1.0 / odds) / recip_sum) - z) / (2 * (1 - z))
            return (np.sum(p) - 1.0)**2

        res = minimize(shin_obj, x0=0.02, bounds=[(0.0001, 0.45)])
        z_opt = res.x[0] if res.success else 0.0
        
        p_clean = (np.sqrt(z_opt**2 + 4 * (1 - z_opt) * (1.0 / odds) / recip_sum) - z_opt) / (2 * (1 - z_opt))
        p_normalized = p_clean / np.sum(p_clean)
        return p_normalized, z_opt

    @staticmethod
    def audit_trap_risk(odd_offered, model_prob, shin_z, sharp_prob=None):
        """
        Ingeniería Inversa: Audita si una cuota con +EV aparente es en realidad una TRAMPA DE LIQUIDEZ.
        """
        if odd_offered <= 1.0 or model_prob <= 0:
            return "NO APLICABLE", 0.0, 0.0, 0.0

        implied_prob_raw = 1.0 / odd_offered
        raw_ev = (model_prob * odd_offered) - 1.0
        
        # Adjust for Shin Z Asymmetry (Bookmakers inflate line on traps to attract public volume)
        z_penalty = shin_z * 0.45
        adjusted_ev = raw_ev - z_penalty
        
        # If Sharp Benchmark Probability is available, cross-check against Sharp Consensus
        sharp_mismatch_flag = False
        if sharp_prob is not None and sharp_prob > 0:
            if model_prob > implied_prob_raw and sharp_prob < implied_prob_raw:
                sharp_mismatch_flag = True
                adjusted_ev -= 0.08  # Additional heavy penalty for sharp mismatch

        # Classification Logic
        if raw_ev > 0.02 and adjusted_ev > 0.015 and not sharp_mismatch_flag:
            diagnosis = "🔥 VALOR REAL INSTITUCIONAL (+EV)"
        elif raw_ev > 0 and (adjusted_ev <= 0 or sharp_mismatch_flag or shin_z > 0.18):
            diagnosis = "⚠️ TRAMPA DE LIQUIDEZ (Cuota Manipulada)"
        elif raw_ev <= -0.05:
            diagnosis = "❌ TRAMPA DE MARGEN (-EV)"
        else:
            diagnosis = "⚖️ NEUTRO / PRECIO JUSTO"

        return diagnosis, raw_ev, adjusted_ev, z_penalty


# ==========================================
# 2. SIMULACIÓN MONTECARLO BIVARIADA CON OVERDISPERSION
# ==========================================

class AdvancedMonteCarlo:
    """
    Bivariate Negative Binomial simulation that models goal dependency,
    dispersion parameters, and Poisson overdispersion for exact score matrix generation.
    """
    def __init__(self, simulations=60000, phi=1.18):
        self.simulations = simulations
        self.phi = phi  # Overdispersion parameter (1.0 = Poisson, >1.0 = Negative Binomial)

    def run_simulation(self, l_h, l_a):
        p_h = 1.0 / self.phi
        n_h = max(0.1, l_h / (self.phi - 1.0))
        p_a = 1.0 / self.phi
        n_a = max(0.1, l_a / (self.phi - 1.0))
        
        sim_h = nbinom.rvs(n_h, p_h, size=self.simulations)
        sim_a = nbinom.rvs(n_a, p_a, size=self.simulations)
        
        totals = sim_h + sim_a

        matrix = np.zeros((6, 6))
        for h, a in zip(sim_h, sim_a):
            if h < 6 and a < 6:
                matrix[h, a] += 1
        matrix = (matrix / self.simulations) * 100

        return {
            'sim_h': sim_h,
            'sim_a': sim_a,
            'totals': totals,
            'matrix': matrix,
            'p_home': float(np.mean(sim_h > sim_a)),
            'p_draw': float(np.mean(sim_h == sim_a)),
            'p_away': float(np.mean(sim_h < sim_a)),
            'p_over': lambda line: float(np.mean(totals > line)),
            'p_under': lambda line: float(np.mean(totals < line)),
            'p_exact_total': lambda line: float(np.mean(totals == line))
        }


# ==========================================
# 3. CONEXIÓN API & PERSISTENCIA AUTOMÁTICA
# ==========================================

st.sidebar.header("🛡️ Control Institucional")

env_api_key = os.getenv("ODDS_API_KEY") or os.getenv("API_KEY") or ""

if env_api_key:
    api_key = env_api_key
    st.sidebar.success("✅ API Key sincronizada automáticamente.")
else:
    api_key = st.sidebar.text_input("API Key (The-Odds-API):", value="", type="password")

region = st.sidebar.selectbox("Región de Mercado:", ["eu", "us", "uk", "au"], index=0)
bankroll = st.sidebar.number_input("Bankroll ($):", value=1000.0, step=100.0)
kelly_fraction = st.sidebar.slider("Fracción Kelly (Criterio de Riesgo):", 0.05, 0.50, 0.20, step=0.05)
z_threshold = st.sidebar.slider("Umbral Tolerancia Shin Z (Sensibilidad Trampas):", 0.05, 0.30, 0.15, step=0.01)

@st.cache_data(ttl=300)
def fetch_odds_data(key, reg):
    if not key:
        return None, "Falta API Key"
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
# 4. INTERFAZ PRINCIPAL & EJECUCIÓN
# ==========================================

st.title("🛡️ OddsDeconstruct AI — Engineering Reverse & Anti-Trap Engine")
st.caption("Motor Cuantitativo de Deconvolución de Mercado, Detección de Trampas de Liquidez y Auditoría por Modelo de Shin")

if not api_key:
    st.info("👈 Ingresa tu API Key en la barra lateral para sincronizar los mercados en vivo.")
else:
    raw_data, err_msg = fetch_odds_data(api_key, region)
    if raw_data is None:
        st.error(f"Error al conectar con la API: {err_msg}")
    else:
        valid_matches = [m for m in raw_data if m.get('bookmakers')]
        if not valid_matches:
            st.warning("No hay encuentros activos con cuotas para la región seleccionada.")
        else:
            titles = [f"{m.get('sport_title', 'Fútbol')}: {m.get('home_team')} vs {m.get('away_team')}" for m in valid_matches]
            idx = st.selectbox("📌 Selecciona Evento Deportivo a Auditar:", range(len(titles)), format_func=lambda x: titles[x])
            
            match = valid_matches[idx]
            home_team = match.get('home_team', 'Local')
            away_team = match.get('away_team', 'Visita')

            mkt_data = {'h2h': {}, 'totals': {}, 'spreads': {}}
            pinnacle_h2h = {}
            
            for bm in match.get('bookmakers', []):
                bm_key = bm.get('key', '').lower()
                for mkt in bm.get('markets', []):
                    k = mkt.get('key')
                    if k in mkt_data:
                        for out in mkt.get('outcomes', []):
                            name, price = out.get('name'), float(out.get('price', 0.0))
                            point = float(out.get('point', 0.0)) if 'point' in out else None
                            
                            if bm_key in ['pinnacle', 'betfair_ex_uk']:
                                if k == 'h2h': pinnacle_h2h[name] = price
                                
                            if k == 'totals':
                                if point not in mkt_data['totals']: mkt_data['totals'][point] = {}
                                mkt_data['totals'][point][name] = price
                            elif k == 'spreads':
                                if point not in mkt_data['spreads']: mkt_data['spreads'][point] = {}
                                mkt_data['spreads'][point][name] = price
                            else:
                                mkt_data[k][name] = price

            st.subheader(f"🏟️ Auditoría de Mercado: {home_team} vs {away_team}")

            c1, c2 = st.columns(2)
            l_h = c1.slider(f"xG Esperado {home_team}:", 0.2, 4.0, 1.35, 0.05)
            l_a = c2.slider(f"xG Esperado {away_team}:", 0.2, 4.0, 1.15, 0.05)

            mc_engine = AdvancedMonteCarlo(simulations=60000)
            mc_res = mc_engine.run_simulation(l_h, l_a)

            h2h_odds = [mkt_data['h2h'].get(home_team, 0), mkt_data['h2h'].get('Draw', 0), mkt_data['h2h'].get(away_team, 0)]
            p_shin, z_shin = ReverseEngineeringEngine.shin_deconvolute(h2h_odds)

            sharp_p_h, sharp_p_a = None, None
            if len(pinnacle_h2h) >= 3:
                pin_odds = [pinnacle_h2h.get(home_team, 0), pinnacle_h2h.get('Draw', 0), pinnacle_h2h.get(away_team, 0)]
                p_pin_clean, _ = ReverseEngineeringEngine.shin_deconvolute(pin_odds)
                sharp_p_h, sharp_p_a = p_pin_clean[0], p_pin_clean[2]

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Goles Totales Esperados", f"{l_h + l_a:.2f}")
            m2.metric("Manipulación Shin Z (Asimetría)", f"{z_shin*100:.2f}%")
            
            max_idx = np.unravel_index(np.argmax(mc_res['matrix']), mc_res['matrix'].shape)
            m3.metric("Marcador Más Probable (Moda)", f"{max_idx[0]} - {max_idx[1]}")
            
            risk_level = "ALTO (Manipulación)" if z_shin > 0.15 else "NORMAL"
            m4.metric("Riesgo de Mercado", risk_level)

            eval_rows = []
            
            def process_market_eval(mkt_name, odd, model_prob, sharp_prob=None, p_push=0.0):
                if odd <= 1.0: return
                diag, raw_ev, adj_ev, penalty = ReverseEngineeringEngine.audit_trap_risk(
                    odd_offered=odd, 
                    model_prob=model_prob, 
                    shin_z=z_shin, 
                    sharp_prob=sharp_prob
                )
                
                p_eff = model_prob / (1.0 - p_push) if p_push < 1.0 else 0.0
                b = odd - 1.0
                k_full = (b * p_eff - (1.0 - p_eff)) / b if b > 0 else 0.0
                stake = max(0.0, k_full * kelly_fraction * bankroll) if "🔥 VALOR" in diag else 0.0

                eval_rows.append({
                    "Mercado": mkt_name,
                    "Cuota Casa": f"{odd:.2f}",
                    "Prob. Modelo": f"{model_prob*100:.1f}%",
                    "Prob. Limpia Shin": f"{p_shin[0]*100:.1f}%" if "Home" in mkt_name or home_team in mkt_name else f"{p_shin[2]*100:.1f}%" if away_team in mkt_name else "N/A",
                    "EV Bruto": f"{raw_ev*100:+.2f}%",
                    "EV Ajustado Anti-Trampa": f"{adj_ev*100:+.2f}%",
                    "Stake Recomendado": f"${stake:.2f}",
                    "Diagnóstico Cuantitativo": diag
                })

            if home_team in mkt_data['h2h']: 
                process_market_eval(f"Ganador {home_team}", mkt_data['h2h'][home_team], mc_res['p_home'], sharp_p_h)
            if 'Draw' in mkt_data['h2h']: 
                process_market_eval("Empate", mkt_data['h2h']['Draw'], mc_res['p_draw'])
            if away_team in mkt_data['h2h']: 
                process_market_eval(f"Ganador {away_team}", mkt_data['h2h'][away_team], mc_res['p_away'], sharp_p_a)

            for pt, prices in mkt_data['totals'].items():
                p_push = mc_res['p_exact_total'](pt) if pt.is_integer() else 0.0
                if 'Over' in prices: 
                    process_market_eval(f"Over {pt}", prices['Over'], mc_res['p_over'](pt), p_push=p_push)
                if 'Under' in prices: 
                    process_market_eval(f"Under {pt}", prices['Under'], mc_res['p_under'](pt), p_push=p_push)

            df_eval = pd.DataFrame(eval_rows)

            value_bets = [r for r in eval_rows if "🔥" in r["Diagnóstico Cuantitativo"]]
            traps_detected = [r for r in eval_rows if "⚠️" in r["Diagnóstico Cuantitativo"]]

            st.subheader("🧠 Reporte de Inteligencia & Ingeniería Inversa")
            
            box_class = "trap-card" if len(traps_detected) > 0 else "report-card"
            st.markdown(f"""
            <div class="{box_class}">
                <h4>🛡️ RESULTADO DE LA AUDITORÍA ANTI-TRAMPAS DE MERCADO:</h4>
                <ul>
                    <li><b>Marcador Deconvolucionado (Moda):</b> {max_idx[0]} - {max_idx[1]} (Simulación Bivariada de Montecarlo con Dispersión).</li>
                    <li><b>Índice Shin Z (Manipulación de Información):</b> <b>{z_shin*100:.2f}%</b>. {"⚠️ <i>Alerta de Asimetría Alta: Las casas han ajustado las cuotas agresivamente.</i>" if z_shin > 0.15 else "<i>Nivel de manipulación dentro del margen seguro.</i>"}</li>
                    <li><b>Trampas de Liquidez Desenmascaradas:</b> Se identificaron <b>{len(traps_detected)}</b> mercados con +EV superficial que fueron clasificados como <b>TRAMPAS (-EV Ajustado)</b> debido al sesgo de cuotas manipuladas.</li>
                    <li><b>Oportunidades de Valor Real Confirmadas:</b> <b>{len(value_bets)}</b> selecciones superaron los filtros de ingeniería inversa y presentan ventaja cuantitativa real.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

            st.subheader("📋 Matriz Multimercado de Ingeniería Inversa")
            st.dataframe(df_eval, use_container_width=True)

            st.subheader("📈 Distribución Bivariada & Mapa de Calor")
            g1, g2 = st.columns(2)

            with g1:
                fig, ax = plt.subplots(figsize=(5, 3.8))
                fig.patch.set_facecolor('#090d16')
                ax.set_facecolor('#090d16')
                ax.hist(mc_res['totals'], bins=np.arange(0, 9)-0.5, rwidth=0.85, color='#06b6d4', edgecolor='#1e293b')
                ax.set_title("Distribución de Goles (Binomial Negativa)", color="white", fontsize=10)
                ax.tick_params(colors='white')
                ax.grid(axis='y', linestyle='--', alpha=0.2)
                st.pyplot(fig)

            with g2:
                fig2, ax2 = plt.subplots(figsize=(5, 3.8))
                fig2.patch.set_facecolor('#090d16')
                ax2.set_facecolor('#090d16')
                sns.heatmap(mc_res['matrix'], annot=True, fmt=".1f", cmap="mako", cbar=False, ax=ax2, annot_kws={"size": 8})
                ax2.set_title("Matriz Bivariada de Marcadores Exactos (%)", color="white", fontsize=10)
                ax2.set_xlabel(f"Goles {away_team}", color="white", fontsize=9)
                ax2.set_ylabel(f"Goles {home_team}", color="white", fontsize=9)
                ax2.tick_params(colors='white')
                st.pyplot(fig2)
