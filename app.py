import os
import sys
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# ==========================================
# 1. MODELO DE DATOS INSTITUCIONAL
# ==========================================

@dataclass
class MarketData:
    match_name: str
    ah_line: float          # Línea AH (ej. -0.75, 0.0, -1.25)
    ah_home_odds: float     # Cuota AH Local
    ah_away_odds: float     # Cuota AH Visitante
    odds_1: float           # Cuota 1X2 Local
    odds_x: float           # Cuota 1X2 Empate
    odds_2: float           # Cuota 1X2 Visitante
    ou_line: float          # Línea Over/Under (ej. 2.5)
    over_odds: float        # Cuota Over
    under_odds: float       # Cuota Under
    btts_yes_odds: float    # Cuota BTTS - Sí
    btts_no_odds: float     # Cuota BTTS - No
    pinnacle_ah_home: Optional[float] = None  # Referencia Sharp (Pinnacle)

@dataclass
class AuditReport:
    match_name: str
    is_approved: bool
    status_label: str
    warnings: List[str]
    true_prob_1: float
    raw_implied_1: float
    adjusted_ev_pct: float
    recommended_bet: Optional[str]
    stake_kelly_pct: float

# ==========================================
# 2. MOTOR MATEMÁTICO: DERIVACIÓN DESDE EL AH
# ==========================================

class QuantitativeEngine:
    """
    Deriva las probabilidades reales desde la Línea de Verdad (Hándicap Asiático)
    y las contrasta contra el mercado 1X2 para encontrar +EV real.
    """
    
    @staticmethod
    def get_clean_ah_probs(ah_home_odds: float, ah_away_odds: float) -> Tuple[float, float]:
        """Elimina el overround del Hándicap Asiático usando Shin/Inversa Implícita."""
        raw_h = 1.0 / ah_home_odds
        raw_a = 1.0 / ah_away_odds
        margin = raw_h + raw_a
        return (raw_h / margin, raw_a / margin)

    @classmethod
    def estimate_true_1x2_from_ah(cls, market: MarketData) -> Dict[str, float]:
        """
        Deriva p(1), p(X), p(2) basándose en el Hándicap Asiático y la expectativa de goles.
        """
        p_ah_home, p_ah_away = cls.get_clean_ah_probs(market.ah_home_odds, market.ah_away_odds)
        
        # Factor de ajuste según la profundidad de la línea AH
        ah = market.ah_line
        
        if ah == 0.0:
            # Partido parejo: la probabilidad del empate es máxima (~28-30%)
            prob_x = 0.29
            rem = 1.0 - prob_x
            prob_1 = p_ah_home * rem
            prob_2 = p_ah_away * rem
        elif ah < 0:
            # Local Favorito: la probabilidad de empate decae suavemente
            prob_x = max(0.18, 0.28 - (abs(ah) * 0.06))
            rem = 1.0 - prob_x
            # Ajuste de cobertura según línea (-0.25, -0.5, -0.75, -1.0, etc.)
            prob_1 = p_ah_home * (1.0 + (abs(ah) * 0.12))
            prob_1 = min(prob_1, rem - 0.05)
            prob_2 = 1.0 - prob_1 - prob_x
        else:
            # Visitante Favorito
            prob_x = max(0.18, 0.28 - (abs(ah) * 0.06))
            rem = 1.0 - prob_x
            prob_2 = p_ah_away * (1.0 + (abs(ah) * 0.12))
            prob_2 = min(prob_2, rem - 0.05)
            prob_1 = 1.0 - prob_2 - prob_x

        return {"prob_1": prob_1, "prob_x": prob_x, "prob_2": prob_2}

# ==========================================
# 3. FILTRO AUDITOR ANTI-TRAMPAS MULTIMERCADO
# ==========================================

class MultiMarketAuditor:
    def __init__(self, kelly_fraction: float = 0.20):
        self.kelly_fraction = kelly_fraction

    def audit(self, market: MarketData) -> AuditReport:
        warnings = []
        is_approved = True

        # 1. Obtener probabilidades verdaderas derivadas del AH
        true_probs = QuantitativeEngine.estimate_true_1x2_from_ah(market)
        prob_1 = true_probs["prob_1"]
        raw_implied_1 = 1.0 / market.odds_1

        # 2. EV Bruto real
        raw_ev = (prob_1 * market.odds_1) - 1.0

        # --- REGLA 1: Partido Parejo vs BTTS ---
        if abs(market.ah_line) <= 0.25 and market.btts_yes_odds > 2.05:
            warnings.append(
                f"⚠️ INCOHERENCIA DE MERCADO: AH indica partido equilibrado ({market.ah_line}), "
                f"pero BTTS-Sí cotiza excesivamente alto ({market.btts_yes_odds})."
            )
            is_approved = False

        # --- REGLA 2: Partido Embudo (Favorito Fuerte + Bajo Totales) ---
        if abs(market.ah_line) >= 1.25 and market.ou_line < 2.25:
            warnings.append(
                f"⚠️ TRAMPA DE LIQUIDEZ: Favoritismo amplio (AH {market.ah_line}) "
                f"incompatible con Totales bajos (O/U {market.ou_line}). Marcador exacto forzado."
            )
            is_approved = False

        # --- REGLA 3: Auditoría por Flujo Sharp (Pinnacle) ---
        if market.pinnacle_ah_home:
            if market.ah_home_odds > (market.pinnacle_ah_home * 1.04):
                warnings.append(
                    f"⚠️ ALERTA DE FLUJO SHARP EN CONTRA: La cuota ofrecida ({market.ah_home_odds}) "
                    f"está inflada respecto a Pinnacle ({market.pinnacle_ah_home}). Dinero inteligente vendiendo."
                )
                is_approved = False

        # --- RECOMENDACIÓN Y KELLY FRACCIONADO ---
        recommended_bet = None
        stake_pct = 0.0

        if is_approved and raw_ev > 0.025:
            b = market.odds_1 - 1.0
            p = prob_1
            q = 1.0 - p
            full_kelly = (b * p - q) / b
            
            if full_kelly > 0:
                stake_pct = round(full_kelly * self.kelly_fraction * 100, 2)
                recommended_bet = f"Victoria Local (1) @ {market.odds_1}"
                status_label = "✅ VALOR INSTITUCIONAL CONFIRMADO"
            else:
                status_label = "🟡 COHERENTE SIN VALOR MATEMÁTICO"
        elif not is_approved:
            status_label = "🔴 OPORTUNIDAD RECHAZADA (Trampa de Mercado / Incoherencia)"
        else:
            status_label = "🟢 PRECIO JUSTO (Sin EV relevante)"

        return AuditReport(
            match_name=market.match_name,
            is_approved=is_approved,
            status_label=status_label,
            warnings=warnings,
            true_prob_1=prob_1,
            raw_implied_1=raw_implied_1,
            adjusted_ev_pct=raw_ev * 100,
            recommended_bet=recommended_bet,
            stake_kelly_pct=stake_pct
        )

# ==========================================
# 4. EJECUCIÓN CONSOLA / PRUEBA AUTOMÁTICA
# ==========================================

def run_cli_audit():
    test_suite = [
        MarketData(
            match_name="Arsenal vs. Brighton (Caso Valor Real)",
            ah_line=-0.75, ah_home_odds=1.91, ah_away_odds=1.99,
            odds_1=1.85, odds_x=3.80, odds_2=4.50,
            ou_line=2.75, over_odds=1.85, under_odds=2.00,
            btts_yes_odds=1.75, btts_no_odds=2.05,
            pinnacle_ah_home=1.90
        ),
        MarketData(
            match_name="Atlético Madrid vs. Getafe (Caso Partido Embudo)",
            ah_line=-1.25, ah_home_odds=1.85, ah_away_odds=2.05,
            odds_1=1.40, odds_x=4.50, odds_2=8.50,
            ou_line=2.00, over_odds=2.10, under_odds=1.75,
            btts_yes_odds=2.20, btts_no_odds=1.65
        ),
        MarketData(
            match_name="Chelsea vs. West Ham (Caso Flujo Sharp en Contra)",
            ah_line=-0.50, ah_home_odds=2.08, ah_away_odds=1.82,
            odds_1=2.00, odds_x=3.50, odds_2=3.80,
            ou_line=2.50, over_odds=1.90, under_odds=1.90,
            btts_yes_odds=1.80, btts_no_odds=2.00,
            pinnacle_ah_home=1.91
        )
    ]

    auditor = MultiMarketAuditor(kelly_fraction=0.20)
    print("=" * 80)
    print(" 🏛️  AUDITORÍA CUANTITATIVA MULTIMERCADO — ODDSDECONSTRUCT AI")
    print("=" * 80)

    for m in test_suite:
        rep = auditor.audit(m)
        print(f"\n📌 EVENTO: {rep.match_name}")
        print(f"   • AH Base: {m.ah_line} | Cuota 1X2 Evaluada: {m.odds_1}")
        print(f"   • Prob. Implícita Casa: {rep.raw_implied_1*100:.1f}% | Prob. Real (Derivada AH): {rep.true_prob_1*100:.1f}%")
        print(f"   • EV Calculado: {rep.adjusted_ev_pct:+.2f}%")
        print(f"   • DICTAMEN: {rep.status_label}")
        if rep.recommended_bet:
            print(f"   🎯 RECOMENDACIÓN: {rep.recommended_bet} (Stake Kelly: {rep.stake_kelly_pct}% Bankroll)")
        if rep.warnings:
            for w in rep.warnings:
                print(f"   {w}")
        print("-" * 80)

# ==========================================
# 5. MÓDULO STREAMLIT (INTERFAZ WEB)
# ==========================================

def run_streamlit_app():
    import streamlit as st
    st.set_page_config(page_title="QuantOdds 360 AI", page_icon="🛡️", layout="wide")
    
    st.title("🛡️ QuantOdds 360 — Motor de Auditoría Anti-Trampas")
    st.caption("Deconvolución de Hándicap Asiático, Verificación Cruzada Multimercado y Filtro de Flujo Sharp")

    st.sidebar.header("⚙️ Parámetros del Partido")
    match_name = st.sidebar.text_input("Nombre del Partido", "Arsenal vs. Brighton")
    
    st.sidebar.subheader("1. Mercado Ancla (Hándicap Asiático)")
    ah_line = st.sidebar.number_input("Línea AH (ej. -0.75)", value=-0.75, step=0.25)
    ah_home_odds = st.sidebar.number_input("Cuota AH Local", value=1.91, step=0.01)
    ah_away_odds = st.sidebar.number_input("Cuota AH Visitante", value=1.99, step=0.01)
    
    st.sidebar.subheader("2. Mercado 1X2")
    odds_1 = st.sidebar.number_input("Cuota Local (1)", value=1.85, step=0.01)
    odds_x = st.sidebar.number_input("Cuota Empate (X)", value=3.80, step=0.01)
    odds_2 = st.sidebar.number_input("Cuota Visitante (2)", value=4.50, step=0.01)

    st.sidebar.subheader("3. Mercado Totales & BTTS")
    ou_line = st.sidebar.number_input("Línea Totales (O/U)", value=2.75, step=0.25)
    btts_yes = st.sidebar.number_input("Cuota BTTS Sí", value=1.75, step=0.01)
    btts_no = st.sidebar.number_input("Cuota BTTS No", value=2.05, step=0.01)

    st.sidebar.subheader("4. Referencia Sharp (Opcional)")
    pinnacle_ah = st.sidebar.number_input("Cuota AH Local Pinnacle", value=1.90, step=0.01)

    market = MarketData(
        match_name=match_name,
        ah_line=ah_line, ah_home_odds=ah_home_odds, ah_away_odds=ah_away_odds,
        odds_1=odds_1, odds_x=odds_x, odds_2=odds_2,
        ou_line=ou_line, over_odds=1.85, under_odds=2.00,
        btts_yes_odds=btts_yes, btts_no_odds=btts_no,
        pinnacle_ah_home=pinnacle_ah if pinnacle_ah > 1.0 else None
    )

    auditor = MultiMarketAuditor()
    report = auditor.audit(market)

    c1, c2, c3 = st.columns(3)
    c1.metric("Probabilidad Real (Derivada AH)", f"{report.true_prob_1*100:.1f}%")
    c2.metric("Probabilidad Implícita Casa", f"{report.raw_implied_1*100:.1f}%")
    c3.metric("Beneficio Esperado (EV)", f"{report.adjusted_ev_pct:+.2f}%")

    st.subheader(f"Dictamen: {report.status_label}")

    if report.warnings:
        st.error("🚨 ALERTAS ANTI-TRAMPA DETECTADAS:")
        for w in report.warnings:
            st.write(f"- {w}")

    if report.recommended_bet:
        st.success(f"🎯 RECOMENDACIÓN: {report.recommended_bet} — Stake Sugerido: {report.stake_kelly_pct}% del Bankroll")

# Detectar el modo de ejecución
if __name__ == "__main__":
    # Si se ejecuta mediante Streamlit CLI
    if "streamlit" in sys.modules or os.environ.get("SERVER_PORT"):
        run_streamlit_app()
    else:
        # Si se ejecuta directamente con `python app.py`
        try:
            import streamlit
            # Si Streamlit está disponible pero fue invocado con python, ejecutamos la CLI y avisamos
            run_cli_audit()
            print("\n💡 Para abrir la Interfaz Web Gráfica, ejecuta en tu terminal:")
            print("   streamlit run app.py\n")
        except ImportError:
            run_cli_audit()
