import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# ==========================================
# 1. MODELOS DE DATOS
# ==========================================

@dataclass
class MarketOdds:
    match_name: str
    ah_line: float        # Ej: -0.75, 0.0, -1.25
    ah_home_odds: float   # Cuota del local en AH
    ah_away_odds: float   # Cuota del visitante en AH
    odds_1: float         # Cuota 1X2 Local
    odds_x: float         # Cuota 1X2 Empate
    odds_2: float         # Cuota 1X2 Visitante
    ou_line: float        # Línea de Totales (Ej: 2.5)
    over_odds: float      # Cuota Over
    under_odds: float     # Cuota Under
    btts_yes_odds: float  # Cuota BTTS - Sí
    btts_no_odds: float   # Cuota BTTS - No
    pinnacle_ah_odds: Optional[float] = None  # Cuota de referencia Sharp (opcional)

@dataclass
class ValidationResult:
    is_valid: bool
    status: str
    warnings: List[str]
    implied_probs: Dict[str, float]
    recommended_bet: Optional[str] = None
    recommended_stake_pct: float = 0.0

# ==========================================
# 2. MOTOR DE ANCLAJE Y PROBABILIDADES IMPLÍCITAS
# ==========================================

class AHBaselineEngine:
    """Calcula las probabilidades reales quitando el margen (overround) del mercado ancla (AH)."""
    
    @staticmethod
    def calculate_implied_probs(odds: MarketOdds) -> Dict[str, float]:
        # Desmargar el mercado 1X2
        raw_prob_1 = 1.0 / odds.odds_1
        raw_prob_x = 1.0 / odds.odds_x
        raw_prob_2 = 1.0 / odds.odds_2
        total_margin_1x2 = raw_prob_1 + raw_prob_x + raw_prob_2
        
        prob_1 = raw_prob_1 / total_margin_1x2
        prob_x = raw_prob_x / total_margin_1x2
        prob_2 = raw_prob_2 / total_margin_1x2

        # Desmargar BTTS
        raw_btts_yes = 1.0 / odds.btts_yes_odds
        raw_btts_no = 1.0 / odds.btts_no_odds
        margin_btts = raw_btts_yes + raw_btts_no
        prob_btts_yes = raw_btts_yes / margin_btts

        return {
            "prob_1": prob_1,
            "prob_x": prob_x,
            "prob_2": prob_2,
            "prob_btts_yes": prob_btts_yes,
            "margin_1x2_pct": (total_margin_1x2 - 1.0) * 100
        }

# ==========================================
# 3. MATRIZ DE CORRELACIÓN Y DETECTOR DE TRAMPAS
# ==========================================

class AntiTrapCrossMarketEngine:
    """Aplica las 3 Reglas de Coherencia Multimercado para evitar 'Value Traps'."""

    def __init__(self, kelly_fraction: float = 0.25):
        self.kelly_fraction = kelly_fraction  # Criterio de Kelly Fraccionado para mitigar varianza

    def evaluate(self, odds: MarketOdds) -> ValidationResult:
        warnings = []
        is_valid = True
        probs = AHBaselineEngine.calculate_implied_probs(odds)

        ah = odds.ah_line
        prob_1 = probs["prob_1"]
        btts_yes_prob = probs["prob_btts_yes"]

        # -------------------------------------------------------------
        # REGLA 1: Partido Parejo vs Expectativa de BTTS
        # -------------------------------------------------------------
        if abs(ah) <= 0.25:
            if odds.btts_yes_odds > 2.05:
                warnings.append(
                    f"⚠️ INCOHERENCIA: Hándicap ({ah}) indica partido equilibrado, "
                    f"pero BTTS-Sí cotiza alto ({odds.btts_yes_odds}). Posible sesgo de baja anotación."
                )
                is_valid = False

        # -------------------------------------------------------------
        # REGLA 2: Favoritismo Extremo vs Línea de Totales (Partido Embudo)
        # -------------------------------------------------------------
        if abs(ah) >= 1.25 and odds.ou_line < 2.25:
            warnings.append(
                f"⚠️ TRAMPA DE LIQUIDEZ: Favoritismo amplio (AH {ah}) pero línea O/U baja ({odds.ou_line}). "
                f"El mercado fuerza un resultado de marcador exacto exclusivo (2-0/1-0)."
            )
            is_valid = False

        # -------------------------------------------------------------
        # REGLA 3: Detección de 'Anchor Bias' (1X2 Desconectado de AH)
        # -------------------------------------------------------------
        # Si el modelo ve +EV en la victoria simple pero hay 'Odds Drift' o discrepancia
        expected_ev_1 = (prob_1 * odds.odds_1) - 1.0
        
        if expected_ev_1 > 0.05: # Si se detecta un +EV > 5%
            # Verificar si la cuota de Pinnacle respalda la selección
            if odds.pinnacle_ah_odds and odds.ah_home_odds > (odds.pinnacle_ah_odds * 1.04):
                warnings.append(
                    "⚠️ ALERTA DE FLUJO EN CONTRA: La cuota actual tiene un valor superior al de Pinnacle. "
                    "El dinero inteligente (Sharp Money) está vendiendo esta posición."
                )
                is_valid = False

        # -------------------------------------------------------------
        # RECOMENDACIÓN Y GESTIÓN DE CAPITAL (KELLY)
        # -------------------------------------------------------------
        recommended_bet = None
        stake_pct = 0.0

        if is_valid and expected_ev_1 > 0.03:
            # Fórmula de Kelly: f = (b*p - q) / b
            b = odds.odds_1 - 1.0
            p = prob_1
            q = 1.0 - p
            full_kelly = (b * p - q) / b
            
            if full_kelly > 0:
                stake_pct = round(full_kelly * self.kelly_fraction * 100, 2)
                recommended_bet = f"Victoria Local Simple (1) @ {odds.odds_1}"
                status = "✅ VALOR CONFIRMADO (Líneas Coherentes)"
            else:
                status = "🟡 SIN VALOR MATEMÁTICO SUFICIENTE"
        elif not is_valid:
            status = "🔴 OPORTUNIDAD RECHAZADA (Incoherencia de Mercado / Trampa)"
        else:
            status = "🟢 MERCADO COHERENTE (Sin +EV significativo para operar)"

        return ValidationResult(
            is_valid=is_valid,
            status=status,
            warnings=warnings,
            implied_probs=probs,
            recommended_bet=recommended_bet,
            recommended_stake_pct=stake_pct
        )

# ==========================================
# 4. INTERFAZ Y EJECUCIÓN PRÁCTICA
# ==========================================

def run_quant_app(matches: List[MarketOdds]):
    engine = AntiTrapCrossMarketEngine(kelly_fraction=0.25)
    
    print("=" * 75)
    print(" 🏛️  QUANTODDS 360 - MOTOR DE AUDITORÍA Y COHERENCIA MULTIMERCADO")
    print("=" * 75)

    for idx, match in enumerate(matches, 1):
        result = engine.evaluate(match)
        
        print(f"\n[{idx}] PARTIDO: {match.match_name}")
        print(f"    • Hándicap Asiático Base (AH): {match.ah_line}")
        print(f"    • Cuotas 1X2: [{match.odds_1} | {match.odds_x} | {match.odds_2}]")
        print(f"    • Totales / BTTS: O/U {match.ou_line} | BTTS SÍ @ {match.btts_yes_odds}")
        print(f"    • Dictamen: {result.status}")
        
        if result.recommended_bet:
            print(f"    🎯 SELECCIÓN SUGERIDA: {result.recommended_bet}")
            print(f"    💰 TAMAÑO DE APUESTA (Kelly 1/4): {result.recommended_stake_pct}% del Bankroll")

        if result.warnings:
            print("    ⚠️  DIAGNÓSTICO ANTI-TRAMPA:")
            for w in result.warnings:
                print(f"       {w}")
        print("-" * 75)

if __name__ == "__main__":
    # CASOS DE PRUEBA REALES

    partido_1 = MarketOdds(
        match_name="Arsenal vs. Brighton",
        ah_line=-0.75,
        ah_home_odds=1.91,
        ah_away_odds=1.99,
        odds_1=1.68,
        odds_x=3.90,
        odds_2=5.20,
        ou_line=2.75,
        over_odds=1.85,
        under_odds=2.00,
        btts_yes_odds=1.75,
        btts_no_odds=2.05,
        pinnacle_ah_odds=1.90 # Coherente
    )

    partido_2_trampa = MarketOdds(
        match_name="Atlético de Madrid vs. Getafe (Caso Partido Embudo)",
        ah_line=-1.25,
        ah_home_odds=1.85,
        ah_away_odds=2.05,
        odds_1=1.40,  # Cuota 1X2 baja tentando al público
        odds_x=4.50,
        odds_2=8.50,
        ou_line=2.00,  # INCOHERENCIA: Línea de goles muy baja para un AH -1.25
        over_odds=2.10,
        under_odds=1.75,
        btts_yes_odds=2.20,
        btts_no_odds=1.65
    )

    partido_3_flujo_contra = MarketOdds(
        match_name="Chelsea vs. West Ham (Caso Flujo Sharp en Contra)",
        ah_line=-0.5,
        ah_home_odds=2.05, # Cuota subiendo/inflada artificialmente
        ah_away_odds=1.85,
        odds_1=2.00,
        odds_x=3.50,
        odds_2=3.80,
        ou_line=2.5,
        over_odds=1.90,
        under_odds=1.90,
        btts_yes_odds=1.80,
        btts_no_odds=2.00,
        pinnacle_ah_odds=1.91 # Pinnacle tiene la cuota mucho más baja -> Sharp Money vendió al Chelsea
    )

    # Ejecutar la aplicación
    run_quant_app([partido_1, partido_2_trampa, partido_3_flujo_contra])
