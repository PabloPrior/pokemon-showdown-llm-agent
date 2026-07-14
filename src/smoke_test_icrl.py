"""
Smoke test del modulo de memoria estructurada (ICRL).

Objetivo: verificar EN VIVO, con 3 batallas, que:
  1) En el TURNO 1 NO se inyecta memoria (no hay turno previo).
  2) Desde el TURNO 2 aparece el bloque "--- MEMORIA DE TURNOS PREVIOS ---"
     con feedback coherente (dano observado, inmunidad, estado, cambio del rival).
  3) No se rompe nada (sin excepciones; el fallback no se dispara por el ICRL).
  4) El coste en tokens sube solo de forma moderada respecto al baseline.

Requiere el servidor local de Showdown levantado:
    node pokemon-showdown start --no-security

No modifica llm_agent.py: instrumenta el agente con una subclase de traza.
"""

import asyncio
from poke_env import AccountConfiguration
from main import HeuristicPlayer
from llm_agent import SimpleLLMPlayer

# --- Configuracion del smoke test ---------------------------------
PROVIDER = "openai"            # gpt-4o-mini es barato y rapido para la prueba
MODEL    = "gpt-4o-mini"
N_BATTLES = 3
# Para usar Llama por OpenRouter en su lugar, comenta las 2 lineas de arriba y descomenta:
# PROVIDER = "openrouter"
# MODEL    = "meta-llama/llama-3.3-70b-instruct"
# ------------------------------------------------------------------


class TracingLLMPlayer(SimpleLLMPlayer):
    """Subclase que imprime el bloque de memoria ICRL cuando se inyecta (no vacio)."""

    def _icrl_feedback(self, battle):
        block = super()._icrl_feedback(battle)
        if block.strip():
            tag = battle.battle_tag[-5:]
            print(f"\n  [batalla ...{tag} | TURNO {battle.turn}] >>> MEMORIA INYECTADA EN EL PROMPT:")
            for linea in block.strip().splitlines():
                print(f"      {linea.strip()}")
        return block


async def smoke():
    print("=" * 64)
    print(f"SMOKE TEST ICRL  |  {MODEL}  vs  HeuristicPlayer  |  {N_BATTLES} batallas")
    print("=" * 64)

    llm_bot = TracingLLMPlayer(
        provider=PROVIDER,
        model_name=MODEL,
        use_icrl=True,                       # <--- modulo de memoria ACTIVADO
        account_configuration=AccountConfiguration("SmokeICRL", None),
        battle_format="gen5randombattle",
        max_concurrent_battles=1,            # secuencial -> traza legible, sin interleaving
    )
    opponent = HeuristicPlayer(
        account_configuration=AccountConfiguration("SmokeHeur", None),
        battle_format="gen5randombattle",
        max_concurrent_battles=1,
    )

    await llm_bot.battle_against(opponent, n_battles=N_BATTLES)

    # --- Resumen de verificacion ---
    total_tokens = sum(llm_bot.token_usage.values())
    total_actions = sum(len(a) for a in llm_bot.action_log.values())
    total_fallbacks = sum(a.count("fallback") for a in llm_bot.action_log.values())
    tokens_por_decision = total_tokens / total_actions if total_actions else 0

    print("\n" + "=" * 64)
    print("RESUMEN DEL SMOKE TEST")
    print("=" * 64)
    print(f"Victorias: {llm_bot.n_won_battles}/{N_BATTLES}")
    print(f"Decisiones totales: {total_actions}")
    print(f"Fallbacks (deberia ser ~0): {total_fallbacks}")
    print(f"Tokens totales: {total_tokens}  |  por decision: {round(tokens_por_decision)}")
    print(f"Batallas con memoria registrada: {sum(1 for m in llm_bot.icrl_mem.values() if m)}/{N_BATTLES}")
    print("=" * 64)
    print("REVISA ARRIBA: que el TURNO 1 no tenga memoria, que desde el 2 aparezca,")
    print("y que los % de dano y las inmunidades sean coherentes con la partida.")


if __name__ == "__main__":
    asyncio.run(smoke())