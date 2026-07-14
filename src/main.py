import asyncio
import time
import csv
import os
from datetime import datetime
from llm_agent import SimpleLLMPlayer
from poke_env.player import Player
from poke_env import AccountConfiguration
import math


# ==================== CONFIGURACION DE LA TANDA ====================
# Ablacion ICRL: ejecuta este script 4 veces cambiando estas lineas.
#   1) gpt-4o-mini                         SIN ICRL  -> PROVIDER="openai",     MODEL="gpt-4o-mini",                       USE_ICRL=False
#   2) gpt-4o-mini                         CON ICRL  -> PROVIDER="openai",     MODEL="gpt-4o-mini",                       USE_ICRL=True
#   3) meta-llama/llama-3.3-70b-instruct   SIN ICRL  -> PROVIDER="openrouter", MODEL="meta-llama/llama-3.3-70b-instruct", USE_ICRL=False
#   4) meta-llama/llama-3.3-70b-instruct   CON ICRL  -> PROVIDER="openrouter", MODEL="meta-llama/llama-3.3-70b-instruct", USE_ICRL=True
PROVIDER  = "openrouter"          # "openai" | "openrouter"
MODEL     = "meta-llama/llama-3.3-70b-instruct"     # "gpt-4o-mini" | "meta-llama/llama-3.3-70b-instruct"
USE_ICRL  = True             # False = sin memoria | True = con memoria (ICRL)
N_MATCHES = 100
# ==================================================================


def wilson_ci(wins, n, z=1.96):
    """Intervalo de confianza de Wilson (95%) para una proporción (winrate)."""
    if n == 0:
        return (0.0, 0.0)
    p = wins / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (max(0.0, center - margin), min(1.0, center + margin))


class MaxDamagePlayer(Player):
    def choose_move(self, battle):
        # Si tiene ataques, usa siempre el de mayor daño base
        if battle.available_moves:
            best_move = max(battle.available_moves, key=lambda m: m.base_power)
            return self.create_order(best_move)
        # Si solo puede cambiar (Force Switch), elige aleatorio
        return self.choose_random_move(battle)


class HeuristicPlayer(Player):
    """
    Oponente heurístico, consciente de tipos, para hacer de benchmark del agente LLM.

    Decisión por turno:
      1. Si está obligado a cambiar -> elige el mejor relevo segun el matchup.
      2. Si el matchup actual es malo y hay un relevo seguro -> cambia.
      3. En caso contrario -> usa el movimiento de mayor daño estimado.

    Daño estimado = poder_base * STAB * mult_tipo * precision * (atk/def).
    No reentrena nada; es determinista salvo desempates.
    """

    STAB = 1.5  # bonus por usar un movimiento del mismo tipo que el atacante

    # ----------------------- utilidades -----------------------

    @staticmethod
    def _accuracy(move):
        # poke-env devuelve la precision como fraccion (0.9), entero (90) o True (no falla)
        acc = getattr(move, "accuracy", 1.0)
        if isinstance(acc, bool) or not isinstance(acc, (int, float)):
            return 1.0
        return acc / 100.0 if acc > 1 else acc

    def _estimate_damage(self, move, attacker, defender):
        """Puntuacion de daño esperado de 'move' (de attacker) contra defender."""
        if move.category.name == "STATUS":
            return 0.0
        stab = self.STAB if move.type in attacker.types else 1.0
        type_mult = defender.damage_multiplier(move)
        if move.category.name == "PHYSICAL":
            ratio = attacker.base_stats.get("atk", 100) / max(defender.base_stats.get("def", 100), 1)
        else:  # SPECIAL
            ratio = attacker.base_stats.get("spa", 100) / max(defender.base_stats.get("spd", 100), 1)
        return move.base_power * stab * type_mult * self._accuracy(move) * ratio

    @staticmethod
    def _incoming_multiplier(attacker, defender):
        """Peor multiplicador de tipo que 'attacker' (por STAB) puede aplicar a 'defender'."""
        mult = 1.0
        for t in attacker.types:
            if t is not None:
                mult = max(mult, defender.damage_multiplier(t))
        return mult

    @staticmethod
    def _best_move_multiplier(attacker, defender):
        """Mejor multiplicador de tipo que 'attacker' puede conseguir atacando a 'defender'."""
        best = 0.0
        for m in attacker.moves.values():
            if m.category.name != "STATUS" and m.base_power > 0:
                best = max(best, defender.damage_multiplier(m))
        return best

    def _best_switch(self, battle, opponent):
        """Relevo con menor amenaza recibida; desempata por pegada y por HP."""
        switches = battle.available_switches
        if not switches:
            return None
        if opponent is None:
            return switches[0]
        return min(
            switches,
            key=lambda p: (
                self._incoming_multiplier(opponent, p),      # 1) que resista al rival
                -self._best_move_multiplier(p, opponent),    # 2) que le pegue fuerte
                -p.current_hp_fraction,                      # 3) que llegue sano
            ),
        )

    # ----------------------- decision -----------------------

    def choose_move(self, battle):
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon

        # Caso limite: no hay nada legal
        if not battle.available_moves and not battle.available_switches:
            return self.choose_default_move(battle)

        # (1) Cambio obligatorio
        if not battle.available_moves and battle.available_switches:
            pivot = self._best_switch(battle, opponent)
            return self.create_order(pivot) if pivot else self.choose_random_move(battle)

        # (2) Cambio voluntario solo si el matchup es claramente malo
        if battle.available_switches and opponent is not None:
            threat = self._incoming_multiplier(opponent, active)       # cuanto me pega el rival
            my_punish = self._best_move_multiplier(active, opponent)   # cuanto le pego yo
            if threat >= 2.0 and my_punish < 2.0:
                pivot = self._best_switch(battle, opponent)
                if (pivot is not None
                        and self._incoming_multiplier(opponent, pivot) <= 1.0
                        and pivot.current_hp_fraction > 0.4):
                    return self.create_order(pivot)

        # (3) Mejor movimiento por daño estimado
        if battle.available_moves:
            best = max(
                battle.available_moves,
                key=lambda m: self._estimate_damage(m, active, opponent) if opponent else m.base_power,
            )
            return self.create_order(best)

        return self.choose_random_move(battle)


async def main():
    # Etiqueta legible de la condicion (para los prints)
    cond = "CON ICRL" if USE_ICRL else "SIN ICRL"

    # Instanciación del bot LLM con la configuracion de la tanda
    llm_account = AccountConfiguration("BotPruebaTFM", None)
    llm_bot = SimpleLLMPlayer(
        provider=PROVIDER,
        model_name=MODEL,
        use_icrl=USE_ICRL,                       # <--- interruptor de la ablacion
        account_configuration=llm_account,
        battle_format="gen5randombattle"
    )

    # Instanciación del oponente heurístico
    heuristic_account = AccountConfiguration("OponentePruebaTFM", None)
    opponent_bot = HeuristicPlayer(
        account_configuration=heuristic_account,
        battle_format="gen5randombattle")

    n_matches = N_MATCHES

    print(f"\nIniciando ablacion ICRL | {MODEL} | {cond} | {n_matches} batallas (servidor local)...")
    start_time = time.time()

    # En el servidor local no hay filtro anti-spam: todas las batallas en una sola llamada.
    await llm_bot.battle_against(opponent_bot, n_battles=n_matches)

    end_time = time.time()

    # Cálculo de métricas
    wins = llm_bot.n_won_battles
    win_rate = (wins / n_matches) * 100

    total_turns = 0
    remaining_pokemon = 0

    # Preparar datos para el CSV detallado
    matches_data = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")  # identificador único para este lote
    used_model = llm_bot.model_name
    opponent_name = "HeuristicPlayer"
    total_tokens = 0
    total_decisions = 0
    total_switches = 0
    total_forced = 0
    total_fallbacks = 0

    # Se itera sobre todas las batallas guardadas en la memoria del bot
    for battle in llm_bot.battles.values():
        turns = battle.turn
        total_turns += battle.turn
        # Se cuentan cuántos Pokémon de nuestro equipo no están debilitados
        alives = sum(1 for mon in battle.team.values() if not mon.fainted)
        remaining_pokemon += alives
        win = 1 if battle.won else 0

        match_tokens = llm_bot.token_usage.get(battle.battle_tag, 0)
        total_tokens += match_tokens

        # Métricas de decisión: cambios voluntarios, relevos forzados y fallbacks (random)
        actions = llm_bot.action_log.get(battle.battle_tag, [])
        decisiones = len(actions)
        cambios = actions.count("switch")     # cambios VOLUNTARIOS
        forzados = actions.count("forced")    # relevos OBLIGADOS (tras KO o U-turn)
        fallbacks = actions.count("fallback")
        total_decisions += decisiones
        total_switches += cambios
        total_forced += forzados
        total_fallbacks += fallbacks

        # Diccionario con la fila de esta partida concreta
        matches_data.append({
            "timestamp": timestamp,
            "batch_id": batch_id,
            "modelo": used_model,
            "icrl": USE_ICRL,                 # <--- condicion de la ablacion (True/False)
            "oponente": opponent_name,
            "victoria": win,
            "turnos": turns,
            "pokemon_vivos": alives,
            "tokens_gastados": match_tokens,
            "decisiones": decisiones,
            "cambios": cambios,
            "forzados": forzados,
            "fallbacks": fallbacks
        })

    avg_turns = total_turns / n_matches if n_matches > 0 else 0
    avg_alives = remaining_pokemon / n_matches if n_matches > 0 else 0
    avg_tokens = total_tokens / n_matches if n_matches > 0 else 0

    # Guardar en archivo CSV (fichero NUEVO y dedicado a la ablacion, con columna 'icrl')
    csv_file = "resultados_ablacion_icrl.csv"
    file_exists = os.path.isfile(csv_file)

    # Abrimos en modo 'a' (append) para acumular las 4 tandas en el mismo fichero
    with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
        fields = ["timestamp", "batch_id", "modelo", "icrl", "oponente", "victoria", "turnos",
                  "pokemon_vivos", "tokens_gastados", "decisiones", "cambios", "forzados", "fallbacks"]
        writer = csv.DictWriter(f, fieldnames=fields)

        if not file_exists:
            writer.writeheader()  # si es la primera vez, escribe la cabecera

        writer.writerows(matches_data)

    ci_low, ci_high = wilson_ci(wins, n_matches)
    active_decisions = total_decisions - total_forced   # excluye relevos obligados
    switch_rate = (total_switches / active_decisions * 100) if active_decisions else 0
    fallback_rate = (total_fallbacks / total_decisions * 100) if total_decisions else 0

    print("\n" + "=" * 60)
    print("RESULTADOS DEL EXPERIMENTO (ABLACION ICRL)")
    print("=" * 60)
    print(f"Modelo: {used_model}   |   {cond}   vs   {opponent_name}")
    print(f"Total de partidas: {n_matches}")
    print(f"Victorias: {wins}  (WR: {round(win_rate, 1)}%  |  IC95% Wilson: "
          f"{round(ci_low*100, 1)}%-{round(ci_high*100, 1)}%)")
    print(f"Promedio de turnos por partida: {round(avg_turns, 1)}")
    print(f"Promedio de Pokémon vivos al final: {round(avg_alives, 1)} / 6.0")
    print(f"Promedio de tokens por partida: {round(avg_tokens)}")
    print(f"Tasa de CAMBIOS voluntarios: {round(switch_rate, 1)}%  "
          f"({total_switches}/{active_decisions} decisiones activas)")
    print(f"   Relevos forzados (excluidos del % anterior): {total_forced}")
    print(f"Tasa de FALLBACK aleatorio: {round(fallback_rate, 1)}%  "
          f"({total_fallbacks}/{total_decisions} decisiones)")
    print(f"Tiempo total de ejecución: {round(end_time - start_time, 2)} segundos")
    print(f"Datos guardados en: {csv_file}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())