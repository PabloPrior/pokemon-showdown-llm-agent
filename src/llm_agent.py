import os
import json
import asyncio
from dotenv import load_dotenv
from groq import AsyncGroq
from collections import Counter, deque
from poke_env.player import Player
from openai import AsyncOpenAI

# Carga de las variables de .env
load_dotenv()

class SimpleLLMPlayer(Player):
    def __init__(self, provider="groq", model_name="llama-3.3-70b-versatile", use_icrl=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.provider = provider
        self.model_name = model_name
        self.use_icrl = use_icrl            # interruptor del modulo de memoria estructurada (ICRL on/off)
        self.token_usage = {}
        self.action_log = {}   # battle_tag -> lista de "move"/"switch"/"fallback"
        self.icrl_mem = {}     # battle_tag -> deque con el feedback de los ultimos turnos
        self.icrl_prev = {}    # battle_tag -> snapshot del estado tras la ultima decision

        # Inicialización dinámica del cliente según el proveedor elegido
        if self.provider == "groq":
            self.client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
        elif self.provider == "openai":
            self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif self.provider == "openrouter":
            # OpenRouter es compatible con la API de OpenAI: mismo cliente, otra base_url
            self.client = AsyncOpenAI(
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
            )
        else:
            raise ValueError(f"Proveedor no soportado: {self.provider}")

    def _log_action(self, battle, kind):
        # Registra el tipo de decisión del turno: "move", "switch" o "fallback" (random).
        self.action_log.setdefault(battle.battle_tag, []).append(kind)

    # ---------- MODULO DE MEMORIA ESTRUCTURADA (ICRL) ----------
    _STATUS_ES = {
        "PSN": "envenenado", "TOX": "gravemente envenenado", "BRN": "quemado",
        "PAR": "paralizado", "SLP": "dormido", "FRZ": "congelado", "FNT": "debilitado",
    }

    def _status_name(self, pokemon):
        if pokemon is None or pokemon.status is None:
            return None
        return pokemon.status.name

    def _boost_note(self, pokemon, quien):
        if pokemon is None:
            return None
        notas = [f"{s} {'+' if v > 0 else ''}{v}"
                 for s, v in pokemon.boosts.items()
                 if s in ("atk", "spa", "spe", "def", "spd") and v != 0]
        if notas:
            return f"{quien} tiene modificadores de estadistica: {', '.join(notas)}."
        return None

    def _icrl_feedback(self, battle):
        tag = battle.battle_tag
        me = battle.active_pokemon
        opp = battle.opponent_active_pokemon
        prev = self.icrl_prev.get(tag)
        lineas = []

        if prev is not None and me is not None and opp is not None:
            opp_sp, my_sp = opp.species, me.species
            opp_hp = opp.current_hp_fraction if opp.current_hp_fraction is not None else 1.0
            my_hp = me.current_hp_fraction if me.current_hp_fraction is not None else 1.0

            if prev["my_move"] and prev["my_move_damaging"] and prev["opp_species"] == opp_sp:
                delta = prev["opp_hp"] - opp_hp
                if delta <= 0.005:
                    lineas.append(
                        f"Tu movimiento '{prev['my_move']}' NO causo dano a {opp_sp} "
                        f"(posible inmunidad por habilidad o tipo): no lo repitas."
                    )
                else:
                    lineas.append(f"Tu '{prev['my_move']}' causo ~{round(delta * 100)}% de dano a {opp_sp}.")

            if prev["opp_species"] != opp_sp:
                lineas.append(f"El rival cambio a {opp_sp}.")

            if prev["opp_species"] == opp_sp:
                opp_status = self._status_name(opp)
                if opp_status and opp_status != prev["opp_status"]:
                    lineas.append(f"El rival {opp_sp} quedo {self._STATUS_ES.get(opp_status, opp_status.lower())}.")

            if prev["my_species"] == my_sp:
                my_status = self._status_name(me)
                if my_status and my_status != prev["my_status"]:
                    lineas.append(f"Tu {my_sp} quedo {self._STATUS_ES.get(my_status, my_status.lower())}.")
                if my_status != "FNT" and prev["my_hp"] - my_hp >= 0.35:
                    lineas.append(f"Tu {my_sp} recibio un golpe fuerte (~{round((prev['my_hp'] - my_hp) * 100)}% de vida).")

        if lineas:
            self.icrl_mem.setdefault(tag, deque(maxlen=3)).append(" ".join(lineas))

        bloque = list(self.icrl_mem.get(tag, []))
        for nota in (self._boost_note(opp, "El rival"), self._boost_note(me, "Tu Pokemon")):
            if nota:
                bloque.append(nota)

        if not bloque:
            return ""
        cuerpo = "\n        ".join(f"- {l}" for l in bloque)
        return ("\n        --- MEMORIA DE TURNOS PREVIOS (resultado de tus ultimas acciones) ---\n        "
                + cuerpo + "\n        Aprovecha esta informacion para no repetir acciones inutiles.\n")

    def _icrl_snapshot(self, battle, decision):
        me = battle.active_pokemon
        opp = battle.opponent_active_pokemon
        if me is None or opp is None:
            return
        my_move, my_move_damaging = None, False
        if decision and decision.get("type") == "move":
            my_move = decision.get("name")
            mv = next((m for m in battle.available_moves if m.id == my_move), None)
            if mv is not None:
                my_move_damaging = (mv.category.name != "STATUS") and ((mv.base_power or 0) > 0)
        self.icrl_prev[battle.battle_tag] = {
            "my_move": my_move,
            "my_move_damaging": my_move_damaging,
            "my_species": me.species,
            "my_hp": me.current_hp_fraction if me.current_hp_fraction is not None else 1.0,
            "my_status": self._status_name(me),
            "opp_species": opp.species,
            "opp_hp": opp.current_hp_fraction if opp.current_hp_fraction is not None else 1.0,
            "opp_status": self._status_name(opp),
        }

    async def choose_move(self, battle):
        # PARSEO DEL ESTADO 
        active_pokemon = battle.active_pokemon
        opponent_pokemon = battle.opponent_active_pokemon
        
        # Extrae los movimientos y cambios disponibles
        available_moves = {m.id: m for m in battle.available_moves}
        available_switches = {s.species: s for s in battle.available_switches}
        
        # Si no se puede hacer nada, elegimos aleatorio
        if not available_moves and not available_switches:
            return self.choose_random_move(battle)

        
        # PROMPT ENGINEERING CON KAG AVANZADO (Pokédex y multiplicadores)
        
        # Función auxiliar para formatear los tipos del Pokémon
        def get_types(pokemon):
            return " / ".join([t.name for t in pokemon.types if t is not None])

        # Función auxiliar para calcular el daño de un movimiento contra el rival
        def get_move_details(move, opponent):
            try:
                # Si es mov. de estado, no calcula daño
                if move.category.name == "STATUS":
                    return f"  - {move.id}: Tipo={move.type.name}, Categoría=ESTADO, Poder Base=0 (Movimiento de apoyo/defensa)"
                
                multiplier = opponent.damage_multiplier(move)
                
                if multiplier >= 4:
                    effectiveness = f"Súper efectivo doble (Daño x{multiplier})"
                elif multiplier >= 2:
                    effectiveness = f"Súper efectivo (Daño x{multiplier})"
                elif multiplier == 1:
                    effectiveness = f"Neutro (Daño x{multiplier})"
                elif multiplier == 0.5:
                    effectiveness = f"Poco efectivo (Daño x{multiplier})"
                elif multiplier <= 0.25 and multiplier > 0:
                    effectiveness = f"Muy poco efectivo (Daño x{multiplier})"
                elif multiplier == 0:
                    effectiveness = "El rival es inmune (Daño x0)"
                else:
                    effectiveness = f"Desconocido (Daño x{multiplier})"
                    
                return f"  - {move.id}: Tipo={move.type.name}, Poder Base={move.base_power}, Efectividad={effectiveness}"            
            
            except Exception as e:
                print(f"Error calculando multiplicador para {move.id}: {e}")
                return f"  - {move.id}: Tipo={move.type.name if move.type else 'Desc'}, Poder Base={move.base_power}"
            
        # Extracción de la lista detallada de movimientos
        moves_info = "\n".join([get_move_details(m, opponent_pokemon) for m in battle.available_moves])

        force_switch_alert = ""
        if battle.force_switch:
            force_switch_alert = "\n¡ATENCIÓN! Has usado un movimiento como U-Turn o tu Pokémon ha sido debilitado. ESTÁS EN UN 'FORCE SWITCH'. Estás obligado a devolver un JSON con un 'switch'.\n"

        # Preparación del contexto táctico inyectando la Pokédex
        # Bloque de memoria estructurada (ICRL); cadena vacia si el modulo esta desactivado
        icrl_block = self._icrl_feedback(battle) if self.use_icrl else ""

        state_description = f"""
        CONTEXTO DEL TURNO: {battle.turn}
        
        --- TU POKÉMON ACTIVO ---
        Especie: {active_pokemon.species}
        Salud: {active_pokemon.current_hp}/{active_pokemon.max_hp}
        Tipos: {get_types(active_pokemon)}
        Estadísticas Base: Atq={active_pokemon.base_stats['atk']}, Def={active_pokemon.base_stats['def']}, Vel={active_pokemon.base_stats['spe']}
        
        --- POKÉMON RIVAL ---
        Especie: {opponent_pokemon.species}
        Salud: {opponent_pokemon.current_hp}/{opponent_pokemon.max_hp} (estimada)
        Tipos: {get_types(opponent_pokemon)}
        
        {icrl_block}
        --- ACCIONES DISPONIBLES ---
        MOVIMIENTOS:
        {moves_info}
        
        CAMBIOS DISPONIBLES: {list(available_switches.keys())}
        """

        # Instrucciones estrictas del sistema
        # system_prompt = """
        # Eres un jugador experto de batallas tácticas de Pokémon. 
        # Analiza el estado del turno y decide la mejor acción.
        # DEBES responder ÚNICAMENTE con un objeto JSON válido con este formato exacto:
        # {"type": "move", "name": "nombre_del_movimiento"} 
        # o 
        # {"type": "switch", "name": "nombre_del_pokemon"}
        # No añadas texto adicional, ni explicaciones, solo el JSON puro.
        # """

        # Instrucciones estrictas y tácticas del sistema (Chain-of-Thought)
        system_prompt = """
        Eres un jugador experto de batallas tácticas de Pokémon (formato randombattle).
        Tu objetivo es debilitar a todo el equipo rival manteniendo vivo el tuyo.

        REGLA DE DECISIÓN (síguela en orden):
        Por defecto, ATACA. Cambiar de Pokémon NO es gratis: pierdes el turno y el Pokémon
        que entra recibe el ataque del rival. Solo debes cambiar si se cumple una de estas
        condiciones concretas:
        (a) Tu Pokémon activo no tiene NINGÚN movimiento que haga daño relevante (todos son
            "Poco efectivo"/"Muy poco efectivo" o el rival es inmune) Y tienes en el banquillo
            un Pokémon que resista los ataques del rival.
        (b) Tu Pokémon va a ser debilitado con casi total seguridad este turno y no puede
            hacer nada útil (ni debilitar al rival, ni aplicar un estado decisivo).
        (c) Estás OBLIGADO a cambiar (force switch).
        Si no se cumple (a), (b) ni (c): elige un MOVIMIENTO. Ante la duda, ataca.

        CÓMO ELEGIR EL MOVIMIENTO:
        - Fíate del campo "Efectividad" que se te proporciona: ya tiene en cuenta los tipos.
          No deduzcas ventajas de tipo por tu cuenta ni las inventes.
        - Prioriza el mayor daño: primero "Súper efectivo", luego "Neutro" con mayor Poder Base.
        - Usa un movimiento de ESTADO solo si te da una ventaja clara y no corres riesgo de caer.

        RESTRICCIONES:
        - "name" debe ser EXACTAMENTE uno de los nombres de la lista de MOVIMIENTOS o de
          CAMBIOS DISPONIBLES. No inventes nombres.
        - Razona en 1-2 frases, directo y decidido. No te recrees en lo peligroso que es el rival.

        Responde SOLO con un JSON válido, sin texto adicional, con este formato exacto:
        {"razonamiento": "<1-2 frases>", "type": "move", "name": "<nombre exacto de la lista>"}
        o
        {"razonamiento": "<1-2 frases>", "type": "switch", "name": "<nombre exacto de la lista>"}
        """

        # print(state_description)  # descomenta solo para depurar; en tiradas N=50 inunda la consola

        # await asyncio.sleep(3.5)

        # try:
        #     # LLAMADA A LA API
        #     api_tasks = []
        #     for _ in range(3):
        #         api_tasks.append(
        #             self.client.chat.completions.create(
        #         messages=[
        #             {"role": "system", "content": system_prompt},
        #             {"role": "user", "content": state_description}
        #         ],
        #         #model="llama-3.3-70b-versatile", # USAR PARA PRUEBAS POTENTES
        #         model="llama-3.1-8b-instant", # modelo muy tonto
        #         response_format={"type": "json_object"}, # forzamos la salida JSON
        #         temperature=0.4 # temperatura para diversiddad táctica
        #     ))

        #     # chat_completion = await self.client.chat.completions.create(
        #     #     messages=[
        #     #         {"role": "system", "content": system_prompt},
        #     #         {"role": "user", "content": state_description}
        #     #     ],
        #     #     model="llama-3.3-70b-versatile", # USAR PARA PRUEBAS POTENTES
        #     #     #model="llama-3.1-8b-instant", # modelo muy tonto
        #     #     response_format={"type": "json_object"}, # forzamos la salida JSON
        #     #     temperature=0.2 # temperatura baja para que sea analítico, no creativo
        #     # )

        #     #Extracción de respuesta del modelo
        #     # response_text = chat_completion.choices[0].message.content
        #     # decision = json.loads(response_text) # texto a diccionario Python
        #     #print(state_description)
        #     #print(f"\n[TURNO {battle.turn}] Llama 3 decide: {decision}")

        #     responses = await asyncio.gather(*api_tasks)

        #     shares_votes = []
        #     for resp in responses:
        #         try:
        #             response_str = resp.choices[0].message.content
        #             data = json.loads(response_str)
        #             # Formato unificado para contar los votos: "tipo_nombre"
        #             vote = f"{data.get('type')}_{data.get('name')}"
        #             shares_votes.append(vote)
        #         except Exception as e:
        #             print(f"Error procesando un voto individual: {e}")

        #     # Si fallan los 3 votos, forzamos la excepción
        #     if not shares_votes:
        #         raise ValueError("Ninguno de los votos devolvió un JSON válido.")

        #     # Extracción de la acción ganadora por mayoría
        #     count = Counter(shares_votes)
        #     winning_action, n_votes = count.most_common(1)[0]

        #     print(f"\n[TURNO {battle.turn}] Comité SC | Votos: {shares_votes} -> Ganador: {winning_action} ({n_votes}/3)")

        #     # Reconstrucción del diccionario 'decision' para el resto de tu código
        #     choose_type, choose_name = winning_action.split("_", 1)
        #     decision = {"type": choose_type, "name": choose_name}

            
        #     # EJECUCIÓN DE LA ACCIÓN 
        #     if decision["type"] == "move" and decision["name"] in available_moves:
        #         return self.create_order(available_moves[decision["name"]])
            
        #     elif decision["type"] == "switch" and decision["name"] in available_switches:
        #         return self.create_order(available_switches[decision["name"]])
            
        #     else:
        #         print("Llama 3 alucinó un movimiento inválido. Usando fallback.")
        #         return self.choose_random_move(battle)

        # except Exception as e:
        #     print(f"Error en la API o parseando JSON: {e}. Usando fallback aleatorio.")
        #     return self.choose_random_move(battle)
        # Variables de control para los reintentos
        max_attempts = 3
        current_attempt = 0
        successful_decision = False

        while current_attempt < max_attempts and not successful_decision:
            try:
                # LLAMADA A LA API
                # Parámetros base de la llamada (comunes a todos los proveedores)
                create_kwargs = {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": state_description},
                    ],
                    "model": self.model_name,
                    "response_format": {"type": "json_object"},
                    "temperature": 0.4,
                }
                # Solo en OpenRouter: fijamos el backend y exigimos soporte del JSON forzado
                if self.provider == "openrouter":
                    create_kwargs["extra_body"] = {
                        "provider": {
                            "require_parameters": True,   # solo proveedores que respetan response_format
                            "order": ["AkashML"],        # backend fijo para reproducibilidad
                            "allow_fallbacks": False
                        }
                    }

                api_tasks = []
                for _ in range(3):
                    api_tasks.append(
                        self.client.chat.completions.create(**create_kwargs)
                    )

                responses = await asyncio.gather(*api_tasks)

                shares_votes = []
                for resp in responses:
                    try:
                        if battle.battle_tag not in self.token_usage:
                            self.token_usage[battle.battle_tag] = 0

                        if hasattr(resp, 'usage') and resp.usage:
                            tokens_request = getattr(resp.usage, 'total_tokens', 0)
                            self.token_usage[battle.battle_tag] += tokens_request

                        response_str = resp.choices[0].message.content
                        data = json.loads(response_str)
                        vote = f"{data.get('type')}_{data.get('name')}"
                        shares_votes.append(vote)
                    except Exception as e:
                        print(f"Error procesando un voto individual: {e}")

                if not shares_votes:
                    raise ValueError("Ninguno de los votos devolvió un JSON válido.")

                # Extraer ganador
                count = Counter(shares_votes)
                winning_action, num_votes = count.most_common(1)[0]
                
                print(f"\n[TURNO {battle.turn}] Comité SC | Votos: {shares_votes} -> Ganador: {winning_action} ({num_votes}/3)")

                chosen_type, chosen_name = winning_action.split("_", 1)
                decision = {"type": chosen_type, "name": chosen_name}
                successful_decision = True 

                # Actualiza la memoria ICRL con el estado y la accion elegida (para el turno siguiente)
                if self.use_icrl:
                    self._icrl_snapshot(battle, decision)

                # EJECUCIÓN DE LA ACCIÓN
                if decision["type"] == "move" and decision["name"] in available_moves:
                    self._log_action(battle, "move")
                    return self.create_order(available_moves[decision["name"]])
                elif decision["type"] == "switch" and decision["name"] in available_switches:
                    # Distinguimos cambio voluntario de relevo obligado (tras KO o U-turn)
                    self._log_action(battle, "forced" if battle.force_switch else "switch")
                    return self.create_order(available_switches[decision["name"]])
                else:
                    print("El modelo alucinó una acción inválida. Usando fallback.")
                    self._log_action(battle, "fallback")
                    return self.choose_random_move(battle)

            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg:
                    current_attempt += 1
                    # Si es error 429, esperamos 6 segundos y volvemos a intentarlo (hasta 3 veces)
                    print(f"\n[!] Rate Limit alcanzado. Esperando 6 segundos (Intento {current_attempt}/{max_attempts})...")
                    await asyncio.sleep(6)
                else:
                    # Si es otro error usamos random
                    print(f"Error fatal en la API: {e}. Usando fallback aleatorio.")
                    self._log_action(battle, "fallback")
                    return self.choose_random_move(battle)
        
        # Si hemos superado los 3 reintentos y sigue fallando por 429, usamos random
        if not successful_decision:
            print("Máximo de reintentos por Rate Limit superado. Usando fallback aleatorio.")
            self._log_action(battle, "fallback")
            return self.choose_random_move(battle)