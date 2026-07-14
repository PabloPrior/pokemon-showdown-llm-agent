import asyncio
from poke_env import AccountConfiguration, LocalhostServerConfiguration
from llm_agent import SimpleLLMPlayer

async def main():
    # 1. Configuramos la cuenta del bot (debe coincidir con la de tus pruebas)
    bot_account = AccountConfiguration("BotPruebaTFM", "contraseña_secreta01@")
    PROVIDER  = "openrouter"          
    MODEL     = "meta-llama/llama-3.3-70b-instruct"   
    USE_ICRL  = True             
    N_MATCHES = 100

    llm_account = AccountConfiguration("BotPruebaTFM", None)

    # 2. Instanciamos a tu "campeón" (He puesto al Llama 70B, que sacó el 39% WR)
    llm_bot = SimpleLLMPlayer(
        provider=PROVIDER,
        model_name=MODEL,
        use_icrl=USE_ICRL,                       # <--- interruptor de la ablacion
        account_configuration=llm_account,
        battle_format="gen5randombattle"
    )

    print("\n" + "="*60)
    print("MODO DESAFÍO: HUMANO VS INTELIGENCIA ARTIFICIAL")
    print("="*60)
    print(f"[*] El bot '{llm_bot.username}' ({llm_bot.model_name}) está en línea.")
    print("\nINSTRUCCIONES PARA EL RETADOR:")
    print("  1. Abre tu navegador web y entra en: http://localhost:8000")
    print("  2. Arriba a la derecha, dale a 'Choose name' y ponte un apodo.")
    print(f"  3. Haz clic en el botón 'Find a user' y busca a: {llm_bot.username}")
    print("  4. Rétale a una batalla asegurando que el formato es: [Gen 5] Random Battle")
    print("\n[!] Esperando el desafío en el servidor local...")

    # El bot se queda "escuchando" hasta que reciba exactamente 1 desafío
    await llm_bot.accept_challenges(opponent=None, n_challenges=1)

    print("\n" + "="*60)
    print("¡Batalla terminada! Revisa la terminal para ver cómo razonó la IA.")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())