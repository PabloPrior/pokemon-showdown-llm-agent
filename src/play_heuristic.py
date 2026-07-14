import asyncio
from poke_env import AccountConfiguration, LocalhostServerConfiguration
from main import HeuristicPlayer  # Importamos tu bot matemático desde tu script principal

async def main():
    # 1. Configuramos la cuenta del bot heurístico
    bot_account = AccountConfiguration("BotMatematicoTFM", "contraseña_secreta02@")
    
    # 2. Instanciamos a tu bot matemático
    heuristic_bot = HeuristicPlayer(
        account_configuration=bot_account,
        server_configuration=LocalhostServerConfiguration,
        battle_format="gen5randombattle"
    )

    print("\n" + "="*60)
    print("MODO DESAFÍO: HUMANO VS BOT HEURÍSTICO (MATEMÁTICAS)")
    print("="*60)
    print(f"[*] El bot '{heuristic_bot.username}' está en línea y calculando.")
    print("\nINSTRUCCIONES PARA EL RETADOR:")
    print("  1. Abre tu navegador web y entra en: http://localhost:8000")
    print("  2. Arriba a la derecha, dale a 'Choose name' y ponte un apodo.")
    print(f"  3. Haz clic en el botón 'Find a user' y busca a: {heuristic_bot.username}")
    print("  4. Rétale a una batalla en el formato: [Gen 5] Random Battle")
    print("\n[!] Esperando el desafío en el servidor local...")

    # El bot se queda "escuchando" hasta que reciba exactamente 1 desafío
    await heuristic_bot.accept_challenges(opponent=None, n_challenges=1)

    print("\n" + "="*60)
    print("¡Batalla terminada!")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())