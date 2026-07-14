import asyncio
from poke_env import AccountConfiguration
from main import MaxDamagePlayer, HeuristicPlayer

async def smoke():
    maxd = MaxDamagePlayer(
        account_configuration=AccountConfiguration("MaxDmgTest", None),
        battle_format="gen5randombattle", max_concurrent_battles=10)
    heur = HeuristicPlayer(
        account_configuration=AccountConfiguration("HeurTest", None),
        battle_format="gen5randombattle", max_concurrent_battles=10)
    await maxd.battle_against(heur, n_battles=50)
    print(f"Heuristic gana {heur.n_won_battles}/50 a MaxDamage")

asyncio.run(smoke())