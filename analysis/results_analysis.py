"""
Análisis comparativo del experimento (PokeLLMon-TFM).

Lee 'resultados_experimento.csv' (una fila por batalla) y, agregando por
(modelo, oponente), calcula:
  - Winrate con intervalo de confianza de Wilson al 95%.
  - Turnos y Pokémon vivos al final (media).
  - Tokens por partida y por decisión (eje de eficiencia, intrínseco al modelo).
  - Tasa de cambios voluntarios (excluye relevos forzados) -> métrica de "pánico".
  - Tasa de fallback aleatorio (control del confound de rate-limit / JSON).

Genera un resumen por consola, un 'resumen_metricas.csv' y tres figuras PNG
listas para la memoria.

Uso:  py analisis_resultados.py
Requiere:  pip install pandas matplotlib
"""

import math
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # guarda PNGs sin necesitar entorno gráfico
import matplotlib.pyplot as plt

CSV_FILE = "resultados_experimento.csv"


def wilson_ci(wins, n, z=1.96):
    """Intervalo de confianza de Wilson (95%) para una proporción."""
    if n == 0:
        return (0.0, 0.0)
    p = wins / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (max(0.0, center - margin), min(1.0, center + margin))


def short_name(modelo):
    """Etiqueta corta para los ejes (quita el prefijo del proveedor)."""
    return modelo.split("/")[-1]


def main():
    df = pd.read_csv(CSV_FILE)

    # Agregamos por (modelo, oponente): robusto si más adelante añades oponentes
    rows = []
    for (modelo, oponente), g in df.groupby(["modelo", "oponente"]):
        n = len(g)
        wins = int(g["victoria"].sum())
        lo, hi = wilson_ci(wins, n)

        decisiones = int(g["decisiones"].sum())
        cambios = int(g["cambios"].sum())
        forzados = int(g["forzados"].sum())
        fallbacks = int(g["fallbacks"].sum())
        activas = decisiones - forzados  # decisiones "libres" (sin relevos obligados)

        rows.append({
            "modelo": modelo,
            "oponente": oponente,
            "N": n,
            "victorias": wins,
            "WR": wins / n if n else 0,
            "WR_lo": lo,
            "WR_hi": hi,
            "turnos_media": g["turnos"].mean(),
            "vivos_media": g["pokemon_vivos"].mean(),
            "tokens_partida": g["tokens_gastados"].mean(),
            "tokens_decision": (g["tokens_gastados"].sum() / decisiones) if decisiones else 0,
            "cambios_pct": (cambios / activas) if activas else 0,
            "forzados_total": forzados,
            "fallback_pct": (fallbacks / decisiones) if decisiones else 0,
        })

    res = pd.DataFrame(rows).sort_values("WR", ascending=False).reset_index(drop=True)

    # ---------------- Resumen por consola ----------------
    print("\n" + "=" * 64)
    print("RESUMEN COMPARATIVO POR MODELO")
    print("=" * 64)
    for _, r in res.iterrows():
        print(f"\n{r['modelo']}  vs  {r['oponente']}   (N={r['N']})")
        print(f"  WR: {r['WR']*100:5.1f}%   IC95% Wilson: "
              f"{r['WR_lo']*100:.1f}% - {r['WR_hi']*100:.1f}%   ({r['victorias']}/{r['N']})")
        print(f"  Turnos/partida: {r['turnos_media']:.1f}    "
              f"Pokémon vivos al final: {r['vivos_media']:.2f}/6")
        print(f"  Tokens/partida: {r['tokens_partida']:.0f}    "
              f"Tokens/decisión: {r['tokens_decision']:.0f}")
        print(f"  Cambios voluntarios: {r['cambios_pct']*100:.1f}%    "
              f"Relevos forzados: {r['forzados_total']}")
        print(f"  Fallback aleatorio: {r['fallback_pct']*100:.2f}%")
    print("\n" + "=" * 64)

    # Guardar tabla
    res.to_csv("resumen_metricas.csv", index=False)
    print("Resumen guardado en: resumen_metricas.csv")

    # ---------------- Figuras ----------------
    etiquetas = [short_name(m) for m in res["modelo"]]
    paleta = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
    colores = paleta[:len(etiquetas)]

    # Fig 1: Winrate con IC de Wilson
    fig, ax = plt.subplots(figsize=(7, 5))
    wr = res["WR"].values * 100
    err_lo = (res["WR"] - res["WR_lo"]).values * 100
    err_hi = (res["WR_hi"] - res["WR"]).values * 100
    ax.bar(etiquetas, wr, color=colores, yerr=[err_lo, err_hi], capsize=8)
    ax.set_ylabel("Winrate (%)")
    ax.set_title("Winrate por modelo vs HeuristicPlayer (IC95% de Wilson)")
    ax.set_ylim(0, 100)
    for i, v in enumerate(wr):
        ax.text(i, v + err_hi[i] + 2, f"{v:.0f}%", ha="center", fontweight="bold")
    fig.tight_layout()
    fig.savefig("fig_winrate.png", dpi=150)
    plt.close(fig)

    # Fig 2: Tasa de cambios voluntarios
    fig, ax = plt.subplots(figsize=(7, 5))
    sr = res["cambios_pct"].values * 100
    ax.bar(etiquetas, sr, color=colores)
    ax.set_ylabel("Cambios voluntarios (%)")
    ax.set_title("Tasa de cambios voluntarios (métrica de pánico)")
    for i, v in enumerate(sr):
        ax.text(i, v + max(sr) * 0.02 + 0.1, f"{v:.1f}%", ha="center", fontweight="bold")
    fig.tight_layout()
    fig.savefig("fig_cambios.png", dpi=150)
    plt.close(fig)

    # Fig 3: Tokens por decisión (eficiencia)
    fig, ax = plt.subplots(figsize=(7, 5))
    td = res["tokens_decision"].values
    ax.bar(etiquetas, td, color=colores)
    ax.set_ylabel("Tokens por decisión")
    ax.set_title("Coste en tokens por decisión")
    for i, v in enumerate(td):
        ax.text(i, v + max(td) * 0.01, f"{v:.0f}", ha="center", fontweight="bold")
    fig.tight_layout()
    fig.savefig("fig_tokens.png", dpi=150)
    plt.close(fig)

    print("Figuras guardadas: fig_winrate.png, fig_cambios.png, fig_tokens.png")


if __name__ == "__main__":
    main()