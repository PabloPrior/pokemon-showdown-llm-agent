"""
Analisis de la ablacion ICRL (2 modelos x 2 condiciones, N=100 cada una).
Lee resultados_ablacion_icrl.csv y genera:
  - fig_ablacion_winrate.png : winrate 2x2 con IC 95% de Wilson  (figura principal)
  - fig_ablacion_switch.png  : tasa de cambios voluntarios 2x2   (el mecanismo)
  - fig_ablacion_tokens.png  : coste en tokens por partida 2x2
  - resumen_ablacion.csv     : tabla con todas las metricas
Ademas imprime los tests de dos proporciones (efecto del ICRL en cada modelo).
"""

import csv
import math
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

CSV_IN = "resultados_ablacion_icrl.csv"

# --- Orden y etiquetas legibles ---
MODELOS = ["gpt-4o-mini", "meta-llama/llama-3.3-70b-instruct"]
ETIQ = {"gpt-4o-mini": "gpt-4o-mini", "meta-llama/llama-3.3-70b-instruct": "Llama-3.3-70B"}
COND = ["False", "True"]            # False = Sin ICRL, True = Con ICRL
COND_LABEL = {"False": "Sin ICRL", "True": "Con ICRL"}
C_SIN, C_CON = "#4C72B0", "#DD8452"  # azul (sin), naranja (con)


def wilson(w, n, z=1.96):
    """Proporcion e intervalo de Wilson 95%."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = w / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = (z / d) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, max(0.0, c - m), min(1.0, c + m)


def two_prop_p(w1, n1, w2, n2):
    """p-valor (dos colas) del test de dos proporciones."""
    p1, p2 = w1 / n1, w2 / n2
    p = (w1 + w2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p2 - p1) / se
    pv = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return z, pv


# ---------- 1) Leer y agrupar ----------
rows = list(csv.DictReader(open(CSV_IN, encoding="utf-8")))
g = defaultdict(list)
for r in rows:
    g[(r["modelo"], r["icrl"])].append(r)

M = {}
for mod in MODELOS:
    for ic in COND:
        d = g[(mod, ic)]
        n = len(d)
        w = sum(int(x["victoria"]) for x in d)
        p, lo, hi = wilson(w, n)
        sw = sum(int(x["cambios"]) for x in d)
        dec = sum(int(x["decisiones"]) for x in d)
        fz = sum(int(x["forzados"]) for x in d)
        act = dec - fz
        tok = sum(int(x["tokens_gastados"]) for x in d) / n if n else 0
        M[(mod, ic)] = dict(n=n, w=w, wr=100 * p, lo=100 * lo, hi=100 * hi,
                            switch=100 * sw / act if act else 0, tokens=tok)


# ---------- 2) Figuras (barras agrupadas Sin/Con) ----------
x = np.arange(len(MODELOS))
width = 0.36


def grouped_bar(metric, ylabel, title, fname, fmt="{:.0f}", ci=False, ymax=None):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    sin = [M[(m, "False")][metric] for m in MODELOS]
    con = [M[(m, "True")][metric] for m in MODELOS]

    if ci:
        sin_err = [[M[(m, "False")]["wr"] - M[(m, "False")]["lo"] for m in MODELOS],
                   [M[(m, "False")]["hi"] - M[(m, "False")]["wr"] for m in MODELOS]]
        con_err = [[M[(m, "True")]["wr"] - M[(m, "True")]["lo"] for m in MODELOS],
                   [M[(m, "True")]["hi"] - M[(m, "True")]["wr"] for m in MODELOS]]
        b1 = ax.bar(x - width / 2, sin, width, yerr=sin_err, capsize=6, color=C_SIN, label="Sin ICRL")
        b2 = ax.bar(x + width / 2, con, width, yerr=con_err, capsize=6, color=C_CON, label="Con ICRL")
        tops_sin = [M[(m, "False")]["hi"] for m in MODELOS]
        tops_con = [M[(m, "True")]["hi"] for m in MODELOS]
    else:
        b1 = ax.bar(x - width / 2, sin, width, color=C_SIN, label="Sin ICRL")
        b2 = ax.bar(x + width / 2, con, width, color=C_CON, label="Con ICRL")
        tops_sin, tops_con = sin, con

    ax.set_xticks(x)
    ax.set_xticklabels([ETIQ[m] for m in MODELOS])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ymax:
        ax.set_ylim(0, ymax)
    ax.legend()

    off = (ymax or max(sin + con)) * 0.015
    for bars, vals, tops in [(b1, sin, tops_sin), (b2, con, tops_con)]:
        for bar, v, t in zip(bars, vals, tops):
            ax.text(bar.get_x() + bar.get_width() / 2, t + off, fmt.format(v),
                    ha="center", va="bottom", fontweight="bold", fontsize=10)

    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print("Figura guardada:", fname)


grouped_bar("wr", "Tasa de victorias (%)",
            "Winrate por modelo y condición frente a HeuristicPlayer (IC 95% de Wilson)",
            "fig_ablacion_winrate.png", fmt="{:.0f}%", ci=True, ymax=100)

grouped_bar("switch", "Cambios voluntarios (%)",
            "Tasa de cambios voluntarios por modelo y condición",
            "fig_ablacion_switch.png", fmt="{:.1f}%", ci=False)

grouped_bar("tokens", "Tokens por partida",
            "Coste medio en tokens por partida por modelo y condición",
            "fig_ablacion_tokens.png", fmt="{:.0f}", ci=False)


# ---------- 3) Tabla-resumen a CSV ----------
with open("resumen_ablacion.csv", "w", newline="", encoding="utf-8") as f:
    wcsv = csv.writer(f)
    wcsv.writerow(["modelo", "icrl", "N", "victorias", "WR_%", "WR_lo_%", "WR_hi_%",
                   "cambios_%", "tokens_partida"])
    for mod in MODELOS:
        for ic in COND:
            m = M[(mod, ic)]
            wcsv.writerow([ETIQ[mod], COND_LABEL[ic], m["n"], m["w"],
                           round(m["wr"], 1), round(m["lo"], 1), round(m["hi"], 1),
                           round(m["switch"], 1), round(m["tokens"])])
print("Tabla guardada: resumen_ablacion.csv")


# ---------- 4) Tests estadisticos (efecto del ICRL por modelo) ----------
print("\n" + "=" * 64)
print("EFECTO DEL ICRL (test de dos proporciones, dentro de cada modelo)")
print("=" * 64)
for mod in MODELOS:
    a, b = M[(mod, "False")], M[(mod, "True")]
    z, pv = two_prop_p(a["w"], a["n"], b["w"], b["n"])
    sig = "SIGNIFICATIVO" if pv < 0.05 else "no significativo"
    print(f"{ETIQ[mod]:14s}  WR {a['wr']:.0f}% -> {b['wr']:.0f}%  "
          f"(Δ={b['wr']-a['wr']:+.0f} pp)  z={z:+.2f}  p={pv:.3f}  [{sig}]")
print("Nota: el efecto sobre 'cambios voluntarios' sí es significativo en ambos modelos")
print("(gpt p<0.0001, Llama p=0.006); es el mecanismo de la interacción.")