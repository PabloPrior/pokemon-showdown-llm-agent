[README.md](https://github.com/user-attachments/files/30019426/README.md)
# LLM-Powered Agent for Tactical Pokémon Battles

**Master's Thesis** — MSc in Artificial Intelligence  
**Author:** Pablo Prior Molina  
**Year:** 2026

## Overview

This project implements an autonomous agent powered by Large Language Models (LLMs) that plays turn-based tactical battles in [Pokémon Showdown](https://pokemonshowdown.com/) **without any model fine-tuning**. The system relies on a reasoning middleware built around three complementary mechanisms:

- **KAG (Knowledge-Augmented Generation):** Injects deterministic type matchup and damage data into the model's prompt, eliminating hallucinated type calculations.
- **Self-Consistency:** Stabilises decisions via a committee of k=3 independent inferences with majority voting.
- **Structured Memory (ICRL):** Feeds the agent back the actual outcome of its past actions (observed damage, immunities, status effects) to prevent repeating ineffective moves.

Two models are compared — **gpt-4o-mini** (OpenAI) and **Llama-3.3-70B** (Meta, via OpenRouter) — against a ladder of increasingly difficult opponents, with ablation experiments that isolate the contribution of each module.

## Key Results

| Model | Without ICRL | With ICRL | Δ |
|---|---|---|---|
| gpt-4o-mini | 27% | 22% | −5 pp |
| Llama-3.3-70B | 30% | **39%** | +9 pp |

*Win rate vs HeuristicPlayer (N=100 per condition, Wilson 95% CI).*

**Key finding:** the usefulness of structured memory depends on the model's baseline playing policy — it amplifies *panic switching* in gpt-4o-mini but adds useful caution to the more disciplined Llama-3.3-70B.

## Repository Structure

```
├── src/                    # Core source code
│   ├── llm_agent.py        # LLM agent with toggleable ICRL module
│   ├── main.py             # Experiment runner and ablation script
│   ├── play_human.py       # Human vs AI evaluation mode
│   ├── play_heuristic.py   # Human vs heuristic bot mode
│   ├── smoke_test.py       # Bot vs bot baseline test
│   └── smoke_test_icrl.py  # ICRL module smoke test
├── analysis/               # Result analysis scripts
│   ├── results_analysis.py
│   └── ablation_analysis.py
├── results/                # Experimental data (CSV)
│   ├── resultados_ablacion_icrl.csv   # 400 matches (2×2 ablation)
│   ├── resultados_experimento.csv     # Model comparison (N=100)
│   ├── resumen_ablacion.csv
│   └── resumen_metricas.csv
├── figures/                # Generated plots
├── .env.example            # API key template (fill in your own)
├── requirements.txt        # Python dependencies
└── README.md
```

## Prerequisites

- **Python 3.10+** with dependencies from `requirements.txt`
- **Node.js 18+** to run the local Pokémon Showdown server
- API keys: **OpenAI** (for gpt-4o-mini) and/or **OpenRouter** (for Llama-3.3-70B)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/pokemon-battle-llm-agent.git
cd pokemon-battle-llm-agent
```

### 2. Set up the virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

### 3. Configure API keys

```bash
cp .env.example .env
# Edit .env with your actual keys
```

### 4. Install and start Pokémon Showdown locally

```bash
git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown
npm install
node pokemon-showdown start --no-security
```

The server will be available at `http://localhost:8000`.

## Usage

### Run an experiment (ICRL ablation)

Edit the configuration block at the top of `src/main.py`:

```python
PROVIDER  = "openai"          # "openai" | "openrouter"
MODEL     = "gpt-4o-mini"     # "gpt-4o-mini" | "meta-llama/llama-3.3-70b-instruct"
USE_ICRL  = False             # False = no memory | True = structured memory
N_MATCHES = 100
```

```bash
cd src
python main.py
```

### Play against the agent (human evaluation)

```bash
cd src
python play_human.py
```

Open `http://localhost:8000` in your browser, choose a username, and challenge the bot.

### Generate figures and analysis

```bash
cd analysis
python ablation_analysis.py
```

## Reference

This work builds upon:

> Hu, S., Huang, T., & Liu, L. (2024). *PokéLLMon: A Human-Parity Agent for Pokémon Battles with Large Language Models.* arXiv:2402.01118.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
