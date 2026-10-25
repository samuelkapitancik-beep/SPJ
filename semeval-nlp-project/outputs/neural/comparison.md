# Namerané výsledky

| Model | Split | Seed / vstup | ACC | TCE | CS |
|---|---|---|---:|---:|---:|
| B0a majority | dev | — | 50.00 | 50.00 | 10.14 |
| B0b plausibility==validity (diagnostic) | dev | — | 50.52 | 100.00 | 9.00 |
| B1 TF-IDF+LR (train only) | dev | — | 67.71 | 9.09 | 20.44 |
| B0a_majority | test_st1 | — | 50.26 | 50.00 | 10.19 |
| B0b_plaus_heuristic | test_st1 | — | 50.26 | 100.00 | 8.95 |
| B1_tfidf_lr | test_st1 | — | 65.45 | 10.90 | 18.82 |
| S0 | dev | — | 90.10 | 9.40 | 26.97 |
| S0 | test_st1 | — | 90.58 | 10.42 | 26.37 |
| microsoft/deberta-v3-base | dev | 0 | 82.29 | 5.99 | 27.94 |
| microsoft/deberta-v3-base | test_st1 | 0 | 79.58 | 7.58 | 25.27 |
| microsoft/deberta-v3-base | dev | 1 | 78.65 | 5.05 | 28.08 |
| microsoft/deberta-v3-base | test_st1 | 1 | 72.77 | 3.12 | 30.11 |
| microsoft/deberta-v3-base | dev | 2 | 77.08 | 4.91 | 27.76 |
| microsoft/deberta-v3-base | test_st1 | 2 | 74.35 | 11.46 | 21.11 |
| microsoft/deberta-v3-base | dev | 3 | 77.60 | 6.97 | 25.23 |
| microsoft/deberta-v3-base | test_st1 | 3 | 76.44 | 8.67 | 23.39 |
| microsoft/deberta-v3-base | dev | 4 | 79.17 | 9.06 | 23.93 |
| microsoft/deberta-v3-base | test_st1 | 4 | 79.58 | 16.00 | 20.76 |
| meta-llama/Meta-Llama-3-8B-Instruct | dev | abstract | 53.12 | 47.87 | 10.87 |
| meta-llama/Meta-Llama-3-8B-Instruct | dev | canonical | 66.67 | 38.32 | 14.27 |
| meta-llama/Meta-Llama-3-8B-Instruct | dev | original | 64.06 | 44.68 | 13.29 |
| meta-llama/Meta-Llama-3-8B-Instruct | test_st1 | abstract | 52.36 | 48.96 | 10.66 |
| meta-llama/Meta-Llama-3-8B-Instruct | test_st1 | canonical | 79.58 | 27.66 | 18.27 |
| meta-llama/Meta-Llama-3-8B-Instruct | test_st1 | original | 67.02 | 43.62 | 13.97 |

Chýbajúce metriky znamenajú nejednoznačné odpovede LLM; v ACC sú počítané ako chyby.
Priemery a smerodajné odchýlky medzi seedmi sú v comparison.json.
