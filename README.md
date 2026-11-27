# dlnn-xai-project

Projeto da disciplina de Redes Neurais e Aprendizado Profundo comparando a fidelidade de métodos de explicabilidade post-hoc (Grad-CAM, Attention Rollout, LIME, Partition SHAP e Integrated Gradients) em uma Vision GNN isotrópica (ViG-Ti) e em um Vision Transformer (DeiT-Tiny) ajustados no Imagenette2-160.

## Estrutura

```text
configs/            Hiperparâmetros e caminhos dos experimentos (default.yaml)
data/               Dataset local (ignorado no git) e splits em data/splits/
checkpoints/        Pesos pré-treinados e de fine-tuning (ignorados no git)
docs/               Proposta (dlnn-proposta/), logs (logs/) e relatório final (dlnn-final/)
src/                Módulos de dados, modelos (ViG-Ti, DeiT-Tiny), XAI e métricas
scripts/            Pontos de entrada do pipeline (E1 a E6, ablações e análise)
results/            Mapas .npz (ignorados), CSVs/JSONs de métricas e figuras
```

## Ambiente

Requer Python 3.11 a 3.13 (recomendado CUDA GPU). Com `uv`:

```bash
uv sync
```

Com `venv` e `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Toda inferência e geração de explicações roda em FP32 estrito (`cudnn.allow_tf32 = False` em [`src/config.py`](src/config.py)) para evitar variação discreta nos vizinhos `k`-NN da `ViG-Ti` entre tamanhos de lote distintos.

## Pipeline Experimental

Todos os scripts leem [`configs/default.yaml`](configs/default.yaml) por padrão:

```bash
python scripts/prepare_data.py
python scripts/train.py --model all
python scripts/run_xai.py --model all --method all
python scripts/evaluate.py --experiment all   # e3, e4, e5, e6 ou all
python scripts/analyze.py --mode all          # ablations, summary, figures ou all
```

Resumo dos experimentos implementados:
- `E1` ([`train.py`](scripts/train.py)): fine-tuning controlado no Imagenette (`train`: 9.469, `val`: 3.140, `test`: 785, sem vazamento).
- `E2` ([`run_xai.py`](scripts/run_xai.py)): geração e persistência dos mapas na amostra fixa estratificada (`xai_subset.csv`, `N = 100`, 10 por classe).
- `E3` ([`evaluate.py`](scripts/evaluate.py)): Deletion e Insertion AUC (`black` e `blur`, 50 passos) contra baseline aleatório + testes pareados de Wilcoxon.
- `E4` ([`evaluate.py`](scripts/evaluate.py)): sanity checks de aleatorização em cascata (Adebayo et al., `Spearman` e `SSIM`).
- `E5` ([`evaluate.py`](scripts/evaluate.py)): análise estrutural de *graph rewiring* `k`-NN na `ViG-Ti` sob Deletion (20 passos, Bloco 11 e perfil por camada).
- `E6` ([`evaluate.py`](scripts/evaluate.py)): teste de robustez a ruído gaussiano aditivo (`sigma in {0.02, 0.05, 0.10}`).
- `Ablações e análise` ([`analyze.py`](scripts/analyze.py)): sensibilidade de LIME (`n_segments`) e Integrated Gradients (`n_steps`), tabela mestra e figuras qualitativas.

## Principais Resultados

- **E1 (Acurácia no teste, `N = 785` / `xai_subset`, `N = 100`):** `ViG-Ti` `97,20%` (`97/100`), `DeiT-Tiny` `96,94%` (`97/100`).

| Modelo | Método | Del AUC (`blur`) | Ins AUC (`blur`) | E4 `head` Spearman | E4 `all` Spearman | E5 Rew. AUC (`blur`) | E6 `sigma=0.05` Spearman |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `ViG-Ti` | `gradcam` | 0,461 | 0,914 | -0,015 | 0,027 | 0,812 | 0,697 |
| `ViG-Ti` | `lime` | 0,577 | 0,914 | -0,068 | -0,053 | 0,811 | 0,224 |
| `ViG-Ti` | `partition_shap` | 0,481 | 0,947 | -0,249 | -0,140 | 0,813 | 0,601 |
| `ViG-Ti` | `integrated_gradients` | 0,662 | 0,684 | 0,583 | 0,430 | 0,806 | 0,535 |
| `ViG-Ti` | `random` | 0,542 | 0,547 | - | - | 0,810 | - |
| `DeiT-Tiny` | `gradcam` | 0,725 | 0,782 | -0,067 | -0,019 | - | 0,805 |
| `DeiT-Tiny` | `rollout` | 0,672 | 0,851 | 1,000 | -0,024 | - | 0,959 |
| `DeiT-Tiny` | `lime` | 0,644 | 0,905 | -0,023 | -0,136 | - | 0,223 |
| `DeiT-Tiny` | `partition_shap` | 0,445 | 0,956 | -0,011 | 0,059 | - | 0,822 |
| `DeiT-Tiny` | `integrated_gradients` | 0,649 | 0,879 | 0,698 | 0,419 | - | 0,838 |
| `DeiT-Tiny` | `random` | 0,747 | 0,746 | - | - | - | - |

Tabela completa em [`results/metrics/summary_master_table.csv`](results/metrics/summary_master_table.csv).

## Documentação

- Proposta: [`docs/dlnn-proposta.pdf`](docs/dlnn-proposta.pdf) ([`main.tex`](docs/dlnn-proposta/main.tex))
- Registros experimentais: [`e1_log.md`](docs/logs/e1_log.md), [`e2_log.md`](docs/logs/e2_log.md), [`e3_log.md`](docs/logs/e3_log.md), [`e4_log.md`](docs/logs/e4_log.md), [`e5_log.md`](docs/logs/e5_log.md), [`e6_log.md`](docs/logs/e6_log.md) e [`ablations_analysis_log.md`](docs/logs/ablations_analysis_log.md)
- Relatório final: [`docs/dlnn-final.pdf`](docs/dlnn-final.pdf) ([`main.tex`](docs/dlnn-final/main.tex))

## Licença

GPL-3.0. Veja [`LICENSE`](LICENSE).
