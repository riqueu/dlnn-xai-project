# dlnn-xai-project

Projeto da disciplina de Redes Neurais e Aprendizado Profundo comparando a fidelidade de métodos de explicabilidade post-hoc (Grad-CAM, Attention Rollout, LIME, Partition SHAP e Integrated Gradients) em uma Vision GNN isotrópica (ViG-Ti) e em um Vision Transformer (DeiT-Tiny) ajustados no Imagenette.

## Estrutura

```text
configs/       Hiperparâmetros e caminhos dos experimentos (default.yaml)
data/          Dataset local (ignorado no git) e splits determinísticos em data/splits/
checkpoints/   Pesos pré-treinados e checkpoints de fine-tuning (ignorados no git)
docs/          Proposta (dlnn-proposta/), logs experimentais (logs/) e relatório final (dlnn-final/)
src/           Módulos de dados, modelos (ViG-Ti, DeiT-Tiny), XAI e métricas
scripts/       Pontos de entrada do pipeline experimental
results/       Mapas salvos (ignorados), CSVs de métricas e figuras
```

## Ambiente

Requer Python 3.11 a 3.13 (GPU com CUDA recomendada).

Com `uv`:

```bash
uv sync
```

Com `venv` e `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Pipeline

Todos os scripts leem [`configs/default.yaml`](configs/default.yaml) por padrão:

```bash
python scripts/prepare_data.py
python scripts/train.py --model all
python scripts/run_xai.py --model all --method all
python scripts/evaluate.py --experiment all
python scripts/analyze.py
```

## Proposta

A proposta completa está em [`docs/dlnn-proposta.pdf`](docs/dlnn-proposta.pdf) (fonte em [`docs/dlnn-proposta/main.tex`](docs/dlnn-proposta/main.tex)).

## Licença

GPL-3.0. Veja [`LICENSE`](LICENSE).
