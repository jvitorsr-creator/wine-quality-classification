# Classificação da Qualidade de Vinhos com Machine Learning

Projeto do **Tech Challenge – POSTECH Data Analytics (Fase 2)**.

## Objetivo
Prever se um vinho é de **alta qualidade** (nota ≥ 7) ou de **baixa/média qualidade** (nota < 7) a partir de suas características físico-químicas, apoiando enólogos e produtores na padronização da qualidade.

## Dados
`data/WineQT.csv` – Wine Quality Dataset (Kaggle): 1.143 amostras, 11 variáveis físico-químicas + `quality` + `Id`. Sem nulos; 125 duplicatas (ignorando o Id) removidas, restando **1.018 amostras**, das quais **~13%** são de alta qualidade.

## Estrutura
```
wine-quality-classification/
├── data/            # WineQT.csv
├── notebooks/       # 01_eda.ipynb, 02_modelagem.ipynb
├── src/             # preprocessing.py, modeling.py, evaluation.py
├── results/         # gráficos e métricas
├── presentation/    # storytelling.pptx (apresentação executiva)
├── requirements.txt
└── README.md
```

## Como executar
```bash
pip install -r requirements.txt
cd notebooks
jupyter notebook   # rode 01_eda.ipynb e depois 02_modelagem.ipynb (Restart & Run All)
```

## Pipeline
1. **Problema:** alvo binário `alta_qualidade = (quality >= 7)`.
2. **EDA:** distribuições, correlações justificadas, outliers (IQR), balanceamento e duplicatas.
3. **Pré-processamento:** remoção de duplicatas; split estratificado 80/20 antes das transformações; `log1p` nas variáveis assimétricas; `StandardScaler` dentro de `Pipeline` (sem vazamento). Outliers mantidos por serem plausíveis.
4. **Feature engineering:** razão SO₂ livre/total, acidez total e álcool × sulfatos.
5. **Modelos:** Regressão Logística (baseline interpretável) e Random Forest, com `class_weight` e `GridSearchCV` (5 folds estratificados, métrica PR-AUC).
6. **Avaliação:** precisão, recall, F1, ROC-AUC, PR-AUC, matriz de confusão e curva Precisão-Recall. O limiar é escolhido no treino (probabilidades out-of-fold).

## Resultados (teste: 204 amostras, 27 de alta qualidade)
| Modelo | Limiar | Precisão | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Regressão Logística | 0,50 | 0,364 | 0,741 | 0,488 | 0,891 | 0,486 |
| Regressão Logística | 0,64 (ajustado) | 0,472 | 0,630 | 0,540 | 0,891 | 0,486 |
| Random Forest | 0,50 | 0,607 | 0,630 | 0,618 | 0,918 | 0,711 |
| Random Forest | 0,40 (ajustado) | 0,475 | 0,704 | 0,567 | 0,918 | 0,711 |

- A **Random Forest** é melhor em ranqueamento (ROC-AUC 0,92 e PR-AUC 0,71, contra 0,89 e 0,49) e é a recomendada.
- A **Regressão Logística** tem recall alto com limiar 0,5, mas gera muitos falsos alarmes (precisão 0,36).
- O ajuste do limiar permite escolher entre mais recall (não perder vinhos premium) ou mais precisão (menos revisões).
- As features criadas deram ganho pequeno (PR-AUC da Random Forest 0,711 com e 0,703 sem).
- **Cenário complementar (nota ≥ 6):** a classe positiva sobe para ~54% e o problema fica mais equilibrado (PR-AUC ≈ 0,83–0,84; ROC-AUC ≈ 0,81).

> Atenção: o teste tem só 27 vinhos de alta qualidade; as métricas variam com a semente do split.

## Interpretação
- **Variáveis mais influentes:** na Random Forest (importância por permutação), sulfatos (≈ 0,22) e álcool (≈ 0,20), seguidos de acidez volátil (≈ 0,10), SO₂ total e ácido cítrico. Na Regressão Logística, álcool (+) e acidez volátil (−) têm os maiores efeitos, além de álcool × sulfatos (+) e densidade, cloretos e SO₂ total (−).
- **Implicações para a produção:** controlar a acidez volátil; acompanhar o ponto de colheita/maturação (teor alcoólico); monitorar sulfatos e SO₂ total em faixas adequadas; usar o modelo como triagem de lotes, com o enólogo decidindo.
- **Limites:** correlação não é causalidade; base pequena; nota sensorial subjetiva.

**Autor:** João Vitor Silvestre Ribeiro
