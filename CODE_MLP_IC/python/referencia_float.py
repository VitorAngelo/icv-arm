"""Etapa 2 — Referência do forward pass em ponto flutuante (com verificação).

Reimplementa a inferência do MLP à mão, em NumPy, e verifica que ela reproduz
o ``predict`` do scikit-learn em todo o conjunto de teste — a comparação (A) da
cadeia de validação, feita dentro do Python. Produz, além disso, uma camada de
verificação em ``resultados/``: um gráfico de concordância de probabilidades e
uma planilha de comparação.

Consome os artefatos da Etapa 1 (``modelo/pesos_float.npz`` e
``modelo/mlp_mnist.joblib``).
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from openpyxl import Workbook
import joblib

from treino import carregar_mnist, separar_treino_teste, DIR_MODELO

RESULTADOS = DIR_MODELO.parent / "resultados"


# ----------------------------------------------------------------------------
# Núcleo: forward pass à mão
# ----------------------------------------------------------------------------
def carregar_artefatos():
    """Carrega os pesos crus e o Pipeline treinado da Etapa 1.

    Retorna
    -------
    pesos : dict
        Chaves ``W0``, ``b0``, ``W1``, ``b1``, ``scaler_mean``, ``scaler_scale``.
    modelo : sklearn.pipeline.Pipeline
        Pipeline treinado, usado como referência-verdade via ``predict``.
    """
    z = np.load(DIR_MODELO / "pesos_float.npz")
    pesos = {k: z[k] for k in ("W0", "b0", "W1", "b1", "scaler_mean", "scaler_scale")}
    modelo = joblib.load(DIR_MODELO / "mlp_mnist.joblib")
    return pesos, modelo


def padronizar(X, media, desvio):
    """Aplica a padronização do StandardScaler: (X - média) / desvio.

    Reproduz à mão o primeiro passo do Pipeline, por característica (pixel).

    Parâmetros
    ----------
    X : numpy.ndarray, shape (n, 784)
    media, desvio : numpy.ndarray, shape (784,)

    Retorna
    -------
    numpy.ndarray, shape (n, 784)
        Entradas padronizadas.
    """
    return (X - media) / desvio


def relu(z):
    """Ativação ReLU: ``max(0, z)`` elemento a elemento."""
    return np.maximum(0, z)


def softmax(z):
    """Softmax numericamente estável por linha.

    Usada apenas na verificação, para comparar probabilidades com o
    ``predict_proba`` do scikit-learn. Não é necessária para a decisão de
    classe (o ``argmax`` dos logits já basta).

    Parâmetros
    ----------
    z : numpy.ndarray, shape (n, 10)
        Logits.

    Retorna
    -------
    numpy.ndarray, shape (n, 10)
        Probabilidades por classe.
    """
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def forward_float(X, pesos):
    """Executa o forward pass completo em ponto flutuante.

    Reproduz o Pipeline na ordem correta (padronização, camada oculta com
    ReLU, camada de saída), seguindo a convenção do scikit-learn em que
    ``coefs_[i]`` tem forma (n_entradas, n_saídas), logo a conta é ``X @ W + b``.

    Parâmetros
    ----------
    X : numpy.ndarray, shape (n, 784)
    pesos : dict
        Pesos retornados por ``carregar_artefatos``.

    Retorna
    -------
    numpy.ndarray, shape (n, 10)
        Logits de cada amostra.
    """
    Xs = padronizar(X, pesos["scaler_mean"], pesos["scaler_scale"])
    h = relu(Xs @ pesos["W0"] + pesos["b0"])
    return h @ pesos["W1"] + pesos["b1"]


def prever(X, pesos, classes):
    """Devolve a classe prevista para cada amostra (argmax dos logits)."""
    logits = forward_float(X, pesos)
    return classes[np.argmax(logits, axis=1)]


# ----------------------------------------------------------------------------
# Camada de verificação
# ----------------------------------------------------------------------------
def _fig_concordancia(prob_minha, prob_sklearn, caminho):
    """Grava dispersão prob. minha × prob. scikit-learn (deve cair na diagonal).

    Se a reimplementação estiver correta, todos os pontos se alinham à reta
    y = x, evidência visual de que os dois cálculos coincidem.

    Parâmetros
    ----------
    prob_minha, prob_sklearn : numpy.ndarray, shape (n, 10)
        Probabilidades por classe de cada implementação.
    caminho : pathlib.Path
        Arquivo PNG de saída.
    """
    idx = np.random.default_rng(0).choice(prob_minha.size, 4000, replace=False)
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.scatter(prob_sklearn.ravel()[idx], prob_minha.ravel()[idx], s=5, alpha=0.3)
    ax.plot([0, 1], [0, 1], "r--", lw=1)
    ax.set_xlabel("probabilidade — scikit-learn")
    ax.set_ylabel("probabilidade — minha referência")
    ax.set_title("Concordância de probabilidades")
    fig.tight_layout()
    fig.savefig(caminho, dpi=120)
    plt.close(fig)


def _planilha_comparacao(pred_sklearn, pred_minha, maxdiff, caminho, n=50):
    """Grava planilha comparando as decisões e o resumo da verificação.

    Parâmetros
    ----------
    pred_sklearn, pred_minha : numpy.ndarray
        Classes previstas por cada implementação.
    maxdiff : float
        Maior diferença absoluta de probabilidade entre as implementações.
    caminho : pathlib.Path
        Arquivo XLSX de saída.
    n : int
        Número de linhas de amostra detalhadas.
    """
    wb = Workbook()
    ws = wb.active
    ws.title = "comparacao"
    ws.append(["indice", "sklearn", "referencia", "coincide"])
    for i in range(min(n, len(pred_sklearn))):
        ws.append([i, int(pred_sklearn[i]), int(pred_minha[i]),
                   bool(pred_sklearn[i] == pred_minha[i])])
    ws.append([])
    ws.append(["total_coincidencias", int((pred_sklearn == pred_minha).sum())])
    ws.append(["total_amostras", int(len(pred_sklearn))])
    ws.append(["maior_dif_probabilidade", float(maxdiff)])
    wb.save(caminho)


# ----------------------------------------------------------------------------
# Orquestração
# ----------------------------------------------------------------------------
def main():
    """Valida a reimplementação contra o ``predict`` e grava a verificação."""
    RESULTADOS.mkdir(exist_ok=True)

    pesos, modelo = carregar_artefatos()
    X, y = carregar_mnist()
    _, X_te, _, _ = separar_treino_teste(X, y)

    classes = modelo.named_steps["mlp"].classes_
    pred_minha = prever(X_te, pesos, classes)
    pred_sklearn = modelo.predict(X_te)

    iguais = int(np.sum(pred_minha == pred_sklearn))
    total = len(X_te)
    print(f"Coincidencias com o predict do sklearn: {iguais}/{total}")

    prob_minha = softmax(forward_float(X_te, pesos))
    prob_sklearn = modelo.predict_proba(X_te)
    maxdiff = float(np.max(np.abs(prob_minha - prob_sklearn)))
    print(f"Maior diferenca de probabilidade: {maxdiff:.2e}")

    if iguais == total:
        print("SUCESSO: a referencia em float reproduz o predict em todas as amostras.")
    else:
        print("FALHA: ha divergencia de decisao — investigar antes de seguir.")

    _fig_concordancia(prob_minha, prob_sklearn, RESULTADOS / "ref_concordancia.png")
    _planilha_comparacao(pred_sklearn, pred_minha, maxdiff, RESULTADOS / "ref_comparacao.xlsx")
    print(f"Verificacao salva em: {RESULTADOS}")


if __name__ == "__main__":
    main()