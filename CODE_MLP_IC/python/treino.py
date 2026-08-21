"""Etapa 1 — Treinamento do MLP em MNIST (com camada de verificação).

Treina uma rede totalmente conectada (MLP) de uma camada oculta sobre o MNIST,
mede a acurácia baseline em ponto flutuante e persiste os artefatos consumidos
pelas etapas seguintes. Além do treino, produz uma camada de verificação —
imagens, matriz de confusão e planilha — na pasta ``resultados/``, para
inspeção visual do acerto/erro.

Artefatos em ``modelo/``:
- ``mlp_mnist.joblib`` — o Pipeline treinado (scaler + rede);
- ``pesos_float.npz`` — pesos, vieses e parâmetros do scaler em ``float64``.

Artefatos em ``resultados/``:
- ``treino_curva_perda.png`` — perda de treino por iteração;
- ``treino_matriz_confusao.png`` — matriz de confusão no teste;
- ``treino_amostras.png`` — dígitos de teste com previsto/verdadeiro;
- ``treino_metricas.xlsx`` — precisão/revocação/F1 por classe e acurácia.
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend sem janela: apenas grava arquivos
import matplotlib.pyplot as plt
from openpyxl import Workbook
from sklearn.datasets import fetch_openml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

RAIZ = Path(__file__).resolve().parent.parent
DIR_MODELO = RAIZ / "modelo"
RESULTADOS = RAIZ / "resultados"
SEMENTE = 42


# ----------------------------------------------------------------------------
# Núcleo: dados, modelo, treino
# ----------------------------------------------------------------------------
def carregar_mnist():
    """Baixa o MNIST e o devolve como arrays NumPy prontos para treino.

    Obtém o conjunto ``mnist_784`` via ``fetch_openml``: 70.000 imagens de
    dígitos manuscritos já achatadas em vetores de 784 valores. O download
    ocorre só na primeira execução; depois usa-se o cache local.

    Retorna
    -------
    X : numpy.ndarray, shape (70000, 784), dtype float64
        Intensidade de cada pixel de cada imagem.
    y : numpy.ndarray, shape (70000,), dtype int64
        Classe de cada imagem (0 a 9).
    """
    X, y = fetch_openml(
        "mnist_784", version=1, as_frame=False, return_X_y=True, parser="liac-arff"
    )
    return X.astype(np.float64), y.astype(np.int64)


def separar_treino_teste(X, y):
    """Aplica a divisão canônica do MNIST: 60.000 treino, 10.000 teste.

    A partição é posicional (sem embaralhar), o que mantém a acurácia
    comparável à de outros trabalhos sobre o mesmo conjunto.

    Parâmetros
    ----------
    X : numpy.ndarray, shape (70000, 784)
    y : numpy.ndarray, shape (70000,)

    Retorna
    -------
    X_tr, X_te, y_tr, y_te : numpy.ndarray
        Conjuntos de treino (60000) e teste (10000).
    """
    return X[:60000], X[60000:], y[:60000], y[60000:]


def montar_modelo():
    """Constrói o Pipeline (StandardScaler + MLPClassifier) a treinar.

    O Pipeline amarra normalização e rede num só objeto, garantindo que o
    scaler — parte integrante do modelo — seja sempre aplicado antes da rede.
    Arquitetura ``784 -> 64 -> 10`` (uma camada oculta, ativação ``relu``).

    Retorna
    -------
    sklearn.pipeline.Pipeline
        Pipeline não treinado, pronto para ``fit``.
    """
    scaler = StandardScaler()
    mlp = MLPClassifier(
        hidden_layer_sizes=(64,),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=128,
        learning_rate_init=1e-3,
        max_iter=30,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
        random_state=SEMENTE,
        verbose=True,
    )
    return Pipeline([("scaler", scaler), ("mlp", mlp)])


# ----------------------------------------------------------------------------
# Camada de verificação (imagens, diagrama, planilha)
# ----------------------------------------------------------------------------
def _fig_curva_perda(mlp, caminho):
    """Grava a curva de perda de treino por iteração.

    Parâmetros
    ----------
    mlp : MLPClassifier
        Rede treinada, de onde se lê ``loss_curve_``.
    caminho : pathlib.Path
        Arquivo PNG de saída.
    """
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(mlp.loss_curve_, marker="o", ms=3)
    ax.set_xlabel("iteração")
    ax.set_ylabel("perda de treino")
    ax.set_title("Curva de perda (deve cair)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(caminho, dpi=120)
    plt.close(fig)


def _fig_matriz_confusao(y_true, y_pred, caminho):
    """Grava a matriz de confusão 10x10 do conjunto de teste.

    A diagonal são os acertos; fora da diagonal, os erros — quais dígitos são
    confundidos com quais.

    Parâmetros
    ----------
    y_true, y_pred : numpy.ndarray
        Rótulos verdadeiros e previstos.
    caminho : pathlib.Path
        Arquivo PNG de saída.
    """
    disp = ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred))
    fig, ax = plt.subplots(figsize=(5.5, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Matriz de confusão — teste")
    fig.tight_layout()
    fig.savefig(caminho, dpi=120)
    plt.close(fig)


def _fig_amostras(X, y_true, y_pred, caminho, n=12):
    """Grava uma grade de dígitos de teste com previsto e verdadeiro.

    Verde indica acerto; vermelho, erro. É a inspeção visual mais direta de
    que a rede está classificando dígitos plausíveis.

    Parâmetros
    ----------
    X : numpy.ndarray, shape (n_amostras, 784)
        Imagens de teste.
    y_true, y_pred : numpy.ndarray
        Rótulos verdadeiros e previstos.
    caminho : pathlib.Path
        Arquivo PNG de saída.
    n : int
        Quantidade de dígitos a exibir.
    """
    fig, axes = plt.subplots(2, n // 2, figsize=(9, 3.6))
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(X[i].reshape(28, 28), cmap="gray")
        cor = "green" if y_pred[i] == y_true[i] else "red"
        ax.set_title(f"{y_pred[i]} ({y_true[i]})", color=cor, fontsize=10)
        ax.axis("off")
    fig.suptitle("previsto (verdadeiro) — verde = acerto, vermelho = erro")
    fig.tight_layout()
    fig.savefig(caminho, dpi=120)
    plt.close(fig)


def _planilha_metricas(y_true, y_pred, acc, caminho):
    """Grava planilha com precisão, revocação, F1 e suporte por classe.

    Parâmetros
    ----------
    y_true, y_pred : numpy.ndarray
        Rótulos verdadeiros e previstos.
    acc : float
        Acurácia global no teste.
    caminho : pathlib.Path
        Arquivo XLSX de saída.
    """
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    wb = Workbook()
    ws = wb.active
    ws.title = "metricas_por_classe"
    ws.append(["classe", "precisao", "revocacao", "f1", "suporte"])
    for d in range(10):
        v = rep[str(d)]
        ws.append([d, round(v["precision"], 4), round(v["recall"], 4),
                   round(v["f1-score"], 4), int(v["support"])])
    ws.append([])
    ws.append(["acuracia_global", round(float(acc), 4)])
    wb.save(caminho)


# ----------------------------------------------------------------------------
# Orquestração
# ----------------------------------------------------------------------------
def main():
    """Treina, mede a acurácia baseline, salva artefatos e a verificação."""
    DIR_MODELO.mkdir(exist_ok=True)
    RESULTADOS.mkdir(exist_ok=True)

    print("Carregando MNIST (baixa e cacheia na primeira vez)...")
    X, y = carregar_mnist()
    X_tr, X_te, y_tr, y_te = separar_treino_teste(X, y)
    print(f"treino: {X_tr.shape}, teste: {X_te.shape}")

    modelo = montar_modelo()
    print("Treinando...")
    modelo.fit(X_tr, y_tr)

    y_pred = modelo.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    print(f"\nAcuracia baseline (float) no teste: {acc:.4f}")

    scaler = modelo.named_steps["scaler"]
    mlp = modelo.named_steps["mlp"]

    joblib.dump(modelo, DIR_MODELO / "mlp_mnist.joblib")
    np.savez(
        DIR_MODELO / "pesos_float.npz",
        W0=mlp.coefs_[0], b0=mlp.intercepts_[0],
        W1=mlp.coefs_[1], b1=mlp.intercepts_[1],
        scaler_mean=scaler.mean_, scaler_scale=scaler.scale_,
        acc_baseline=np.array(acc),
    )
    print(f"Artefatos salvos em: {DIR_MODELO}")

    _fig_curva_perda(mlp, RESULTADOS / "treino_curva_perda.png")
    _fig_matriz_confusao(y_te, y_pred, RESULTADOS / "treino_matriz_confusao.png")
    _fig_amostras(X_te, y_te, y_pred, RESULTADOS / "treino_amostras.png")
    _planilha_metricas(y_te, y_pred, acc, RESULTADOS / "treino_metricas.xlsx")
    print(f"Verificacao salva em: {RESULTADOS}")


if __name__ == "__main__":
    main()