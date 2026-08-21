"""Etapa 3 — Quantização em int8 (com verificação).

Converte os pesos e as ativações do MLP treinado para ``int8`` por um esquema
simétrico por tensor, reimplementa a inferência em inteiros com acumulação em
``int32`` e mede a queda de acurácia float -> int8. É a transição de risco (B)
da cadeia de validação: a única em que os números têm o direito de mudar, por
isso o erro é medido, não assumido.

Esquema (simétrico por tensor):
    escala  s = max(|T|) / 127
    quant   T_q = clip(round(T / s), -127, 127)  ->  int8
    valor real reconstruído  ~=  T_q * s

As escalas dos pesos usam o máximo absoluto (os pesos não têm outliers). As
escalas das ativações são obtidas por calibração sobre um subconjunto, mas por
um percentil alto (99,9) em vez do máximo: a entrada padronizada tem outliers
extremos (pixels de borda quase constantes divididos por um desvio minúsculo),
e usar o máximo desperdiçaria quase toda a resolução do int8. Recortar esses
outliers preserva a precisão nos valores típicos. Isso altera apenas o cálculo
das escalas em Python; o forward em C permanece idêntico.

Consome os artefatos das Etapas 1–2. Produz:
- ``modelo/pesos_int8.npz`` — pesos int8, escalas de peso e de ativação;
- ``resultados/quant_pesos_hist.png`` — distribuição dos pesos por camada;
- ``resultados/quant_erro_hist.png`` — erro de quantização por camada;
- ``resultados/quant_matriz_confusao_int8.png`` — confusão do modelo int8;
- ``resultados/quant_resumo.xlsx`` — escalas, acurácias e decisões alteradas.
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from openpyxl import Workbook
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from treino import carregar_mnist, separar_treino_teste, DIR_MODELO
from referencia_float import carregar_artefatos, padronizar, relu, forward_float

RAIZ = DIR_MODELO.parent
RESULTADOS = RAIZ / "resultados"


# ----------------------------------------------------------------------------
# Núcleo: quantização e forward em int8
# ----------------------------------------------------------------------------
def escala_simetrica(T):
    """Calcula a escala simétrica por tensor: ``max(|T|) / 127``.

    Parâmetros
    ----------
    T : numpy.ndarray
        Tensor (matriz de pesos ou bloco de ativações).

    Retorna
    -------
    float
        A escala. Se o tensor for todo nulo, devolve 1.0 para evitar divisão
        por zero (o tensor quantizado será todo zero de qualquer modo).
    """
    m = float(np.max(np.abs(T)))
    return m / 127.0 if m > 0 else 1.0


def quantizar(T, s):
    """Converte um tensor para ``int8`` dada a escala ``s``.

    Aplica ``clip(round(T / s), -127, 127)`` e devolve ``int8``. O recorte em
    127 (e não 128) mantém o intervalo simétrico.

    Parâmetros
    ----------
    T : numpy.ndarray
        Tensor em ponto flutuante.
    s : float
        Escala de quantização.

    Retorna
    -------
    numpy.ndarray, dtype int8
        Tensor quantizado.
    """
    return np.clip(np.round(T / s), -127, 127).astype(np.int8)


def escala_por_percentil(T, percentil):
    """Calcula a escala de ativação recortando outliers por um percentil.

    Diferente de ``escala_simetrica`` (que usa o máximo), toma o percentil alto
    do valor absoluto como limite. Valores acima dele saturarão em ±127 na
    quantização — o que é desejável para as ativações, cujos raros outliers,
    se usados como escala, arruinariam a resolução dos valores típicos.

    Parâmetros
    ----------
    T : numpy.ndarray
        Bloco de ativações.
    percentil : float
        Percentil do valor absoluto usado como limite (ex.: 99.9).

    Retorna
    -------
    float
        A escala. Devolve 1.0 se o percentil for nulo.
    """
    v = float(np.percentile(np.abs(T), percentil))
    return v / 127.0 if v > 0 else 1.0


def calibrar_ativacoes(X_cal, pesos, percentil=99.9):
    """Mede as escalas das ativações rodando o forward float na calibração.

    As ativações a quantizar são a entrada de cada multiplicação matriz-vetor:
    a entrada padronizada (antes da 1ª camada) e a saída da ReLU (antes da 2ª).
    Cada escala é ``percentil(|.|)/127`` sobre o conjunto de calibração — um
    percentil alto, e não o máximo, para recortar outliers (ver docstring do
    módulo).

    Parâmetros
    ----------
    X_cal : numpy.ndarray, shape (n_cal, 784)
        Subconjunto de calibração (dados de treino).
    pesos : dict
        Pesos float da Etapa 1.
    percentil : float
        Percentil de recorte dos outliers de ativação.

    Retorna
    -------
    s_x : float
        Escala da entrada padronizada.
    s_h : float
        Escala da ativação oculta (pós-ReLU).
    """
    Xs = padronizar(X_cal, pesos["scaler_mean"], pesos["scaler_scale"])
    s_x = escala_por_percentil(Xs, percentil)
    h = relu(Xs @ pesos["W0"] + pesos["b0"])
    s_h = escala_por_percentil(h, percentil)
    return s_x, s_h


def quantizar_modelo(pesos, X_cal):
    """Quantiza pesos e calibra ativações, montando o pacote int8.

    Parâmetros
    ----------
    pesos : dict
        Pesos float da Etapa 1.
    X_cal : numpy.ndarray
        Subconjunto de calibração.

    Retorna
    -------
    dict
        Pacote com pesos int8 (``W0q``, ``W1q``), escalas de peso (``sW0``,
        ``sW1``), escalas de ativação (``sx``, ``sh``), vieses float (``b0``,
        ``b1``) e os parâmetros do scaler.
    """
    sW0 = escala_simetrica(pesos["W0"])
    sW1 = escala_simetrica(pesos["W1"])
    sx, sh = calibrar_ativacoes(X_cal, pesos)
    return {
        "W0q": quantizar(pesos["W0"], sW0), "sW0": sW0,
        "W1q": quantizar(pesos["W1"], sW1), "sW1": sW1,
        "sx": sx, "sh": sh,
        "b0": pesos["b0"], "b1": pesos["b1"],
        "scaler_mean": pesos["scaler_mean"], "scaler_scale": pesos["scaler_scale"],
    }


def forward_int8(X, q):
    """Executa a inferência em int8 com acumulação em int32.

    Espelha o que o C fará na placa: o scaler é aplicado em float; a entrada é
    quantizada; cada camada acumula produtos int8 em int32; o acumulador é
    reconvertido a real multiplicando pelas escalas; aplica-se ReLU; e a
    ativação é requantizada para a camada seguinte. Os vieses entram como
    inteiros no domínio do acumulador (``round(b/(s_ativacao*s_peso))``).

    Parâmetros
    ----------
    X : numpy.ndarray, shape (n, 784)
        Entradas brutas (pixels).
    q : dict
        Pacote int8 de ``quantizar_modelo``.

    Retorna
    -------
    numpy.ndarray, shape (n, 10)
        Logits reconstruídos em ponto flutuante (a decisão é o ``argmax``).
    """
    Xs = padronizar(X, q["scaler_mean"], q["scaler_scale"])
    xq = quantizar(Xs, q["sx"]).astype(np.int32)

    b0i = np.round(q["b0"] / (q["sx"] * q["sW0"])).astype(np.int32)
    acc1 = xq @ q["W0q"].astype(np.int32) + b0i          # int32
    z1 = acc1 * (q["sx"] * q["sW0"])                     # reconstrói o real
    h = np.maximum(0, z1)

    hq = quantizar(h, q["sh"]).astype(np.int32)
    b1i = np.round(q["b1"] / (q["sh"] * q["sW1"])).astype(np.int32)
    acc2 = hq @ q["W1q"].astype(np.int32) + b1i          # int32
    return acc2 * (q["sh"] * q["sW1"])                   # logits reais


def prever_int8(X, q, classes):
    """Classe prevista pela inferência int8 (argmax dos logits)."""
    return classes[np.argmax(forward_int8(X, q), axis=1)]


# ----------------------------------------------------------------------------
# Camada de verificação
# ----------------------------------------------------------------------------
def _fig_hist_pesos(pesos, caminho):
    """Grava histogramas das distribuições de pesos por camada.

    Mostra como os pesos se concentram em torno de zero — o que justifica a
    escala simétrica funcionar bem.

    Parâmetros
    ----------
    pesos : dict
        Pesos float.
    caminho : pathlib.Path
        Arquivo PNG de saída.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, (nome, W) in zip(axes, [("W0 (784x64)", pesos["W0"]), ("W1 (64x10)", pesos["W1"])]):
        ax.hist(W.ravel(), bins=60)
        ax.set_title(f"{nome} — float")
        ax.set_xlabel("valor do peso")
        ax.grid(alpha=0.3)
    fig.suptitle("Distribuição dos pesos por camada")
    fig.tight_layout()
    fig.savefig(caminho, dpi=120)
    plt.close(fig)


def _fig_hist_erro(pesos, q, caminho):
    """Grava histogramas do erro de quantização por camada.

    Erro = ``W - W_q * s``. Deve ser pequeno e centrado em zero; caudas largas
    indicariam perda relevante.

    Parâmetros
    ----------
    pesos : dict
        Pesos float.
    q : dict
        Pacote int8.
    caminho : pathlib.Path
        Arquivo PNG de saída.
    """
    erro0 = pesos["W0"] - q["W0q"].astype(np.float64) * q["sW0"]
    erro1 = pesos["W1"] - q["W1q"].astype(np.float64) * q["sW1"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, (nome, e) in zip(axes, [("W0", erro0), ("W1", erro1)]):
        ax.hist(e.ravel(), bins=60, color="#c0504d")
        ax.set_title(f"erro de quantização — {nome}")
        ax.set_xlabel("W - W_q * s")
        ax.grid(alpha=0.3)
    fig.suptitle("Erro de quantização por camada (deve ser pequeno)")
    fig.tight_layout()
    fig.savefig(caminho, dpi=120)
    plt.close(fig)


def _fig_confusao_int8(y_true, y_pred, caminho):
    """Grava a matriz de confusão do modelo int8 no teste."""
    disp = ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred))
    fig, ax = plt.subplots(figsize=(5.5, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Oranges")
    ax.set_title("Matriz de confusão — int8")
    fig.tight_layout()
    fig.savefig(caminho, dpi=120)
    plt.close(fig)


def _planilha_resumo(q, acc_float, acc_int8, decisoes_alteradas, y_te, pred_f, pred_q, caminho):
    """Grava planilha-resumo da quantização.

    Inclui as escalas, as acurácias float e int8, a queda, quantas decisões
    mudaram em relação ao float, e a acurácia por classe nos dois regimes.

    Parâmetros
    ----------
    q : dict
        Pacote int8 (para as escalas).
    acc_float, acc_int8 : float
        Acurácias no teste.
    decisoes_alteradas : int
        Número de amostras em que a classe int8 difere da float.
    y_te : numpy.ndarray
        Rótulos verdadeiros.
    pred_f, pred_q : numpy.ndarray
        Previsões float e int8.
    caminho : pathlib.Path
        Arquivo XLSX de saída.
    """
    wb = Workbook()
    ws = wb.active
    ws.title = "resumo"
    ws.append(["metrica", "valor"])
    ws.append(["escala_W0", q["sW0"]])
    ws.append(["escala_W1", q["sW1"]])
    ws.append(["escala_entrada_sx", q["sx"]])
    ws.append(["escala_oculta_sh", q["sh"]])
    ws.append(["acuracia_float", round(float(acc_float), 4)])
    ws.append(["acuracia_int8", round(float(acc_int8), 4)])
    ws.append(["queda_acuracia", round(float(acc_float - acc_int8), 4)])
    ws.append(["decisoes_alteradas_vs_float", int(decisoes_alteradas)])

    ws2 = wb.create_sheet("acuracia_por_classe")
    ws2.append(["classe", "acuracia_float", "acuracia_int8"])
    for d in range(10):
        m = y_te == d
        af = float(np.mean(pred_f[m] == y_te[m])) if m.any() else 0.0
        aq = float(np.mean(pred_q[m] == y_te[m])) if m.any() else 0.0
        ws2.append([d, round(af, 4), round(aq, 4)])
    wb.save(caminho)


# ----------------------------------------------------------------------------
# Orquestração
# ----------------------------------------------------------------------------
def main():
    """Quantiza, mede a queda de acurácia e grava artefatos e verificação."""
    RESULTADOS.mkdir(exist_ok=True)

    pesos, modelo = carregar_artefatos()
    classes = modelo.named_steps["mlp"].classes_

    X, y = carregar_mnist()
    X_tr, X_te, y_tr, y_te = separar_treino_teste(X, y)
    X_cal = X_tr[:5000]  # subconjunto de calibração das ativações

    q = quantizar_modelo(pesos, X_cal)

    pred_f = classes[np.argmax(forward_float(X_te, pesos), axis=1)]
    pred_q = prever_int8(X_te, q, classes)
    acc_float = accuracy_score(y_te, pred_f)
    acc_int8 = accuracy_score(y_te, pred_q)
    decisoes_alteradas = int(np.sum(pred_f != pred_q))

    print(f"Acuracia  float : {acc_float:.4f}")
    print(f"Acuracia  int8  : {acc_int8:.4f}")
    print(f"Queda           : {acc_float - acc_int8:.4f}")
    print(f"Decisoes alteradas vs float: {decisoes_alteradas}/{len(X_te)}")

    np.savez(
        DIR_MODELO / "pesos_int8.npz",
        W0q=q["W0q"], W1q=q["W1q"],
        sW0=q["sW0"], sW1=q["sW1"], sx=q["sx"], sh=q["sh"],
        b0=q["b0"], b1=q["b1"],
        scaler_mean=q["scaler_mean"], scaler_scale=q["scaler_scale"],
        acc_int8=np.array(acc_int8),
    )
    print(f"Pacote int8 salvo em: {DIR_MODELO / 'pesos_int8.npz'}")

    _fig_hist_pesos(pesos, RESULTADOS / "quant_pesos_hist.png")
    _fig_hist_erro(pesos, q, RESULTADOS / "quant_erro_hist.png")
    _fig_confusao_int8(y_te, pred_q, RESULTADOS / "quant_matriz_confusao_int8.png")
    _planilha_resumo(q, acc_float, acc_int8, decisoes_alteradas, y_te, pred_f, pred_q,
                     RESULTADOS / "quant_resumo.xlsx")
    print(f"Verificacao salva em: {RESULTADOS}")


if __name__ == "__main__":
    main()