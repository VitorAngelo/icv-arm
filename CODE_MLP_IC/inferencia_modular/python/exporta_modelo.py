"""Etapa 5 - Exportacao do modelo para a implementacao C modular.

PAPEL DESTE SCRIPT NA CADEIA
============================
    treino.py            treina o Pipeline (StandardScaler + MLPClassifier 784-64-10)
                         e salva modelo/pesos_float.npz
    referencia_float.py  prova que a formula manual reproduz o predict (10000/10000)
    quantiza.py          calcula escalas e pesos int8 e salva modelo/pesos_int8.npz
    exporta_modelo.py    (ESTE) transforma tudo em arquivos que o C le e gera as
                         respostas esperadas ("referencias") para cada modulo C

Nada e retreinado e nao se usa scikit-learn: apenas NumPy sobre os artefatos
congelados, para que os numeros nao mudem entre maquinas.

ARQUIVOS GERADOS (em modelo/exportado/)
=======================================
    mlp_pesos_float.h     pesos float32 originais e parametros do scaler
    mlp_pesos_int16.h     tabela de entrada, pesos int16, bias int32, M e N
    mlp_pesos_int8.h      tabela de entrada, pesos int8, bias int32, M e N
    teste_u8.bin          10.000 registros de 785 bytes: [rotulo][784 pixels]
    ref_float32.bin       logits float32 que o C deve produzir  (10000 x 10, '<f4')
    ref_int16.bin         logits int64 que o C deve produzir    (10000 x 10, '<i8')
    ref_int8.bin          logits int32 que o C deve produzir    (10000 x 10, '<i4')
    ref_python_pred.bin   classe prevista pelo modelo Python original (10000, uint8)
    relatorio_exportacao.txt  todas as verificacoes abaixo

MATEMATICA DO MODELO (valores reais)
====================================
    x = (pixel - media) / desvio              padronizacao (StandardScaler)
    a = W0 . x + b0                           camada oculta, 64 neuronios
    h = max(a, 0)                             ReLU
    z = W1 . h + b1                           camada de saida, 10 logits
    classe = argmax(z)                        softmax desnecessaria: e monotona

QUANTIZACAO SIMETRICA (mesmo metodo para int8 e int16; LIM = 127 ou 32767)
=========================================================================
Cada tensor real T e representado por um inteiro T_q e uma escala s > 0:
    T  ~=  T_q * s,      T_q = clip(round(T / s), -LIM, LIM)

    s_W0 = max|W0| / LIM                      escala dos pesos (pior caso cabe)
    s_W1 = max|W1| / LIM
    s_x  = percentil_99,9(|x|) / LIM          escala da entrada (calibrada no treino)
    s_h  = percentil_99,9(|h|) / LIM          escala da ativacao oculta

Substituindo T = T_q * s nas formulas:
    a = W0 . x + b0 = s_x*s_W0 * (W0_q . x_q + b0_q),   b0_q = round(b0 / (s_x*s_W0))
Logo o acumulador inteiro A = W0_q . x_q + b0_q vale a / (s_x*s_W0).
    h_q = round(h / s_h) = round(max(A, 0) * s_x*s_W0/s_h)
O fator real s_x*s_W0/s_h e trocado por M / 2^N (inteiros), eliminando o float:
    h_q = min((max(A, 0) * M + 2^(N-1)) >> N, LIM)
Na saida: z = s_h*s_W1 * (W1_q . h_q + b1_q). Como s_h*s_W1 > 0 e comum as
10 classes, argmax(z) = argmax(W1_q . h_q + b1_q): o C decide no inteiro.

Arredondamento: metade para longe de zero em toda quantizacao. Como a ReLU
e aplicada ao acumulador antes da requantizacao (max(0, s*A) = s*max(0, A)
para s > 0), o deslocamento com +2^(N-1) coincide exatamente com essa regra.

Escalas int16: mesmo metodo do int8 trocando 127 por 32767. O limiar de
recorte das ativacoes e o mesmo percentil ja calibrado (s8 * 127), logo
s16 = s8 * 127 / 32767; os pesos usam max|W| / 32767.

Disposicao das matrizes nos headers: uma linha por neuronio (W0[j*784 + i],
W1[j*64 + i]). O scikit-learn guarda coefs_[0] com forma (784, 64); a matriz
e transposta para que os 784 pesos de cada neuronio fiquem contiguos na
memoria, como exige o produto escalar vetorizado (NEON).

Uso (WSL, ambiente virtual ativo, na raiz do projeto):
    python3 python/exporta_modelo.py
"""

import math
import sys
from math import comb
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------
# Caminhos e constantes
# ----------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent.parent
DIR_MODELO = RAIZ / "modelo"
DIR_SAIDA = DIR_MODELO / "exportado"

ENTRADAS, OCULTA, SAIDAS = 784, 64, 10
N_TESTE = 10000
BLOCO = 250          # imagens por bloco na simulacao float32 (limita a RAM)
LIM8 = 127           # int8 simetrico: [-127, 127] (o -128 nao e usado)
LIM16 = 32767        # int16 simetrico: [-32767, 32767]
INT32_MAX = 2**31 - 1
INT64_MAX = 2**63 - 1
PIXEL_EXEMPLO = 14 * 28 + 14   # pixel central da imagem 28x28, usado nos comentarios

_relatorio = []
_falhas = []
_alertas = []


# ----------------------------------------------------------------------------
# Registro de verificacoes
# ----------------------------------------------------------------------------
def log(texto=""):
    """Imprime e guarda a linha para o relatorio_exportacao.txt."""
    print(texto)
    _relatorio.append(texto)


def verificar(nome, ok, detalhe="", fatal=True):
    """Registra uma verificacao como OK, FALHA (interrompe) ou ALERTA (nao interrompe)."""
    if ok:
        estado = "[OK]    "
    elif fatal:
        estado = "[FALHA] "
        _falhas.append(nome)
    else:
        estado = "[ALERTA]"
        _alertas.append(nome)
    log(f"{estado} {nome}" + (f" -> {detalhe}" if detalhe else ""))
    return ok


# ----------------------------------------------------------------------------
# Aritmetica de quantizacao
# ----------------------------------------------------------------------------
def arredondar_meio_longe_zero(v):
    """Arredondamento exato, metade para longe de zero (o mesmo do roundf do C).

    np.round arredonda metades para o par (0.5 -> 0, 1.5 -> 2); aqui 0.5 -> 1 e
    -0.5 -> -1. A parte fracionaria a - floor(a) e calculada sem erro em float64.
    """
    v = np.asarray(v, dtype=np.float64)
    a = np.abs(v)
    r = np.floor(a)
    r = r + ((a - r) >= 0.5)
    return np.copysign(r, v)


def quantizar(T, s, lim):
    """T_q = clip(round(T / s), -lim, lim), em int64."""
    return np.clip(arredondar_meio_longe_zero(np.asarray(T, np.float64) / s), -lim, lim).astype(np.int64)


def escolher_multiplicador(mult, a_max):
    """Troca o fator real mult por M / 2^N, com M e N inteiros.

    Escolhe o maior N (melhor precisao) tal que:
      - M = round(mult * 2^N) cabe em int32 (M <= 2^31 - 1);
      - a_max * M + 2^(N-1) cabe em int64, onde a_max e o maior acumulador
        possivel no pior caso (nao apenas o observado nos dados).
    """
    escolhido = None
    for n in range(1, 63):
        m = math.floor(mult * (2.0 ** n) + 0.5)
        if m > INT32_MAX:
            break
        if m == 0:
            continue
        if a_max * m + (1 << (n - 1)) > INT64_MAX:
            break
        escolhido = (m, n)
    if escolhido is None:
        raise RuntimeError("Nao existe par (M, N) valido para o multiplicador.")
    return escolhido


def requantizar(acc, m, n, lim):
    """h_q = min((max(acc, 0) * M + 2^(N-1)) >> N, lim), exato em int64."""
    a = np.maximum(acc.astype(np.int64), 0)
    return np.minimum((a * np.int64(m) + np.int64(1 << (n - 1))) >> np.int64(n), lim)


def produto_inteiro_exato(A, W):
    """A @ W para inteiros cujas somas ficam abaixo de 2^53 (float64 e exato nesse caso)."""
    limite = np.abs(A).max(axis=0, initial=0).astype(np.float64) @ np.abs(W).astype(np.float64)
    if np.max(limite) >= 2.0**53:
        raise RuntimeError("Produto inteiro excede 2^53; float64 deixaria de ser exato.")
    return np.rint(A.astype(np.float64) @ W.astype(np.float64)).astype(np.int64)


def mcnemar_exato(b, c):
    """p-valor bilateral exato do teste de McNemar: 2 * P(X <= min(b, c)), X ~ Bin(b + c, 1/2).

    b e c sao as amostras em que apenas um dos dois classificadores acerta.
    Hipotese nula: os dois tem a mesma taxa de erro; b e c diferem so por acaso.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def pareado(pred_a, pred_b, y):
    """Compara duas previsoes amostra a amostra: (discordantes, B corrige, B erra novo, ambos errados, p)."""
    a_ok = pred_a == y
    b_ok = pred_b == y
    corrige = int(np.sum(~a_ok & b_ok))
    erra_novo = int(np.sum(a_ok & ~b_ok))
    ambos = int(np.sum(~a_ok & ~b_ok & (pred_a != pred_b)))
    return int(np.sum(pred_a != pred_b)), corrige, erra_novo, ambos, mcnemar_exato(corrige, erra_novo)


def comparar_pareado(rotulo, pred_a, pred_b, y):
    """Registra no relatorio a comparacao pareada A x B."""
    _, corrige, erra_novo, ambos, p = pareado(pred_a, pred_b, y)
    log(f"  {rotulo:<36} liquido {corrige - erra_novo:+4d} | B corrige {corrige:3d} | "
        f"B erra novo {erra_novo:3d} | ambos errados {ambos:2d} | p = {p:.4f}")
    return p


# ----------------------------------------------------------------------------
# Leitura e validacao dos artefatos
# ----------------------------------------------------------------------------
def carregar():
    """Le os artefatos congelados das Etapas 1 e 3 e o conjunto de teste."""
    f = np.load(DIR_MODELO / "pesos_float.npz")
    q = np.load(DIR_MODELO / "pesos_int8.npz")
    F = {k: f[k] for k in ("W0", "b0", "W1", "b1", "scaler_mean", "scaler_scale")}
    F["acc_baseline"] = float(f["acc_baseline"])
    Q = {k: q[k] for k in ("W0q", "W1q", "b0", "b1", "scaler_mean", "scaler_scale")}
    for k in ("sx", "sW0", "sh", "sW1", "acc_int8"):
        Q[k] = float(q[k])

    X = np.fromfile(DIR_MODELO / "X_te.bin", dtype="<f4")
    y = np.fromfile(DIR_MODELO / "y_te.bin", dtype="<i4")
    return F, Q, X, y


def validar_artefatos(F, Q, X, y):
    log("=" * 70)
    log("1. INTEGRIDADE DOS ARTEFATOS")
    log("=" * 70)
    verificar("X_te.bin com 10000 x 784 float32", X.size == N_TESTE * ENTRADAS, f"{X.size} valores")
    verificar("y_te.bin com 10000 int32", y.size == N_TESTE, f"{y.size} valores")
    X = X.reshape(N_TESTE, ENTRADAS)
    verificar("Pixels inteiros em [0, 255] (conversao para uint8 sem perda)",
              bool(np.all(X == np.round(X)) and X.min() >= 0 and X.max() <= 255),
              f"min={X.min()}, max={X.max()}")
    verificar("Rotulos em [0, 9]", bool(y.min() >= 0 and y.max() <= 9))

    verificar("Scaler identico nos dois .npz",
              bool(np.array_equal(F["scaler_mean"], Q["scaler_mean"])
                   and np.array_equal(F["scaler_scale"], Q["scaler_scale"])))
    verificar("Bias float identicos nos dois .npz",
              bool(np.array_equal(F["b0"], Q["b0"]) and np.array_equal(F["b1"], Q["b1"])))
    verificar("Nenhuma escala do scaler nula", bool(np.all(F["scaler_scale"] != 0)))

    for nome, W, s, Wq in (("W0", F["W0"], Q["sW0"], Q["W0q"]), ("W1", F["W1"], Q["sW1"], Q["W1q"])):
        s_recalc = float(np.max(np.abs(W))) / 127.0
        verificar(f"Escala {nome} = max|{nome}|/127", abs(s_recalc - s) <= 1e-12 * s,
                  f"diferenca relativa {abs(s_recalc - s) / s:.2e}")
        Wq_recalc = np.clip(np.round(W / s), -127, 127).astype(np.int64)
        dif = int(np.sum(Wq_recalc != Wq.astype(np.int64)))
        verificar(f"{nome}q do .npz = quantizacao de {nome} float", dif == 0, f"{dif} elementos diferentes")
    return X.astype(np.uint8), y.astype(np.int64)


# ----------------------------------------------------------------------------
# Caminho float
# ----------------------------------------------------------------------------
def forward_float64(Xu, F):
    """Modelo Python original (mesma formula validada contra o predict na Etapa 2)."""
    Xs = (Xu.astype(np.float64) - F["scaler_mean"]) / F["scaler_scale"]
    h = np.maximum(Xs @ F["W0"] + F["b0"], 0.0)
    return h, h @ F["W1"] + F["b1"]


def ativacao_oculta_recortada(Xu, F, lim_x, lim_h):
    """Ativacao oculta float64 com os mesmos recortes das versoes inteiras, sem arredondamento.

    Isola o erro de resolucao (arredondamento) do erro de recorte (saturacao),
    que e identico no int8 e no int16 por construcao.
    """
    Xs = np.clip((Xu.astype(np.float64) - F["scaler_mean"]) / F["scaler_scale"], -lim_x, lim_x)
    return np.clip(Xs @ F["W0"] + F["b0"], 0.0, lim_h)


def simular_float32_c(Xu, P):
    """Replica bit a bit o C float32.

    Float nao e associativo: (a + b) + c pode diferir de a + (b + c) no ultimo
    bit. Por isso a simulacao soma na mesma ordem do C (bias primeiro, depois
    i = 0, 1, 2, ...) com np.add.accumulate, que e estritamente sequencial.
    """
    mu, sc = P["media"], P["escala"]
    W0, b0, W1, b1 = P["W0"], P["b0"], P["W1"], P["b1"]
    logits = np.empty((N_TESTE, SAIDAS), dtype=np.float32)
    for ini in range(0, N_TESTE, BLOCO):
        p = Xu[ini:ini + BLOCO].astype(np.float32)
        n = p.shape[0]
        xs = (p - mu) / sc
        seq = np.empty((n, ENTRADAS + 1, OCULTA), dtype=np.float32)
        seq[:, 0, :] = b0
        np.multiply(xs[:, :, None], W0[None, :, :], out=seq[:, 1:, :])
        acc = np.add.accumulate(seq, axis=1, dtype=np.float32)[:, -1, :]
        h = np.where(acc > np.float32(0), acc, np.float32(0)).astype(np.float32)
        seq2 = np.empty((n, OCULTA + 1, SAIDAS), dtype=np.float32)
        seq2[:, 0, :] = b1
        np.multiply(h[:, :, None], W1[None, :, :], out=seq2[:, 1:, :])
        logits[ini:ini + n] = np.add.accumulate(seq2, axis=1, dtype=np.float32)[:, -1, :]
        if (ini + n) % 2500 == 0:
            print(f"  simulacao float32: {ini + n}/{N_TESTE}", flush=True)
    return logits


# ----------------------------------------------------------------------------
# Caminhos inteiros
# ----------------------------------------------------------------------------
def construir_inteiro(F, sx, sw0, sh, sw1, lim, rotulo):
    """Monta LUT, pesos, bias e (M, N) de um caminho inteiro com limite simetrico lim.

    LUT: para cada pixel i e cada valor possivel p (0 a 255), guarda
    x_q = clip(round(((p - media[i]) / desvio[i]) / s_x)). Como o pixel so
    assume 256 valores, a tabela e uma copia exata da conta, nao uma aproximacao.
    """
    pix = np.arange(256, dtype=np.float64)
    xs_tab = (pix[None, :] - F["scaler_mean"][:, None]) / F["scaler_scale"][:, None]
    P = {
        "lim": lim, "sx": sx, "sw0": sw0, "sh": sh, "sw1": sw1,
        "lut": quantizar(xs_tab, sx, lim),                    # (784, 256)
        "W0": quantizar(F["W0"], sw0, lim),                   # (784, 64)
        "W1": quantizar(F["W1"], sw1, lim),                   # (64, 10)
        "b0": arredondar_meio_longe_zero(F["b0"] / (sx * sw0)).astype(np.int64),
        "b1": arredondar_meio_longe_zero(F["b1"] / (sh * sw1)).astype(np.int64),
    }
    for nome in ("b0", "b1"):
        maior = int(np.max(np.abs(P[nome])))
        verificar(f"{rotulo}: bias {nome} cabe em int32", maior <= INT32_MAX, f"max|{nome}| = {maior}")

    a_max = ENTRADAS * lim * lim + int(np.max(np.abs(P["b0"])))
    P["a_max"] = a_max
    P["mult"] = sx * sw0 / sh
    P["M"], P["N"] = escolher_multiplicador(P["mult"], a_max)
    P["z_max"] = OCULTA * lim * lim + int(np.max(np.abs(P["b1"])))
    return P


def forward_inteiro(Xu, P):
    """Especificacao exata do caminho inteiro (o C deve reproduzir bit a bit)."""
    xq = P["lut"][np.arange(ENTRADAS)[None, :], Xu]           # (N, 784)
    acc1 = produto_inteiro_exato(xq, P["W0"]) + P["b0"]
    h = requantizar(acc1, P["M"], P["N"], P["lim"])
    z = produto_inteiro_exato(h, P["W1"]) + P["b1"]
    return xq, acc1, h, z


def forward_int8_antigo(Xu, Q):
    """Reproduz forward_int8 do quantiza.py (np.round, reconstrucao em float64)."""
    Xs = (Xu.astype(np.float64) - Q["scaler_mean"]) / Q["scaler_scale"]
    xq = np.clip(np.round(Xs / Q["sx"]), -127, 127)
    b0i = np.round(Q["b0"] / (Q["sx"] * Q["sW0"]))
    acc1 = xq @ Q["W0q"].astype(np.float64) + b0i
    h = np.maximum(0, acc1 * (Q["sx"] * Q["sW0"]))
    hq = np.clip(np.round(h / Q["sh"]), -127, 127)
    b1i = np.round(Q["b1"] / (Q["sh"] * Q["sW1"]))
    return (hq @ Q["W1q"].astype(np.float64) + b1i) * (Q["sh"] * Q["sW1"])


def ablacao(Xu, y, F, P8, pred_f64, pred8):
    """Decompoe a diferenca entre o modelo original e o int8 em etapas isoladas.

    Cada etapa acrescenta UMA transformacao a anterior e e comparada a ela
    por McNemar. Pesos, bias e entradas quantizados sao reconstruidos em
    float (T_q * s) para medir o efeito de cada quantizacao separadamente.
    """
    sx, sw0, sh, sw1 = P8["sx"], P8["sw0"], P8["sh"], P8["sw1"]
    lim_x = sx * LIM8
    Xs = (Xu.astype(np.float64) - F["scaler_mean"]) / F["scaler_scale"]
    W0d, W1d = P8["W0"] * sw0, P8["W1"] * sw1
    b0d, b1d = P8["b0"] * (sx * sw0), P8["b1"] * (sh * sw1)
    xq_d = P8["lut"][np.arange(ENTRADAS)[None, :], Xu] * sx

    def prever(x, W0, b0, W1, b1):
        return np.argmax(np.maximum(x @ W0 + b0, 0.0) @ W1 + b1, axis=1)

    etapas = [
        ("0 modelo Python original", pred_f64),
        ("1 + recorte da entrada", prever(np.clip(Xs, -lim_x, lim_x), F["W0"], F["b0"], F["W1"], F["b1"])),
        ("2 + pesos e bias int8", prever(np.clip(Xs, -lim_x, lim_x), W0d, b0d, W1d, b1d)),
        ("3 + entrada int8 (tabela)", prever(xq_d, W0d, b0d, W1d, b1d)),
        ("4 + ativacao oculta int8 = int8", pred8),
    ]
    log("")
    log("  Ablacao do int8 (cada etapa x etapa anterior; McNemar exato)")
    log(f"  {'etapa':<34} {'acuracia':>8} {'discord.':>8} {'corrige':>7} {'erra novo':>9} {'p':>7}")
    anterior = None
    for nome, pred in etapas:
        acc = float(np.mean(pred == y))
        if anterior is None:
            log(f"  {nome:<34} {acc:8.4f} {'-':>8} {'-':>7} {'-':>9} {'-':>7}")
        else:
            d, corrige, erra, _, p = pareado(anterior, pred, y)
            log(f"  {nome:<34} {acc:8.4f} {d:8d} {corrige:7d} {erra:9d} {p:7.4f}")
        anterior = pred
    pred_antigo = prever(Xs, W0d, b0d, W1d, b1d)
    d, corrige, erra, _, p = pareado(pred_f64, pred_antigo, y)
    log(f"  {'A pesos int8, sem recorte':<34} {float(np.mean(pred_antigo == y)):8.4f} "
        f"{d:8d} {corrige:7d} {erra:9d} {p:7.4f}")
    log("  A = prever_float do testar_acuracia.c antigo; comparado com a etapa 0.")


# ----------------------------------------------------------------------------
# Escrita dos arquivos C
# ----------------------------------------------------------------------------
def _fmt_float32(v):
    """Literal C float32 com 10 algarismos significativos (ida e volta exata)."""
    return f"{float(v):.9e}f"


def _bloco_comentario(linhas):
    return "/*\n" + "".join(f" * {l}".rstrip() + "\n" for l in linhas) + " */\n"


def _array_c(tipo, nome, valores, fmt, por_linha, comentario):
    """Declara um array C static const; comentario pode ter varias linhas."""
    valores = list(valores)
    linhas = []
    for i in range(0, len(valores), por_linha):
        linhas.append("    " + ", ".join(fmt(v) for v in valores[i:i + por_linha]))
    return (_bloco_comentario(comentario)
            + f"static const {tipo} {nome}[{len(valores)}] = {{\n"
            + ",\n".join(linhas) + "\n};\n\n")


def _cabecalho(guarda, descricao):
    return (_bloco_comentario(["Gerado automaticamente por python/exporta_modelo.py. Nao editar a mao.",
                               "Para regenerar: python3 python/exporta_modelo.py", ""] + descricao)
            + f"#ifndef {guarda}\n#define {guarda}\n\n#include <stdint.h>\n\n")


def _faixa(v):
    return f"faixa dos valores: [{np.min(v):.6g}, {np.max(v):.6g}]"


def escrever_header_float(caminho, P):
    txt = _cabecalho("MLP_PESOS_FLOAT_H", [
        "PESOS FLOAT32 ORIGINAIS (modulo mlp_float.c)",
        "",
        "Origem: treino.py treina Pipeline(StandardScaler, MLPClassifier(64, relu))",
        "e salva em modelo/pesos_float.npz. Aqui os valores sao convertidos de",
        "float64 para float32, sem nenhuma outra alteracao numerica.",
        "",
        "  nome C        chave no .npz   atributo scikit-learn     forma no C",
        "  MLPF_MEDIA    scaler_mean     scaler.mean_              [784]",
        "  MLPF_ESCALA   scaler_scale    scaler.scale_             [784]",
        "  MLPF_W0       W0              mlp.coefs_[0] (transp.)   [64*784]",
        "  MLPF_B0       b0              mlp.intercepts_[0]        [64]",
        "  MLPF_W1       W1              mlp.coefs_[1] (transp.)   [10*64]",
        "  MLPF_B1       b1              mlp.intercepts_[1]        [10]",
        "",
        "Uso no C:",
        "  x[i] = (pixel[i] - MLPF_MEDIA[i]) / MLPF_ESCALA[i]",
        "  a[j] = MLPF_B0[j] + soma_i x[i] * MLPF_W0[j*784 + i]",
        "  h[j] = max(a[j], 0)",
        "  z[k] = MLPF_B1[k] + soma_j h[j] * MLPF_W1[k*64 + j]",
    ])
    txt += _array_c("float", "MLPF_MEDIA", P["media"], _fmt_float32, 8, [
        "MLPF_MEDIA[i]: media do pixel i nas 60.000 imagens de treino (escala 0 a 255).",
        _faixa(P["media"])])
    txt += _array_c("float", "MLPF_ESCALA", P["escala"], _fmt_float32, 8, [
        "MLPF_ESCALA[i]: desvio-padrao do pixel i no treino. Pixels sempre apagados",
        "(variancia zero) recebem 1, convencao do scikit-learn que evita divisao por zero.",
        _faixa(P["escala"])])
    txt += _array_c("float", "MLPF_W0", P["W0"].T.ravel(), _fmt_float32, 8, [
        "MLPF_W0[j*784 + i]: peso que liga a entrada i (pixel padronizado) ao neuronio oculto j.",
        "Linha j = os 784 pesos do neuronio j, contiguos na memoria.",
        _faixa(P["W0"])])
    txt += _array_c("float", "MLPF_B0", P["b0"], _fmt_float32, 8, [
        "MLPF_B0[j]: bias do neuronio oculto j.", _faixa(P["b0"])])
    txt += _array_c("float", "MLPF_W1", P["W1"].T.ravel(), _fmt_float32, 8, [
        "MLPF_W1[k*64 + j]: peso que liga o neuronio oculto j a classe k (digito k).",
        _faixa(P["W1"])])
    txt += _array_c("float", "MLPF_B1", P["b1"], _fmt_float32, 8, [
        "MLPF_B1[k]: bias da classe k.", _faixa(P["b1"])])
    txt += "#endif /* MLP_PESOS_FLOAT_H */\n"
    caminho.write_text(txt, encoding="utf-8", newline="\n")


def escrever_header_inteiro(caminho, P, F, prefixo, tipo_w, bits):
    guarda = f"MLP_PESOS_INT{bits}_H"
    lim = P["lim"]
    i_ex = PIXEL_EXEMPLO
    mu_ex, sd_ex = float(F["scaler_mean"][i_ex]), float(F["scaler_scale"][i_ex])
    xs_ex = (255.0 - mu_ex) / sd_ex
    a_ex = 10000
    h_ex = (a_ex * P["M"] + (1 << (P["N"] - 1))) >> P["N"]
    origem_escala_x = ("s_x = percentil 99,9 de |x| no treino / 127 (quantiza.py)" if bits == 8 else
                       "s_x = s_x(int8) * 127 / 32767 (mesmo limiar de recorte do int8)")
    origem_escala_h = ("s_h = percentil 99,9 de |h| no treino / 127 (quantiza.py)" if bits == 8 else
                       "s_h = s_h(int8) * 127 / 32767 (mesmo limiar de recorte do int8)")
    txt = _cabecalho(guarda, [
        f"PESOS DO CAMINHO INT{bits} (modulo mlp_int{bits}.c): nenhum ponto flutuante em execucao.",
        "",
        "Cada valor real T e guardado como inteiro T_q com uma escala s:  T ~= T_q * s",
        f"  T_q = clip(round(T / s), -{lim}, {lim}),  arredondamento metade para longe de zero",
        "",
        "ESCALAS (numeros reais usados SO no Python para gerar os inteiros abaixo):",
        f"  s_W0 = max|W0| / {lim}                         = {P['sw0']:.17g}",
        f"  s_W1 = max|W1| / {lim}                         = {P['sw1']:.17g}",
        f"  {origem_escala_x}",
        f"       = {P['sx']:.17g}",
        f"  {origem_escala_h}",
        f"       = {P['sh']:.17g}",
        "",
        "DE ONDE VEM CADA ARRAY:",
        f"  {prefixo}_LUT[i*256 + p] = clip(round(((p - media[i]) / desvio[i]) / s_x))",
        "      p = valor do pixel (0 a 255); media e desvio vem do StandardScaler.",
        "      Como p so tem 256 valores, a tabela e a conta exata pre-calculada.",
        f"  {prefixo}_W0[j*784 + i]  = clip(round(W0[i][j] / s_W0))   W0 = mlp.coefs_[0]",
        f"  {prefixo}_B0[j]          = round(b0[j] / (s_x * s_W0))     b0 = mlp.intercepts_[0]",
        f"  {prefixo}_W1[k*64 + j]   = clip(round(W1[j][k] / s_W1))   W1 = mlp.coefs_[1]",
        f"  {prefixo}_B1[k]          = round(b1[k] / (s_h * s_W1))     b1 = mlp.intercepts_[1]",
        f"  {prefixo}_M, {prefixo}_N   M / 2^N ~= s_x * s_W0 / s_h (fator de requantizacao)",
        "",
        "POR QUE O BIAS USA A ESCALA s_x * s_W0:",
        "  a = soma(W0 * x) + b0 = soma(W0_q*s_W0 * x_q*s_x) + b0 = (s_x*s_W0) * (soma(W0_q*x_q) + b0/(s_x*s_W0))",
        "  Assim o bias inteiro e somado na mesma unidade do acumulador inteiro.",
        "",
        "EXEMPLO NUMERICO (pixel central i = 406, aceso com p = 255):",
        f"  media[406] = {mu_ex:.6g}, desvio[406] = {sd_ex:.6g}",
        f"  x = (255 - {mu_ex:.6g}) / {sd_ex:.6g} = {xs_ex:.6g} desvios-padrao",
        f"  x / s_x = {xs_ex / P['sx']:.6g}  ->  {prefixo}_LUT[406*256 + 255] = {int(P['lut'][i_ex, 255])}",
        "",
        "EXEMPLO NUMERICO DA REQUANTIZACAO (acumulador A = 10000):",
        f"  valor real de h = A * s_x*s_W0 = {a_ex * P['sx'] * P['sw0']:.6g};  h / s_h = {a_ex * P['mult']:.6g}",
        f"  inteiro: (10000 * {P['M']} + 2^{P['N'] - 1}) >> {P['N']} = {h_ex}",
    ])
    txt += _bloco_comentario([
        f"{prefixo}_LIM: limite simetrico do tipo int{bits}.",
        f"{prefixo}_M e {prefixo}_N: fator real {P['mult']:.17g} aproximado por",
        f"  M / 2^N = {P['M'] / 2**P['N']:.17g} (erro relativo {abs(P['M'] / 2**P['N'] - P['mult']) / P['mult']:.2e}).",
        "  N e o maior deslocamento com M < 2^31 e sem estouro de int64 no pior caso."])
    txt += f"#define {prefixo}_LIM {lim}\n"
    txt += f"#define {prefixo}_M INT64_C({P['M']})\n"
    txt += f"#define {prefixo}_N {P['N']}\n\n"
    txt += _array_c(tipo_w, f"{prefixo}_LUT", P["lut"].ravel(), str, 16, [
        f"{prefixo}_LUT[i*256 + p]: entrada i ja padronizada e quantizada para o pixel de valor p.",
        f"Valores em [-{lim}, {lim}]; os extremos correspondem a entradas recortadas."])
    txt += _array_c(tipo_w, f"{prefixo}_W0", P["W0"].T.ravel(), str, 16, [
        f"{prefixo}_W0[j*784 + i]: peso quantizado da entrada i para o neuronio oculto j.",
        f"Valor real ~= {prefixo}_W0 * s_W0. Linha j contigua (784 valores).",
        _faixa(P["W0"])])
    txt += _array_c("int32_t", f"{prefixo}_B0", P["b0"], str, 16, [
        f"{prefixo}_B0[j]: bias do neuronio j na unidade do acumulador (s_x * s_W0).",
        _faixa(P["b0"])])
    txt += _array_c(tipo_w, f"{prefixo}_W1", P["W1"].T.ravel(), str, 16, [
        f"{prefixo}_W1[k*64 + j]: peso quantizado do neuronio oculto j para a classe k.",
        f"Valor real ~= {prefixo}_W1 * s_W1.",
        _faixa(P["W1"])])
    txt += _array_c("int32_t", f"{prefixo}_B1", P["b1"], str, 16, [
        f"{prefixo}_B1[k]: bias da classe k na unidade do logit inteiro (s_h * s_W1).",
        _faixa(P["b1"])])
    txt += f"#endif /* {guarda} */\n"
    caminho.write_text(txt, encoding="utf-8", newline="\n")


# ----------------------------------------------------------------------------
# Principal
# ----------------------------------------------------------------------------
def main():
    DIR_SAIDA.mkdir(parents=True, exist_ok=True)
    F, Q, X, y = carregar()
    Xu, y = validar_artefatos(F, Q, X, y)

    # --- 2. float64 (modelo Python original) e float32 (espelho exato do C) ---
    log("")
    log("=" * 70)
    log("2. CAMINHO FLOAT")
    log("=" * 70)
    h_f64, z_f64 = forward_float64(Xu, F)
    pred_f64 = np.argmax(z_f64, axis=1)
    acertos_f64 = int(np.sum(pred_f64 == y))
    esperado = int(round(F["acc_baseline"] * N_TESTE))
    verificar("float64 reproduz acc_baseline do scikit-learn", acertos_f64 == esperado,
              f"{acertos_f64}/{N_TESTE} (esperado {esperado})")

    PF = {"media": F["scaler_mean"].astype(np.float32), "escala": F["scaler_scale"].astype(np.float32),
          "W0": F["W0"].astype(np.float32), "b0": F["b0"].astype(np.float32),
          "W1": F["W1"].astype(np.float32), "b1": F["b1"].astype(np.float32)}
    for chave in PF:
        v = PF[chave].ravel()
        volta = np.array([np.float32(float(_fmt_float32(x)[:-1])) for x in v], dtype=np.float32)
        verificar(f"Literal C float32 de {chave} ida e volta exato", bool(np.array_equal(volta, v)))
    z_f32 = simular_float32_c(Xu, PF)
    pred_f32 = np.argmax(z_f32, axis=1)
    div_32_64 = int(np.sum(pred_f32 != pred_f64))
    verificar("float32 decide igual ao float64", div_32_64 == 0, f"{div_32_64} decisoes diferentes", fatal=False)

    # --- 3. int8 ---------------------------------------------------------------
    log("")
    log("=" * 70)
    log("3. CAMINHO INT8")
    log("=" * 70)
    P8 = construir_inteiro(F, Q["sx"], Q["sW0"], Q["sh"], Q["sW1"], LIM8, "int8")
    verificar("int8: W0 reconstruido = W0q do .npz",
              bool(np.array_equal(P8["W0"], Q["W0q"].astype(np.int64))), fatal=False)
    verificar("int8: W1 reconstruido = W1q do .npz",
              bool(np.array_equal(P8["W1"], Q["W1q"].astype(np.int64))), fatal=False)
    xq8, a8, h8, z8 = forward_inteiro(Xu, P8)
    pred8 = np.argmax(z8, axis=1)
    pred8_antigo = np.argmax(forward_int8_antigo(Xu, Q), axis=1)
    div_antigo = int(np.sum(pred8 != pred8_antigo))
    verificar("int8 novo decide igual ao int8 do quantiza.py", div_antigo == 0,
              f"{div_antigo} decisoes diferentes", fatal=False)
    esperado8 = int(round(Q["acc_int8"] * N_TESTE))
    verificar("int8 novo reproduz acc_int8 do .npz", int(np.sum(pred8 == y)) == esperado8,
              f"{int(np.sum(pred8 == y))}/{N_TESTE} (esperado {esperado8})", fatal=False)

    # --- 4. int16 --------------------------------------------------------------
    log("")
    log("=" * 70)
    log("4. CAMINHO INT16")
    log("=" * 70)
    sx16 = Q["sx"] * LIM8 / LIM16
    sh16 = Q["sh"] * LIM8 / LIM16
    sw0_16 = float(np.max(np.abs(F["W0"]))) / LIM16
    sw1_16 = float(np.max(np.abs(F["W1"]))) / LIM16
    P16 = construir_inteiro(F, sx16, sw0_16, sh16, sw1_16, LIM16, "int16")
    xq16, a16, h16, z16 = forward_inteiro(Xu, P16)
    pred16 = np.argmax(z16, axis=1)

    # --- 5. limites numericos --------------------------------------------------
    log("")
    log("=" * 70)
    log("5. LIMITES NUMERICOS (teorico = pior caso possivel)")
    log("=" * 70)
    for rot, P, a, z, tipo_acc, lim_acc in (("int8", P8, a8, z8, "int32", INT32_MAX),
                                            ("int16", P16, a16, z16, "int64", INT64_MAX)):
        verificar(f"{rot}: acumulador oculto teorico cabe em {tipo_acc}", P["a_max"] <= lim_acc,
                  f"teorico {P['a_max']:.3e}, observado [{int(a.min())}, {int(a.max())}]")
        verificar(f"{rot}: produto a*M + 2^(N-1) cabe em int64",
                  P["a_max"] * P["M"] + (1 << (P["N"] - 1)) <= INT64_MAX,
                  f"M={P['M']}, N={P['N']}, teorico {P['a_max'] * P['M']:.3e}")
        verificar(f"{rot}: logit teorico cabe em {tipo_acc}", P["z_max"] <= lim_acc,
                  f"teorico {P['z_max']:.3e}, observado max|z| = {int(np.abs(z).max())}")
        erro_mult = abs(P["M"] / 2**P["N"] - P["mult"]) / P["mult"]
        log(f"         {rot}: erro relativo de M/2^N = {erro_mult:.2e}")
    verificar("int16: produto x*w teorico cabe em int32", LIM16 * LIM16 <= INT32_MAX,
              f"{LIM16 * LIM16} <= {INT32_MAX}")
    log(f"         int32 transbordaria no int16? observado max|a| = {int(np.abs(a16).max())} "
        f"({'SIM' if np.abs(a16).max() > INT32_MAX else 'nao'})")

    # --- 6. comparacao e analise -----------------------------------------------
    log("")
    log("=" * 70)
    log("6. COMPARACAO ENTRE CAMINHOS")
    log("=" * 70)
    acertos = {"float64": acertos_f64, "float32": int(np.sum(pred_f32 == y)),
               "int16": int(np.sum(pred16 == y)), "int8": int(np.sum(pred8 == y))}
    for k, v in acertos.items():
        log(f"  Acuracia {k:<8}: {v / N_TESTE:.4f}  ({v}/{N_TESTE})")
    div16 = int(np.sum(pred16 != pred_f64))
    div8 = int(np.sum(pred8 != pred_f64))
    log(f"  Decisoes diferentes do float64 -> int16: {div16}   int8: {div8}")
    log(f"  Decisoes diferentes int16 x int8: {int(np.sum(pred16 != pred8))}")

    rmse = lambda h, s, ref: float(np.sqrt(np.mean((h * s - ref) ** 2)))
    tot8, tot16 = rmse(h8, P8["sh"], h_f64), rmse(h16, P16["sh"], h_f64)
    log(f"  RMSE de h vs float64 (inclui recorte)   -> int16: {tot16:.6f}   int8: {tot8:.6f}")
    h_rec = ativacao_oculta_recortada(Xu, F, Q["sx"] * LIM8, Q["sh"] * LIM8)
    rmse8, rmse16 = rmse(h8, P8["sh"], h_rec), rmse(h16, P16["sh"], h_rec)
    razao = rmse8 / rmse16 if rmse16 > 0 else float("inf")
    log(f"  RMSE de h vs float recortado (resolucao) -> int16: {rmse16:.6f}   int8: {rmse8:.6f}"
        f"   (razao int8/int16 = {razao:.1f}; ideal ~{LIM16 / LIM8:.0f})")
    log(f"  Entradas saturadas -> int16: {np.mean(np.abs(xq16) == LIM16):.4%}"
        f"   int8: {np.mean(np.abs(xq8) == LIM8):.4%}")
    log(f"  Ativacoes ocultas saturadas -> int16: {np.mean(h16 == LIM16):.4%}"
        f"   int8: {np.mean(h8 == LIM8):.4%}")
    n_dist16, n_dist8 = int(np.unique(h16).size), int(np.unique(h8).size)
    log(f"  Valores distintos de h -> int16: {n_dist16}   int8: {n_dist8}")

    log("")
    log("  Comparacoes pareadas (McNemar exato; A = primeiro, B = segundo)")
    comparar_pareado("Python original x int16", pred_f64, pred16, y)
    comparar_pareado("Python original x int8", pred_f64, pred8, y)
    comparar_pareado("int16 x int8", pred16, pred8, y)

    log("")
    log("  Efeito isolado do recorte (float64, sem nenhuma quantizacao)")
    lim_x, lim_h = Q["sx"] * LIM8, Q["sh"] * LIM8
    Xs = (Xu.astype(np.float64) - F["scaler_mean"]) / F["scaler_scale"]
    log(f"  Limiar da entrada: {lim_x:.4f} desvios-padrao | maior entrada padronizada no teste: "
        f"{float(np.max(np.abs(Xs))):.1f}")
    log(f"  Imagens com ao menos uma entrada recortada: {int(np.sum(np.any(np.abs(Xs) > lim_x, axis=1)))}"
        f"/{N_TESTE}")
    Xc = np.clip(Xs, -lim_x, lim_x)
    hc = np.maximum(Xc @ F["W0"] + F["b0"], 0.0)
    pred_rec_x = np.argmax(hc @ F["W1"] + F["b1"], axis=1)
    pred_rec_xh = np.argmax(np.minimum(hc, lim_h) @ F["W1"] + F["b1"], axis=1)
    log(f"  Acuracia float64 com recorte da entrada           : {np.mean(pred_rec_x == y):.4f}")
    log(f"  Acuracia float64 com recorte da entrada e oculta  : {np.mean(pred_rec_xh == y):.4f}")
    comparar_pareado("Python original x float64 recortado", pred_f64, pred_rec_xh, y)
    comparar_pareado("float64 recortado x int16", pred_rec_xh, pred16, y)
    comparar_pareado("float64 recortado x int8", pred_rec_xh, pred8, y)

    ablacao(Xu, y, F, P8, pred_f64, pred8)

    log("")
    verificar("Alarme A: int16 diverge do float64 menos que o int8", div16 < div8 or div8 == 0,
              f"int16={div16}, int8={div8}", fatal=False)
    verificar("Alarme B: int16 usa mais de 128 niveis de ativacao", n_dist16 > 128,
              f"{n_dist16} niveis distintos")
    verificar("Alarme B: erro de resolucao do int16 ao menos 10x menor que o do int8", razao >= 10.0,
              f"razao = {razao:.1f}")

    # --- 7. escrita --------------------------------------------------------------
    log("")
    log("=" * 70)
    log("7. ARQUIVOS GERADOS")
    log("=" * 70)
    escrever_header_float(DIR_SAIDA / "mlp_pesos_float.h", PF)
    escrever_header_inteiro(DIR_SAIDA / "mlp_pesos_int16.h", P16, F, "MLP16", "int16_t", 16)
    escrever_header_inteiro(DIR_SAIDA / "mlp_pesos_int8.h", P8, F, "MLP8", "int8_t", 8)

    registros = np.empty((N_TESTE, 1 + ENTRADAS), dtype=np.uint8)
    registros[:, 0] = y
    registros[:, 1:] = Xu
    registros.tofile(DIR_SAIDA / "teste_u8.bin")
    z_f32.astype("<f4").tofile(DIR_SAIDA / "ref_float32.bin")
    z16.astype("<i8").tofile(DIR_SAIDA / "ref_int16.bin")
    z8.astype("<i4").tofile(DIR_SAIDA / "ref_int8.bin")
    pred_f64.astype(np.uint8).tofile(DIR_SAIDA / "ref_python_pred.bin")

    for nome in ("mlp_pesos_float.h", "mlp_pesos_int16.h", "mlp_pesos_int8.h",
                 "teste_u8.bin", "ref_float32.bin", "ref_int16.bin", "ref_int8.bin",
                 "ref_python_pred.bin"):
        log(f"  {nome:<22} {(DIR_SAIDA / nome).stat().st_size:>10} bytes")

    log("")
    log("=" * 70)
    if _falhas:
        log(f"RESULTADO: FALHA em {len(_falhas)} verificacao(oes). Nao prosseguir.")
        for n in _falhas:
            log(f"  - {n}")
    else:
        log("RESULTADO: todas as verificacoes fatais passaram.")
    if _alertas:
        log(f"ALERTAS ({len(_alertas)}):")
        for n in _alertas:
            log(f"  - {n}")
    log("=" * 70)

    (DIR_SAIDA / "relatorio_exportacao.txt").write_text("\n".join(_relatorio) + "\n",
                                                        encoding="utf-8", newline="\n")
    return 1 if _falhas else 0


if __name__ == "__main__":
    sys.exit(main())
