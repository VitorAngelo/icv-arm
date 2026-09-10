/*
 * mlp.h - Interface dos modulos de inferencia do MLP 784 -> 64 -> 10 (MNIST).
 *
 * ============================================================================
 * 1. O MODELO
 * ============================================================================
 * Rede densa treinada em Python (python/treino.py) com scikit-learn:
 *   Pipeline(StandardScaler, MLPClassifier(hidden_layer_sizes=(64,), activation="relu"))
 *
 *   entrada: imagem 28x28 achatada em 784 pixels (0 a 255)
 *   camada oculta: 64 neuronios com ReLU
 *   camada de saida: 10 neuronios, um por digito (0 a 9)
 *
 * Fluxo de dados (nomes usados em todos os modulos):
 *
 *   pixel[i] --padronizacao--> x[i] --W0,b0--> a[j] --ReLU--> h[j] --W1,b1--> z[k] --argmax--> classe
 *   i = 0..783                               j = 0..63                 k = 0..9
 *
 *   x[i] = (pixel[i] - media[i]) / desvio[i]    entrada padronizada
 *   a[j] = b0[j] + soma_i W0[j][i] * x[i]        pre-ativacao do neuronio oculto j
 *   h[j] = max(a[j], 0)                          ativacao ReLU
 *   z[k] = b1[k] + soma_j W1[k][j] * h[j]        logit da classe k
 *   classe = indice do maior z[k]
 *
 * Por que nao ha softmax: softmax(z)_k = e^z_k / soma e^z_j preserva a ordem
 * dos logits (a exponencial e crescente e o denominador e comum). Portanto
 * argmax(softmax(z)) = argmax(z). A softmax so e necessaria para obter
 * probabilidades ou treinar; para decidir a classe, o argmax dos logits basta.
 *
 * ============================================================================
 * 2. DE ONDE VEM CADA PESO
 * ============================================================================
 * O scikit-learn guarda, apos o treino:
 *   scaler.mean_         media de cada pixel no treino           [784]
 *   scaler.scale_        desvio-padrao de cada pixel (1 se zero) [784]
 *   mlp.coefs_[0]        W0, forma (784, 64): linha = entrada, coluna = neuronio
 *   mlp.intercepts_[0]   b0, forma (64,)
 *   mlp.coefs_[1]        W1, forma (64, 10)
 *   mlp.intercepts_[1]   b1, forma (10,)
 *
 * python/treino.py salva esses arrays em modelo/pesos_float.npz;
 * python/quantiza.py calcula as escalas int8 e salva modelo/pesos_int8.npz;
 * python/exporta_modelo.py gera os headers em modelo/exportado/:
 *
 *   header               nomes                                     usado por
 *   mlp_pesos_float.h    MLPF_MEDIA, MLPF_ESCALA, MLPF_W0, MLPF_B0,  mlp_float.c
 *                        MLPF_W1, MLPF_B1
 *   mlp_pesos_int16.h    MLP16_LUT, MLP16_W0, MLP16_B0, MLP16_W1,    mlp_int16.c
 *                        MLP16_B1, MLP16_M, MLP16_N, MLP16_LIM
 *   mlp_pesos_int8.h     MLP8_LUT, MLP8_W0, MLP8_B0, MLP8_W1,        mlp_int8.c
 *                        MLP8_B1, MLP8_M, MLP8_N, MLP8_LIM
 *
 * As matrizes sao transpostas e achatadas com uma linha por neuronio:
 *   W0[j*784 + i] = peso da entrada i para o neuronio oculto j
 *   W1[k*64  + j] = peso do neuronio oculto j para a classe k
 * Assim os pesos de cada neuronio ficam contiguos na memoria, o que o
 * produto escalar vetorizado (NEON) exige.
 * Cada header traz, em comentarios, a formula exata e um exemplo numerico.
 *
 * ============================================================================
 * 3. QUANTIZACAO (modulos int16 e int8)
 * ============================================================================
 * Um valor real T e representado por um inteiro T_q e uma escala real s > 0:
 *   T ~= T_q * s,    T_q = clip(round(T / s), -LIM, LIM)   (LIM = 127 ou 32767)
 * As escalas s_x, s_W0, s_h, s_W1 so existem no Python; o C recebe apenas
 * inteiros ja calculados:
 *
 *   LUT[i*256 + p]  entrada i ja padronizada e quantizada, para pixel de valor p.
 *                   O pixel so tem 256 valores, entao a tabela e a conta exata.
 *   W0, W1          pesos quantizados:  W0_q = round(W0 / s_W0)
 *   B0              bias na unidade do acumulador: round(b0 / (s_x * s_W0))
 *   B1              bias na unidade do logit:      round(b1 / (s_h * s_W1))
 *   M, N            inteiros tais que M / 2^N ~= s_x * s_W0 / s_h
 *
 * Por que funciona (camada oculta):
 *   a = soma W0*x + b0 = soma (W0_q*s_W0)(x_q*s_x) + b0 = (s_x*s_W0) * A,
 *   com A = soma W0_q*x_q + B0 (acumulador inteiro).
 *   ReLU: max(0, (s_x*s_W0)*A) = (s_x*s_W0)*max(0, A), pois a escala e positiva.
 *   Requantizacao: h_q = round(h / s_h) = round(max(A, 0) * s_x*s_W0/s_h)
 *                      = (max(A, 0) * M + 2^(N-1)) >> N,  limitado a LIM.
 *   O termo 2^(N-1) faz o deslocamento arredondar em vez de truncar.
 * Camada de saida:
 *   z = (s_h*s_W1) * Z, com Z = soma W1_q*h_q + B1. A escala e positiva e
 *   comum as 10 classes, logo argmax(z) = argmax(Z): nao ha requantizacao.
 *
 * ============================================================================
 * 4. CONTRATO DE VALIDACAO
 * ============================================================================
 * Todos os modulos recebem os 784 pixels brutos e escrevem os 10 logits.
 * Em empate vence o menor indice (comparacao estrita '>').
 * Para cada imagem de teste, os logits devem ser identicos, bit a bit, aos
 * gravados pelo python/exporta_modelo.py em ref_<modulo>.bin. Igualdade de
 * acuracia nao basta: um erro sistematico pode alterar todos os logits e
 * manter a acuracia (ver -ffp-contract=off no Makefile).
 */
#ifndef MLP_H
#define MLP_H

#include <stdint.h>

#define MLP_ENTRADAS 784   /* 28 x 28 pixels */
#define MLP_OCULTA   64    /* neuronios da camada oculta */
#define MLP_SAIDAS   10    /* digitos 0 a 9 */

/* Registro do arquivo de teste teste_u8.bin: 1 byte de rotulo + 784 bytes de pixels. */
#define MLP_BYTES_REGISTRO (1 + MLP_ENTRADAS)

/*
 * float32 (referencia em C): pesos float originais, sem quantizacao nem recorte.
 *   x[i] = (pixel[i] - MLPF_MEDIA[i]) / MLPF_ESCALA[i]
 *   a[j] = MLPF_B0[j] + soma sequencial em i de x[i] * MLPF_W0[j*784 + i]
 *   h[j] = a[j] > 0 ? a[j] : 0
 *   z[k] = MLPF_B1[k] + soma sequencial em j de h[j] * MLPF_W1[k*64 + j]
 * A ordem sequencial das somas faz parte da especificacao (em float,
 * (a + b) + c pode diferir de a + (b + c)); compilar sem -ffast-math e com
 * -ffp-contract=off.
 */
void mlp_float_inferir(const uint8_t pixels[MLP_ENTRADAS], float logits[MLP_SAIDAS]);

/*
 * int16: sem ponto flutuante. Produto 16 x 16 -> 32 bits, acumulacao em 64 bits.
 *   x[i] = MLP16_LUT[i*256 + pixel[i]]
 *   A[j] = MLP16_B0[j] + soma de x[i] * MLP16_W0[j*784 + i]             (int64)
 *   h[j] = min((max(A[j], 0) * MLP16_M + 2^(MLP16_N-1)) >> MLP16_N, 32767)
 *   Z[k] = MLP16_B1[k] + soma de h[j] * MLP16_W1[k*64 + j]              (int64)
 */
void mlp_int16_inferir(const uint8_t pixels[MLP_ENTRADAS], int64_t logits[MLP_SAIDAS]);

/*
 * int8: sem ponto flutuante. Produto 8 x 8 -> 16 bits, acumulacao em 32 bits.
 *   x[i] = MLP8_LUT[i*256 + pixel[i]]
 *   A[j] = MLP8_B0[j] + soma de x[i] * MLP8_W0[j*784 + i]               (int32)
 *   h[j] = min((max(A[j], 0) * MLP8_M + 2^(MLP8_N-1)) >> MLP8_N, 127)   (produto em int64)
 *   Z[k] = MLP8_B1[k] + soma de h[j] * MLP8_W1[k*64 + j]                (int32)
 */
void mlp_int8_inferir(const uint8_t pixels[MLP_ENTRADAS], int32_t logits[MLP_SAIDAS]);

#endif /* MLP_H */
