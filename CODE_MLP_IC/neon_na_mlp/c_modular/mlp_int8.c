/*
 * mlp_int8.c - Inferencia quantizada em int8, sem ponto flutuante.
 *
 * O que este modulo prova: que a inferencia inteira especificada no Python
 * (python/exporta_modelo.py) roda em C usando so inteiros, com logits
 * identicos, bit a bit, a ref_int8.bin.
 *
 * Tipos de cada etapa e por que cabem:
 *   x, pesos, h        int8   em [-127, 127]
 *   x[i] * w[i]        int16  |127 * 127| = 16.129 <= 32.767
 *   acumulador A       int32  pior caso 784 * 16.129 + |B0| ~ 1,27e7 <= 2,1e9
 *   A * M              int64  pior caso ~ 1,4e16 <= 9,2e18 (verificado no Python)
 *   logit Z            int32  pior caso 64 * 16.129 + |B1| ~ 1,0e6
 *
 * Origem das constantes (todas em mlp_pesos_int8.h; formulas e exemplo
 * numerico no proprio header):
 *   MLP8_LUT[i*256 + p]  entrada i quantizada para pixel de valor p
 *                        = clip(round(((p - media[i]) / desvio[i]) / s_x))
 *   MLP8_W0[j*784 + i]   round(W0[i][j] / s_W0)
 *   MLP8_B0[j]           round(b0[j] / (s_x * s_W0))
 *   MLP8_W1[k*64 + j]    round(W1[j][k] / s_W1)
 *   MLP8_B1[k]           round(b1[k] / (s_h * s_W1))
 *   MLP8_M, MLP8_N       M / 2^N ~= s_x * s_W0 / s_h
 *
 * Correspondencia NEON prevista: vmull_s8 (produto 8 -> 16 bits) seguido de
 * vpadalq_s16 (soma em pares acumulando em 32 bits, instrucao sadalp).
 */
#include "mlp.h"
#include "mlp_pesos_int8.h"

_Static_assert(sizeof(MLP8_LUT) / sizeof(MLP8_LUT[0]) == MLP_ENTRADAS * 256, "MLP8_LUT com tamanho errado");
_Static_assert(sizeof(MLP8_W0) / sizeof(MLP8_W0[0]) == MLP_OCULTA * MLP_ENTRADAS, "MLP8_W0 com tamanho errado");
_Static_assert(sizeof(MLP8_B0) / sizeof(MLP8_B0[0]) == MLP_OCULTA, "MLP8_B0 com tamanho errado");
_Static_assert(sizeof(MLP8_W1) / sizeof(MLP8_W1[0]) == MLP_SAIDAS * MLP_OCULTA, "MLP8_W1 com tamanho errado");
_Static_assert(sizeof(MLP8_B1) / sizeof(MLP8_B1[0]) == MLP_SAIDAS, "MLP8_B1 com tamanho errado");
_Static_assert(MLP8_N >= 1 && MLP8_N <= 62, "MLP8_N fora do intervalo");
_Static_assert(MLP8_LIM == INT8_MAX, "MLP8_LIM deve ser 127");

/*
 * Requantizacao: converte o acumulador A (unidade s_x * s_W0) em ativacao
 * int8 (unidade s_h).
 *   valor real:   h = max(A, 0) * s_x * s_W0
 *   quantizado:   h_q = round(h / s_h) = round(max(A, 0) * s_x*s_W0/s_h)
 *   em inteiros:  h_q = (max(A, 0) * M + 2^(N-1)) >> N,  pois M / 2^N ~= s_x*s_W0/s_h
 * A ReLU vem antes: com A >= 0, somar 2^(N-1) e deslocar equivale a
 * arredondar metade para cima, igual ao arredondamento usado no Python.
 * Valores acima de 127 saturam (recorte da ativacao).
 */
static inline int8_t requantizar(int32_t a)
{
    if (a < 0) {
        a = 0;                                   /* ReLU no dominio inteiro */
    }
    int64_t t = ((int64_t)a * MLP8_M + (INT64_C(1) << (MLP8_N - 1))) >> MLP8_N;
    return (int8_t)((t > MLP8_LIM) ? MLP8_LIM : t);
}

void mlp_int8_inferir(const uint8_t pixels[MLP_ENTRADAS], int32_t logits[MLP_SAIDAS])
{
    int8_t x[MLP_ENTRADAS];   /* entrada quantizada (unidade s_x) */
    int8_t h[MLP_OCULTA];     /* ativacao oculta quantizada (unidade s_h) */

    /* 1) Entrada: padronizacao + quantizacao + recorte, tudo pre-calculado.
     *    Linha i da tabela = pixel i; coluna = valor do pixel (0 a 255). */
    for (int i = 0; i < MLP_ENTRADAS; i++) {
        x[i] = MLP8_LUT[i * 256 + pixels[i]];
    }

    /* 2) Camada oculta: A = B0[j] + produto escalar inteiro(W0[j], x),
     *    depois ReLU e requantizacao para int8. */
    for (int j = 0; j < MLP_OCULTA; j++) {
        const int8_t *w = &MLP8_W0[j * MLP_ENTRADAS];  /* linha j: os 784 pesos do neuronio j */
        int32_t a = MLP8_B0[j];
        for (int i = 0; i < MLP_ENTRADAS; i++) {
            a += (int16_t)((int16_t)x[i] * (int16_t)w[i]);   /* produto cabe em 16 bits */
        }
        h[j] = requantizar(a);
    }

    /* 3) Camada de saida: Z = B1[k] + produto escalar inteiro(W1[k], h).
     *    Sem requantizacao: Z = z / (s_h * s_W1), escala positiva e comum as
     *    10 classes, entao o argmax de Z e o mesmo do logit real. */
    for (int k = 0; k < MLP_SAIDAS; k++) {
        const int8_t *w = &MLP8_W1[k * MLP_OCULTA];    /* linha k: os 64 pesos da classe k */
        int32_t z = MLP8_B1[k];
        for (int j = 0; j < MLP_OCULTA; j++) {
            z += (int16_t)((int16_t)h[j] * (int16_t)w[j]);
        }
        logits[k] = z;
    }
}
