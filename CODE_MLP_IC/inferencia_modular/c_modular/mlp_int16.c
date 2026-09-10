/*
 * mlp_int16.c - Inferencia quantizada em int16, sem ponto flutuante.
 *
 * Papel na IC: controle de precisao. Usa exatamente o metodo do int8
 * (mesmas escalas relativas e o mesmo limiar de recorte), trocando 127 por
 * 32767. Se o int8 estiver correto, o int16 deve ficar mais proximo do
 * float; qualquer diferenca entre os dois vem so da largura de bits.
 *
 * Tipos de cada etapa e por que cabem:
 *   x, pesos, h        int16  em [-32767, 32767]
 *   x[i] * w[i]        int32  |32767 * 32767| = 1.073.676.289 <= 2.147.483.647
 *   acumulador A       int64  pior caso 784 * 1,07e9 ~ 8,4e11 (NAO cabe em int32;
 *                             nos dados de teste chegou a 7,9e9)
 *   A * M              int64  pior caso ~ 7,4e18 <= 9,2e18 (verificado no Python)
 *   logit Z            int64  pior caso 64 * 1,07e9 ~ 6,9e10
 *
 * Origem das constantes (todas em mlp_pesos_int16.h; formulas e exemplo
 * numerico no proprio header):
 *   MLP16_LUT[i*256 + p]  entrada i quantizada para pixel de valor p
 *   MLP16_W0, MLP16_W1    pesos quantizados com s_W = max|W| / 32767
 *   MLP16_B0, MLP16_B1    bias nas unidades s_x*s_W0 e s_h*s_W1
 *   MLP16_M, MLP16_N      M / 2^N ~= s_x * s_W0 / s_h
 *
 * Correspondencia NEON prevista: vmull_s16 (produto 16 -> 32 bits) seguido
 * de vpadalq_s32 (acumulacao em 64 bits).
 */
#include "mlp.h"
#include "mlp_pesos_int16.h"

_Static_assert(sizeof(MLP16_LUT) / sizeof(MLP16_LUT[0]) == MLP_ENTRADAS * 256, "MLP16_LUT com tamanho errado");
_Static_assert(sizeof(MLP16_W0) / sizeof(MLP16_W0[0]) == MLP_OCULTA * MLP_ENTRADAS, "MLP16_W0 com tamanho errado");
_Static_assert(sizeof(MLP16_B0) / sizeof(MLP16_B0[0]) == MLP_OCULTA, "MLP16_B0 com tamanho errado");
_Static_assert(sizeof(MLP16_W1) / sizeof(MLP16_W1[0]) == MLP_SAIDAS * MLP_OCULTA, "MLP16_W1 com tamanho errado");
_Static_assert(sizeof(MLP16_B1) / sizeof(MLP16_B1[0]) == MLP_SAIDAS, "MLP16_B1 com tamanho errado");
_Static_assert(MLP16_N >= 1 && MLP16_N <= 62, "MLP16_N fora do intervalo");
_Static_assert(MLP16_LIM == INT16_MAX, "MLP16_LIM deve ser 32767");

/*
 * Requantizacao: acumulador A (unidade s_x * s_W0) -> ativacao int16 (unidade s_h).
 *   h_q = round(max(A, 0) * s_x*s_W0/s_h) = (max(A, 0) * M + 2^(N-1)) >> N
 * Mesma logica do int8 (ver mlp_int8.c); saturacao em 32767.
 */
static inline int16_t requantizar(int64_t a)
{
    if (a < 0) {
        a = 0;                                   /* ReLU no dominio inteiro */
    }
    int64_t t = (a * MLP16_M + (INT64_C(1) << (MLP16_N - 1))) >> MLP16_N;
    return (int16_t)((t > MLP16_LIM) ? MLP16_LIM : t);
}

void mlp_int16_inferir(const uint8_t pixels[MLP_ENTRADAS], int64_t logits[MLP_SAIDAS])
{
    int16_t x[MLP_ENTRADAS];   /* entrada quantizada (unidade s_x) */
    int16_t h[MLP_OCULTA];     /* ativacao oculta quantizada (unidade s_h) */

    /* 1) Entrada: padronizacao + quantizacao + recorte pre-calculados na tabela. */
    for (int i = 0; i < MLP_ENTRADAS; i++) {
        x[i] = MLP16_LUT[i * 256 + pixels[i]];
    }

    /* 2) Camada oculta: A = B0[j] + produto escalar inteiro(W0[j], x),
     *    depois ReLU e requantizacao para int16. */
    for (int j = 0; j < MLP_OCULTA; j++) {
        const int16_t *w = &MLP16_W0[j * MLP_ENTRADAS];  /* linha j: os 784 pesos do neuronio j */
        int64_t a = MLP16_B0[j];
        for (int i = 0; i < MLP_ENTRADAS; i++) {
            a += (int32_t)x[i] * (int32_t)w[i];          /* produto cabe em 32 bits */
        }
        h[j] = requantizar(a);
    }

    /* 3) Camada de saida: Z = B1[k] + produto escalar inteiro(W1[k], h).
     *    Sem requantizacao: escala positiva e comum, argmax inalterado. */
    for (int k = 0; k < MLP_SAIDAS; k++) {
        const int16_t *w = &MLP16_W1[k * MLP_OCULTA];    /* linha k: os 64 pesos da classe k */
        int64_t z = MLP16_B1[k];
        for (int j = 0; j < MLP_OCULTA; j++) {
            z += (int32_t)h[j] * (int32_t)w[j];
        }
        logits[k] = z;
    }
}
