/*
 * mlp_int8_neon.c - Inferencia int8 da MLP com o kernel NEON dot_i8_neon.
 *
 * Pipeline identico ao de c_modular/mlp_int8.c; so muda quem calcula os
 * produtos escalares:
 *
 *   entrada   x[i]   = MLP8_LUT[i*256 + pixel[i]]
 *   784->64   A[j]   = MLP8_B0[j] + dot_i8_neon(x, &MLP8_W0[j*784], 784)
 *             h[j]   = min((max(A[j], 0) * M + 2^(N-1)) >> N, 127)
 *   64->10    Z[k]   = MLP8_B1[k] + dot_i8_neon(h, &MLP8_W1[k*64], 64)
 *
 * O bias e somado fora do kernel: soma inteira, o resultado nao depende da
 * ordem. As matrizes ja estao com uma linha contigua por neuronio, entao
 * &MLP8_W0[j*784] e o vetor de pesos do neuronio j.
 */
#include <string.h>

#include "mlp_int8_neon.h"
#include "dot_i8_neon.h"
#include "mlp_pesos_int8.h"

/* O kernel exige tamanhos multiplos de 16 (nao trata tail). */
_Static_assert(MLP_ENTRADAS % DOT_I8_NEON_BLOCO == 0, "784 deve ser multiplo de 16");
_Static_assert(MLP_OCULTA % DOT_I8_NEON_BLOCO == 0, "64 deve ser multiplo de 16");
_Static_assert(sizeof(MLP8_W0) / sizeof(MLP8_W0[0]) == MLP_OCULTA * MLP_ENTRADAS, "MLP8_W0 com tamanho errado");
_Static_assert(sizeof(MLP8_W1) / sizeof(MLP8_W1[0]) == MLP_SAIDAS * MLP_OCULTA, "MLP8_W1 com tamanho errado");
_Static_assert(sizeof(MLP8_LUT) / sizeof(MLP8_LUT[0]) == MLP_ENTRADAS * 256, "MLP8_LUT com tamanho errado");

/* Requantizacao identica a de c_modular/mlp_int8.c:
 * h = min((max(A, 0) * M + 2^(N-1)) >> N, 127). */
static inline int8_t requantizar(int32_t a)
{
    if (a < 0) {
        a = 0;
    }
    int64_t t = ((int64_t)a * MLP8_M + (INT64_C(1) << (MLP8_N - 1))) >> MLP8_N;
    return (int8_t)((t > MLP8_LIM) ? MLP8_LIM : t);
}

/* Nucleo comum as duas funcoes publicas. Com e == NULL (inferencia normal),
 * o compilador remove todo o registro de estagios. */
static inline __attribute__((always_inline))
void inferir(const uint8_t *pixels, int32_t *logits, Mlp8Estagios *e)
{
    int8_t x[MLP_ENTRADAS];
    int8_t h[MLP_OCULTA];

    /* Entrada: padronizacao + quantizacao pre-calculadas na tabela. */
    for (int i = 0; i < MLP_ENTRADAS; i++) {
        x[i] = MLP8_LUT[i * 256 + pixels[i]];
    }

    /* Camada 784 -> 64: produto escalar NEON por neuronio, bias, ReLU e requantizacao. */
    for (int j = 0; j < MLP_OCULTA; j++) {
        int32_t produto = dot_i8_neon(x, &MLP8_W0[j * MLP_ENTRADAS], MLP_ENTRADAS);
        int32_t a = MLP8_B0[j] + produto;
        h[j] = requantizar(a);
        if (e != NULL) {
            e->produto[j] = produto;
            e->a[j] = a;
            e->relu[j] = (a > 0) ? a : 0;
            e->h[j] = h[j];
        }
    }

    /* Camada 64 -> 10: produto escalar NEON por classe e bias; sem requantizacao. */
    for (int k = 0; k < MLP_SAIDAS; k++) {
        logits[k] = MLP8_B1[k] + dot_i8_neon(h, &MLP8_W1[k * MLP_OCULTA], MLP_OCULTA);
    }

    if (e != NULL) {
        memcpy(e->x, x, sizeof(x));
        memcpy(e->logits, logits, sizeof(e->logits));
    }
}

void mlp_int8_neon_inferir(const uint8_t pixels[MLP_ENTRADAS], int32_t logits[MLP_SAIDAS])
{
    inferir(pixels, logits, NULL);
}

void mlp_int8_neon_estagios(const uint8_t pixels[MLP_ENTRADAS], Mlp8Estagios *e)
{
    inferir(pixels, e->logits, e);
}
