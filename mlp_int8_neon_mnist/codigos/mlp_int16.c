/*
 * mlp_int16.c - MLP int16 puro: nenhum float na inferencia.
 *
 * Mesmo metodo da mlp_int8.c, com valores em [-32767, 32767]:
 *   x = MLP16_LUT[i*256 + pixel]                                  entrada quantizada (int16)
 *   soma = MLP16_B0[j] + soma_i x[i] * W0[j*784 + i]              produto int16*int16 -> int32,
 *                                                                 acumulado em int64 (int32 estouraria)
 *   h = min((max(soma, 0) * MLP16_M + 2^(N-1)) >> MLP16_N, 32767) ReLU + requantizacao (int16)
 *   z = MLP16_B1[k] + soma_j h[j] * W1[k*64 + j]                  logits int64
 * Serve para comparar: mais bits, mesma forma de calcular.
 */
#include "mlp.h"
#include "mlp_pesos_int16.h"

void mlp_int16_inferir(const uint8_t pixels[MLP_ENTRADAS], int64_t logits[MLP_SAIDAS])
{
    int16_t x[MLP_ENTRADAS];
    int16_t h[MLP_OCULTA];

    for (int i = 0; i < MLP_ENTRADAS; i++) {
        x[i] = MLP16_LUT[i * 256 + pixels[i]];
    }

    for (int j = 0; j < MLP_OCULTA; j++) {
        const int16_t *w = &MLP16_W0[j * MLP_ENTRADAS];
        int64_t soma = MLP16_B0[j];
        for (int i = 0; i < MLP_ENTRADAS; i++) {
            soma += (int32_t)x[i] * (int32_t)w[i];          /* int16 * int16 -> int32 */
        }
        if (soma < 0) {
            soma = 0;                                        /* ReLU */
        }
        int64_t t = (soma * MLP16_M + (INT64_C(1) << (MLP16_N - 1))) >> MLP16_N;
        h[j] = (int16_t)((t > MLP16_LIM) ? MLP16_LIM : t);
    }

    for (int k = 0; k < MLP_SAIDAS; k++) {
        const int16_t *w = &MLP16_W1[k * MLP_OCULTA];
        int64_t z = MLP16_B1[k];
        for (int j = 0; j < MLP_OCULTA; j++) {
            z += (int32_t)h[j] * (int32_t)w[j];
        }
        logits[k] = z;
    }
}
