/*
 * mlp_neon_int8.c - MLP int8 com NEON: a mesma conta de mlp_int8.c, com os produtos
 * escalares das duas camadas feitos pelo meu kernel NEON.
 *
 * Kernel: neon_dot_i8, copiado sem alteracao de experimentos_neon_2/homogeneidade_v2.c
 * (mesmo algoritmo de dot_product_int8_neon em ICV ARM NEON/dot_neon.c, validado na placa).
 *   vld1q_s8      carrega 16 bytes de cada vetor
 *   vmull_s8      multiplica 8 pares int8 -> 8 produtos int16 (int8*int8 nao cabe em int8)
 *   vpadalq_s16   soma os produtos em pares e acumula em int32 (instrucao sadalp)
 *   vadd/vpadd    soma os 4 acumuladores e devolve um int32
 * Processa 16 elementos por iteracao: o tamanho deve ser multiplo de 16.
 * Na MLP: 784 = 16 * 49 e 64 = 16 * 4.
 *
 * Unica diferenca para mlp_int8.c: os dois lacos de produto escalar
 *   soma = MLP8_B0[j] + soma_i x[i] * W0[j*784 + i]   ->   MLP8_B0[j] + neon_dot_i8(x, &W0[j*784], 784)
 *   z    = MLP8_B1[k] + soma_j h[j] * W1[k*64 + j]    ->   MLP8_B1[k] + neon_dot_i8(h, &W1[k*64], 64)
 * Entrada pela tabela, ReLU e requantizacao sao identicas. Nenhum float.
 */
#include <arm_neon.h>

#include "mlp.h"
#include "mlp_pesos_int8.h"

_Static_assert(MLP_ENTRADAS % 16 == 0 && MLP_OCULTA % 16 == 0, "tamanhos devem ser multiplos de 16");

/* Kernel NEON: copiado sem alteracao de homogeneidade_v2.c. */
static int32_t neon_dot_i8(const int8_t *a, const int8_t *b, int len)
{
    int32x4_t acc = vdupq_n_s32(0);
    for (int i = 0; i < len; i += 16) {
        int8x16_t va = vld1q_s8(a + i);
        int8x16_t vb = vld1q_s8(b + i);
        acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(va),  vget_low_s8(vb)));
        acc = vpadalq_s16(acc, vmull_s8(vget_high_s8(va), vget_high_s8(vb)));
    }
    int32x2_t s = vadd_s32(vget_low_s32(acc), vget_high_s32(acc));
    return vget_lane_s32(vpadd_s32(s, s), 0);
}

void mlp_neon_int8_inferir(const uint8_t pixels[MLP_ENTRADAS], int32_t logits[MLP_SAIDAS])
{
    int8_t x[MLP_ENTRADAS];   /* entrada quantizada */
    int8_t h[MLP_OCULTA];     /* ativacao oculta quantizada */

    for (int i = 0; i < MLP_ENTRADAS; i++) {
        x[i] = MLP8_LUT[i * 256 + pixels[i]];
    }

    for (int j = 0; j < MLP_OCULTA; j++) {
        int32_t soma = MLP8_B0[j] + neon_dot_i8(x, &MLP8_W0[j * MLP_ENTRADAS], MLP_ENTRADAS);
        if (soma < 0) {
            soma = 0;                                        /* ReLU */
        }
        int64_t t = ((int64_t)soma * MLP8_M + (INT64_C(1) << (MLP8_N - 1))) >> MLP8_N;
        h[j] = (int8_t)((t > MLP8_LIM) ? MLP8_LIM : t);     /* requantizacao, saida int8 */
    }

    for (int k = 0; k < MLP_SAIDAS; k++) {
        logits[k] = MLP8_B1[k] + neon_dot_i8(h, &MLP8_W1[k * MLP_OCULTA], MLP_OCULTA);
    }
}
