/*
 * dot_i8_neon.c - Kernel NEON de produto escalar int8 -> int32.
 *
 * ORIGEM: funcao neon_dot_i8 de experimentos_neon_2/homogeneidade_v2.c
 * (mesmo algoritmo de dot_product_int8_neon em ICV ARM NEON/dot_neon.c,
 * validado na Raspberry Pi 4). O corpo foi copiado sem alteracao; apenas o
 * nome foi padronizado (dot_i8_neon) e a funcao deixou de ser static.
 *
 * Funcionamento, 16 elementos por iteracao (registrador de 128 bits):
 *   vld1q_s8      carrega 16 bytes de a e 16 de b
 *   vget_low_s8 / vget_high_s8   separa as metades de 8 elementos
 *   vmull_s8      multiplica 8 pares int8 -> 8 produtos int16 (sem estouro:
 *                 |produto| <= 16.384 < 32.767)
 *   vpadalq_s16   soma os produtos em pares adjacentes, alarga para int32 e
 *                 acumula nos 4 acumuladores (instrucao sadalp)
 * Reducao horizontal ao final: vadd_s32 + vpadd_s32 + vget_lane_s32.
 *
 * Por que o resultado e exatamente igual ao escalar: tudo e aritmetica
 * inteira sem estouro, e a soma inteira nao depende da ordem.
 *
 * Instrucoes usadas (smull, sadalp, add, addp) sao do ARMv8.0-A: rodam no
 * Cortex-A72. Nao usa a extensao dotprod (vdotq_s32, ARMv8.2).
 */
#include <arm_neon.h>

#include "dot_i8_neon.h"

int32_t dot_i8_neon(const int8_t *a, const int8_t *b, int len)
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
