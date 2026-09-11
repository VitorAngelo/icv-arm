/*
 * mlp_int8.c - MLP int8 puro: nenhum float na inferencia.
 *
 * Correcoes em relacao a prever_int8_puro de testar_acuracia.c (FIXME/TODO do professor):
 *   1. "as imagens tem que ser quantizadas": a entrada quantizada vem pronta da
 *      tabela MLP8_LUT[i*256 + pixel], calculada em Python. Como o pixel so tem
 *      256 valores, a tabela e exata e substitui a padronizacao e o quantizar() em float.
 *   2. "nao precisa dos casts para int32": o produto int8 * int8 cabe em int16
 *      (127 * 127 = 16.129) e e somado ao acumulador int32.
 *   3. "nao pode voltar para float": a conversao float z1 = soma * (s_x * s_W0) e o
 *      quantizar(z1, s_h) foram trocados por inteiros:
 *         h = min((max(soma, 0) * MLP8_M + 2^(N-1)) >> MLP8_N, 127)
 *      onde MLP8_M / 2^MLP8_N = s_x * s_W0 / s_h (calculado em Python).
 *      A ReLU e o max(soma, 0) no inteiro; a saida h e int8.
 *   4. Camada de saida sem requantizacao: o argmax dos logits int32 e o mesmo dos logits reais.
 *
 * Bias: MLP8_B0 = round(b0 / (s_x * s_W0)), na mesma unidade da soma dos produtos.
 */
#include "mlp.h"
#include "mlp_pesos_int8.h"

void mlp_int8_inferir(const uint8_t pixels[MLP_ENTRADAS], int32_t logits[MLP_SAIDAS])
{
    int8_t x[MLP_ENTRADAS];   /* entrada quantizada */
    int8_t h[MLP_OCULTA];     /* ativacao oculta quantizada */

    for (int i = 0; i < MLP_ENTRADAS; i++) {
        x[i] = MLP8_LUT[i * 256 + pixels[i]];
    }

    for (int j = 0; j < MLP_OCULTA; j++) {
        const int8_t *w = &MLP8_W0[j * MLP_ENTRADAS];
        int32_t soma = MLP8_B0[j];
        for (int i = 0; i < MLP_ENTRADAS; i++) {
            soma += (int16_t)(x[i] * w[i]);                 /* int8 * int8 -> int16 */
        }
        if (soma < 0) {
            soma = 0;                                        /* ReLU */
        }
        int64_t t = ((int64_t)soma * MLP8_M + (INT64_C(1) << (MLP8_N - 1))) >> MLP8_N;
        h[j] = (int8_t)((t > MLP8_LIM) ? MLP8_LIM : t);     /* requantizacao, saida int8 */
    }

    for (int k = 0; k < MLP_SAIDAS; k++) {
        const int8_t *w = &MLP8_W1[k * MLP_OCULTA];
        int32_t z = MLP8_B1[k];
        for (int j = 0; j < MLP_OCULTA; j++) {
            z += (int16_t)(h[j] * w[j]);
        }
        logits[k] = z;
    }
}
