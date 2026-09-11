/*
 * mlp_float.c - MLP float puro.
 *
 * Correcao em relacao a testar_acuracia.c: la, prever_float usava os pesos int8
 * reconvertidos para float (W0_q * s_W0). Isso nao e o modelo float: e um
 * terceiro modelo. Aqui sao usados os pesos float32 originais do treino.
 *
 *   x[i] = (pixel[i] - MEDIA[i]) / ESCALA[i]          padronizacao (StandardScaler)
 *   a[j] = B0[j] + soma_i x[i] * W0[j*784 + i]         camada oculta
 *   h[j] = max(a[j], 0)                                ReLU
 *   z[k] = B1[k] + soma_j h[j] * W1[k*64 + j]          logits (sem softmax: nao muda o argmax)
 */
#include "mlp.h"
#include "mlp_pesos_float.h"

void mlp_float_inferir(const uint8_t pixels[MLP_ENTRADAS], float logits[MLP_SAIDAS])
{
    float x[MLP_ENTRADAS];
    float h[MLP_OCULTA];

    for (int i = 0; i < MLP_ENTRADAS; i++) {
        x[i] = ((float)pixels[i] - MLPF_MEDIA[i]) / MLPF_ESCALA[i];
    }

    for (int j = 0; j < MLP_OCULTA; j++) {
        const float *w = &MLPF_W0[j * MLP_ENTRADAS];
        float a = MLPF_B0[j];
        for (int i = 0; i < MLP_ENTRADAS; i++) {
            a += x[i] * w[i];
        }
        h[j] = (a > 0.0f) ? a : 0.0f;
    }

    for (int k = 0; k < MLP_SAIDAS; k++) {
        const float *w = &MLPF_W1[k * MLP_OCULTA];
        float z = MLPF_B1[k];
        for (int j = 0; j < MLP_OCULTA; j++) {
            z += h[j] * w[j];
        }
        logits[k] = z;
    }
}
