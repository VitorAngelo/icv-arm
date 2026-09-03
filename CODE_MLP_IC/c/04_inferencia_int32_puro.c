#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include "pesos.h"

#define ENTRADAS 784
#define OCULTA 64
#define SAIDAS 10

int32_t quantizar_int32(float valor, float escala) {
    float q = roundf(valor / escala);
    if (q > 127.0f) return 127;
    if (q < -127.0f) return -127;
    return (int32_t)q;
}

int main() {
    float imagem[ENTRADAS] = {0.0f};
    for(int i = 4; i < 24; i++) imagem[i * 28 + 14] = 255.0f;

    int32_t x_q[ENTRADAS];
    int32_t h_q[OCULTA];
    int32_t logits[SAIDAS];

    for (int i = 0; i < ENTRADAS; i++) {
        float xs = (scaler_scale[i] != 0.0f) ? (imagem[i] - scaler_mean[i]) / scaler_scale[i] : 0.0f;
        x_q[i] = quantizar_int32(xs, s_x);
    }

    for (int j = 0; j < OCULTA; j++) {
        int32_t soma = b0_int32[j];
        for (int i = 0; i < ENTRADAS; i++) {
            // ==============================================================================
            // CORRECAO PRODUTO LINEAR (Simulacao Estrita de Hardware)
            // Produto forcado em int16_t e acumulado em int32_t.
            // ==============================================================================
            soma += (int32_t)( (int16_t)x_q[i] * (int16_t)W0_q[i * OCULTA + j] );
        }
        float z1 = (float)soma * (s_x * s_W0);
        h_q[j] = quantizar_int32((z1 > 0.0f) ? z1 : 0.0f, s_h);
    }

    for (int j = 0; j < SAIDAS; j++) {
        logits[j] = b1_int32[j];
        for (int i = 0; i < OCULTA; i++) {
            // ==============================================================================
            // CORRECAO PRODUTO LINEAR (Camada de Saida)
            // ==============================================================================
            logits[j] += (int32_t)( (int16_t)h_q[i] * (int16_t)W1_q[i * SAIDAS + j] );
        }
    }

    int classe = 0;
    int32_t max_val = logits[0];
    for (int j = 1; j < SAIDAS; j++) {
        if (logits[j] > max_val) { max_val = logits[j]; classe = j; }
    }

    printf("Classe prevista: %d\n", classe);
    return 0;
}
