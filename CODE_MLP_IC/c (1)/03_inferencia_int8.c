#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include "pesos.h"

#define ENTRADAS 784
#define OCULTA 64
#define SAIDAS 10

int8_t quantizar_int8(float valor, float escala) {
    float q = roundf(valor / escala);
    if (q > 127.0f) return 127;
    if (q < -127.0f) return -127;
    return (int8_t)q;
}

int main() {
    float imagem[ENTRADAS] = {0.0f};
    for(int i = 4; i < 24; i++) imagem[i * 28 + 14] = 255.0f;

    // Arrays ultra comprimidos em 8 bits (Preparando terreno para o Raspberry Pi)
    int8_t x_q[ENTRADAS];
    int8_t h_q[OCULTA];
    int32_t logits[SAIDAS];

    for (int i = 0; i < ENTRADAS; i++) {
        float xs = (scaler_scale[i] != 0.0f) ? (imagem[i] - scaler_mean[i]) / scaler_scale[i] : 0.0f;
        x_q[i] = quantizar_int8(xs, s_x);
    }

    for (int j = 0; j < OCULTA; j++) {
        int32_t soma = b0_int32[j];
        for (int i = 0; i < ENTRADAS; i++) {
            // O cast (int32_t) acontece só no pulso de clock da multiplicacao
            soma += (int32_t)x_q[i] * (int32_t)W0_q[i * OCULTA + j];
        }
        float z1 = (float)soma * (s_x * s_W0);
        h_q[j] = quantizar_int8((z1 > 0.0f) ? z1 : 0.0f, s_h);
    }

    for (int j = 0; j < SAIDAS; j++) {
        logits[j] = b1_int32[j];
        for (int i = 0; i < OCULTA; i++) logits[j] += (int32_t)h_q[i] * (int32_t)W1_q[i * SAIDAS + j];
    }

    int classe = 0;
    int32_t max_val = logits[0];
    for (int j = 1; j < SAIDAS; j++) {
        if (logits[j] > max_val) { max_val = logits[j]; classe = j; }
    }

    printf("Classe prevista: %d\n", classe);
    return 0;
}
