#include <stdio.h>
#include "pesos.h"

#define ENTRADAS 784
#define OCULTA 64
#define SAIDAS 10

float relu(float z) { return (z > 0.0f) ? z : 0.0f; }

int main() {
    float imagem[ENTRADAS] = {0.0f};
    for(int i = 4; i < 24; i++) imagem[i * 28 + 14] = 255.0f; // Desenha o "1"

    float Xs[ENTRADAS];
    for (int i = 0; i < ENTRADAS; i++) {
        Xs[i] = (scaler_scale[i] != 0.0f) ? (imagem[i] - scaler_mean[i]) / scaler_scale[i] : 0.0f;
    }

    float h[OCULTA];
    for (int j = 0; j < OCULTA; j++) {
        float soma = (float)b0_int32[j] * (s_x * s_W0); // Desquantiza o bias
        for (int i = 0; i < ENTRADAS; i++) {
            float peso_float = (float)W0_q[i * OCULTA + j] * s_W0; // Desquantiza o peso
            soma += Xs[i] * peso_float;
        }
        h[j] = relu(soma);
    }

    float logits[SAIDAS];
    for (int j = 0; j < SAIDAS; j++) {
        logits[j] = (float)b1_int32[j] * (s_h * s_W1);
        for (int i = 0; i < OCULTA; i++) {
            float peso_float = (float)W1_q[i * SAIDAS + j] * s_W1;
            logits[j] += h[i] * peso_float;
        }
    }

    int classe = 0;
    float max_val = logits[0];
    for (int j = 1; j < SAIDAS; j++) {
        if (logits[j] > max_val) { max_val = logits[j]; classe = j; }
    }

    printf("Classe prevista: %d\n", classe);
    return 0;
}
