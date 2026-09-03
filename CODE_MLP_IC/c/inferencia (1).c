#include <stdio.h>
#include <stdint.h>
#include <math.h> 

// Como colocamos na mesma pasta, ele acha direto:
#include "pesos.h" 

#define ENTRADAS 784
#define OCULTA 64
#define SAIDAS 10

int32_t quantizar(float valor, float escala) {
    float q = roundf(valor / escala);
    if (q > 127.0f) return 127;
    if (q < -127.0f) return -127;
    return (int32_t)q;
}

int prever_int8(const float *imagem_entrada) {
    int32_t x_q[ENTRADAS];
    int32_t h_q[OCULTA];
    int32_t logits_int32[SAIDAS];

    for (int i = 0; i < ENTRADAS; i++) {
        float xs = 0.0f;
        if (scaler_scale[i] != 0.0f) {
            xs = (imagem_entrada[i] - scaler_mean[i]) / scaler_scale[i];
        }
        x_q[i] = quantizar(xs, s_x);
    }

    for (int j = 0; j < OCULTA; j++) {
        int32_t soma = b0_int32[j]; 
        for (int i = 0; i < ENTRADAS; i++) {
            soma += x_q[i] * W0_q[i * OCULTA + j];
        }
        float z1 = (float)soma * (s_x * s_W0);
        float h_float = (z1 > 0.0f) ? z1 : 0.0f; 
        h_q[j] = quantizar(h_float, s_h);
    }

    for (int j = 0; j < SAIDAS; j++) {
        int32_t soma = b1_int32[j];
        for (int i = 0; i < OCULTA; i++) {
            soma += h_q[i] * W1_q[i * SAIDAS + j];
        }
        logits_int32[j] = soma; 
    }

    int classe_prevista = 0;
    int32_t max_val = logits_int32[0];
    
    for (int j = 1; j < SAIDAS; j++) {
        if (logits_int32[j] > max_val) {
            max_val = logits_int32[j];
            classe_prevista = j;
        }
    }

    return classe_prevista;
}

int main() {
    float imagem_teste[ENTRADAS] = {0.0f}; 
    
    // Desenha uma linha vertical branca (pixel 255) bem no meio da imagem (um dígito "1")
    for(int i = 4; i < 24; i++) {
        imagem_teste[i * 28 + 14] = 255.0f; 
    }
    
    printf("--- Teste do Motor de Inferencia C (int8) ---\n");
    int classe = prever_int8(imagem_teste);
    printf("Classe prevista pelo modelo C: %d\n", classe);
    
    return 0;
}
