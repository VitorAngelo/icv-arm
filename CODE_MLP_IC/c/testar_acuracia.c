#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "pesos.h" 

#define ENTRADAS 784
#define OCULTA 64
#define SAIDAS 10
#define TEST_SIZE 10000

float relu(float z) { return (z > 0.0f) ? z : 0.0f; }

int32_t quantizar(float valor, float escala) {
    float q = roundf(valor / escala);
    if (q > 127.0f) return 127;
    if (q < -127.0f) return -127;
    return (int32_t)q;
}

// Inferência 1: Float32 (Referência)
int prever_float(const float *entrada) {
    float Xs[ENTRADAS], h[OCULTA], logits[SAIDAS];

    for (int i = 0; i < ENTRADAS; i++) {
        Xs[i] = (scaler_scale[i] != 0.0f) ? (entrada[i] - scaler_mean[i]) / scaler_scale[i] : 0.0f;
    }
    for (int j = 0; j < OCULTA; j++) {
        float soma = (float)b0_int32[j] * (s_x * s_W0);
        for (int i = 0; i < ENTRADAS; i++) soma += Xs[i] * ((float)W0_q[i * OCULTA + j] * s_W0);
        h[j] = relu(soma);
    }
    for (int j = 0; j < SAIDAS; j++) {
        float soma = (float)b1_int32[j] * (s_h * s_W1);
        for (int i = 0; i < OCULTA; i++) soma += h[i] * ((float)W1_q[i * SAIDAS + j] * s_W1);
        logits[j] = soma;
    }

    int classe = 0;
    float max_val = logits[0];
    for (int j = 1; j < SAIDAS; j++) {
        if (logits[j] > max_val) { max_val = logits[j]; classe = j; }
    }
    return classe;
}

// Inferência 2: Int8 Puro (Simulação Estrita ARM sem FPU)
int prever_int8_puro(const float *entrada) {
    int8_t x_q[ENTRADAS], h_q[OCULTA];
    int32_t logits[SAIDAS];

    for (int i = 0; i < ENTRADAS; i++) {
        float xs = (scaler_scale[i] != 0.0f) ? (entrada[i] - scaler_mean[i]) / scaler_scale[i] : 0.0f;
        x_q[i] = quantizar(xs, s_x);
    }
    for (int j = 0; j < OCULTA; j++) {
        int32_t soma = b0_int32[j];
        for (int i = 0; i < ENTRADAS; i++) {
            // ==============================================================================
            // CORRECAO PRODUTO LINEAR: MAC forçado em int8_t e acumulado em int32_t
            // ==============================================================================
            soma += (int32_t)( (int8_t)x_q[i] * (int8_t)W0_q[i * OCULTA + j] );
        }
        float z1 = (float)soma * (s_x * s_W0);
        h_q[j] = quantizar((z1 > 0.0f) ? z1 : 0.0f, s_h);
    }
    for (int j = 0; j < SAIDAS; j++) {
        logits[j] = b1_int32[j];
        for (int i = 0; i < OCULTA; i++) {
            // ==============================================================================
            // CORRECAO PRODUTO LINEAR: MAC forçado em int8_t e acumulado em int32_t
            // ==============================================================================
            logits[j] += (int32_t)( (int8_t)h_q[i] * (int8_t)W1_q[i * SAIDAS + j] );
        }
    }

    int classe = 0;
    int32_t max_val = logits[0];
    for (int j = 1; j < SAIDAS; j++) {
        if (logits[j] > max_val) { max_val = logits[j]; classe = j; }
    }
    return classe;
}

// Orquestrador de Validação
int main() {
    printf("Alocando memoria para 10.000 imagens...\n");
    float *X_te = (float *)malloc(TEST_SIZE * ENTRADAS * sizeof(float));
    int32_t *y_te = (int32_t *)malloc(TEST_SIZE * sizeof(int32_t));

    FILE *f_X = fopen("X_te.bin", "rb");
    FILE *f_y = fopen("y_te.bin", "rb");

    if (!f_X || !f_y) {
        printf("ERRO: Nao foi possivel abrir os arquivos .bin.\n");
        printf("Certifique-se de que X_te.bin e y_te.bin estao na pasta /home/daviz/c\n");
        return 1;
    }

    fread(X_te, sizeof(float), TEST_SIZE * ENTRADAS, f_X);
    fread(y_te, sizeof(int32_t), TEST_SIZE, f_y);
    fclose(f_X);
    fclose(f_y);

    int acertos_float = 0;
    int acertos_int8_puro = 0;
    int divergencias = 0;
    
    int matriz_confusao[SAIDAS][SAIDAS] = {0}; 

    printf("Iniciando bateria de testes...\n");
    
    for (int i = 0; i < TEST_SIZE; i++) {
        const float *imagem_atual = &X_te[i * ENTRADAS];
        
        int pred_f = prever_float(imagem_atual);
        int pred_q_puro = prever_int8_puro(imagem_atual);
        int real = y_te[i];

        if (pred_f == real) acertos_float++;
        if (pred_q_puro == real) acertos_int8_puro++;
        if (pred_f != pred_q_puro) divergencias++;

        matriz_confusao[real][pred_q_puro]++;
    }

    printf("\n======================================================\n");
    printf("     RELATORIO DE VALIDACAO C - SIMULACAO ARM NEON    \n");
    printf("======================================================\n");
    printf("Total de Amostras Testadas: %d\n", TEST_SIZE);
    printf("Acuracia Float32 (Referencia) : %.4f\n", (float)acertos_float / TEST_SIZE);
    printf("Acuracia Int8 Puro (NEON SIM) : %.4f\n", (float)acertos_int8_puro / TEST_SIZE);
    printf("Decisoes Alteradas (Float vs Int8 Puro): %d / %d\n", divergencias, TEST_SIZE);
    
    printf("\n==== MATRIZ DE CONFUSAO (INT8 PURO em C) ====\n");
    printf("   |    0    1    2    3    4    5    6    7    8    9\n");
    printf("---+--------------------------------------------------\n");
    for(int i = 0; i < SAIDAS; i++) {
        printf(" %d |", i);
        for(int j = 0; j < SAIDAS; j++) {
            if (i == j) {
                printf(" [%2d]", matriz_confusao[i][j]); 
            } else {
                printf(" %4d", matriz_confusao[i][j]);
            }
        }
        printf("\n");
    }
    printf("======================================================\n");

    free(X_te);
    free(y_te);
    return 0;
}
