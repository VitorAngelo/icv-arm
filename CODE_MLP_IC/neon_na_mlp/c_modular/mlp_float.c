/*
 * mlp_float.c - Inferencia de referencia em float32.
 *
 * O que este modulo prova: que os pesos exportados e a matematica em C
 * reproduzem o modelo Python original. Nao ha quantizacao, recorte nem
 * otimizacao. Resultado esperado: as mesmas 10.000 decisoes do Python e
 * logits identicos, bit a bit, a ref_float32.bin.
 *
 * Origem das constantes (todas em mlp_pesos_float.h, gerado pelo Python):
 *   MLPF_MEDIA[i]       scaler.mean_[i]       media do pixel i no treino
 *   MLPF_ESCALA[i]      scaler.scale_[i]      desvio-padrao do pixel i
 *   MLPF_W0[j*784 + i]  mlp.coefs_[0][i][j]   entrada i -> neuronio oculto j
 *   MLPF_B0[j]          mlp.intercepts_[0][j]
 *   MLPF_W1[k*64 + j]   mlp.coefs_[1][j][k]   neuronio oculto j -> classe k
 *   MLPF_B1[k]          mlp.intercepts_[1][k]
 */
#include "mlp.h"
#include "mlp_pesos_float.h"

/* Conferencia em tempo de compilacao: se o header gerado tiver outro tamanho,
 * a compilacao falha aqui em vez de ler memoria fora dos arrays. */
_Static_assert(sizeof(MLPF_MEDIA)  / sizeof(MLPF_MEDIA[0])  == MLP_ENTRADAS, "MLPF_MEDIA com tamanho errado");
_Static_assert(sizeof(MLPF_ESCALA) / sizeof(MLPF_ESCALA[0]) == MLP_ENTRADAS, "MLPF_ESCALA com tamanho errado");
_Static_assert(sizeof(MLPF_W0) / sizeof(MLPF_W0[0]) == MLP_OCULTA * MLP_ENTRADAS, "MLPF_W0 com tamanho errado");
_Static_assert(sizeof(MLPF_B0) / sizeof(MLPF_B0[0]) == MLP_OCULTA, "MLPF_B0 com tamanho errado");
_Static_assert(sizeof(MLPF_W1) / sizeof(MLPF_W1[0]) == MLP_SAIDAS * MLP_OCULTA, "MLPF_W1 com tamanho errado");
_Static_assert(sizeof(MLPF_B1) / sizeof(MLPF_B1[0]) == MLP_SAIDAS, "MLPF_B1 com tamanho errado");

void mlp_float_inferir(const uint8_t pixels[MLP_ENTRADAS], float logits[MLP_SAIDAS])
{
    float x[MLP_ENTRADAS];   /* entrada padronizada, em desvios-padrao */
    float h[MLP_OCULTA];     /* ativacao dos 64 neuronios ocultos (apos ReLU) */

    /* 1) Padronizacao (StandardScaler): x = (pixel - media) / desvio. */
    for (int i = 0; i < MLP_ENTRADAS; i++) {
        x[i] = ((float)pixels[i] - MLPF_MEDIA[i]) / MLPF_ESCALA[i];
    }

    /* 2) Camada oculta: para cada neuronio j, a = b0[j] + produto escalar(W0[j], x);
     *    em seguida ReLU, h = max(a, 0). */
    for (int j = 0; j < MLP_OCULTA; j++) {
        const float *w = &MLPF_W0[j * MLP_ENTRADAS];   /* linha j: os 784 pesos do neuronio j */
        float a = MLPF_B0[j];                          /* o bias entra primeiro: ordem da especificacao */
        for (int i = 0; i < MLP_ENTRADAS; i++) {
            a += x[i] * w[i];
        }
        h[j] = (a > 0.0f) ? a : 0.0f;
    }

    /* 3) Camada de saida: para cada classe k, z = b1[k] + produto escalar(W1[k], h).
     *    Sem softmax: o argmax dos logits ja define a classe (ver mlp.h). */
    for (int k = 0; k < MLP_SAIDAS; k++) {
        const float *w = &MLPF_W1[k * MLP_OCULTA];     /* linha k: os 64 pesos da classe k */
        float z = MLPF_B1[k];
        for (int j = 0; j < MLP_OCULTA; j++) {
            z += h[j] * w[j];
        }
        logits[k] = z;
    }
}
