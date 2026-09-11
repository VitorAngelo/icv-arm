/*
 * mlp.h - Interface das tres MLPs 784 -> 64 -> 10 (MNIST): float puro, int16 puro, int8 puro.
 *
 * Origem: a MLP de testar_acuracia.c, separada em tres arquivos e corrigida
 * conforme os comentarios FIXME/TODO do professor:
 *   - na inferencia inteira nao pode haver nada de float;
 *   - a entrada deve chegar ja quantizada;
 *   - pesos int8 multiplicam entradas int8 sem cast para int32; a soma e acumulada em int32;
 *   - a ativacao (ReLU) e aplicada sem voltar para float e sai em int8.
 *
 * Todas as funcoes recebem os 784 pixels da imagem (uint8, 0 a 255) e escrevem
 * os 10 logits. A classe prevista e o indice do maior logit (em empate, o menor).
 *
 * Pesos: dados/mlp_pesos_float.h, dados/mlp_pesos_int16.h, dados/mlp_pesos_int8.h
 * (gerados em Python a partir do modelo treinado; cada header e incluido por um
 * unico .c). Matrizes com uma linha por neuronio: W0[j*784 + i], W1[k*64 + j].
 */
#ifndef MLP_H
#define MLP_H

#include <stdint.h>

#define MLP_ENTRADAS 784   /* 28 x 28 pixels */
#define MLP_OCULTA   64    /* neuronios da camada oculta */
#define MLP_SAIDAS   10    /* digitos 0 a 9 */

/* float puro: padronizacao, camadas e ReLU em float32, com os pesos originais. */
void mlp_float_inferir(const uint8_t pixels[MLP_ENTRADAS], float logits[MLP_SAIDAS]);

/* int16 puro: sem float. Produto 16x16 -> 32 bits, acumulacao em 64 bits. */
void mlp_int16_inferir(const uint8_t pixels[MLP_ENTRADAS], int64_t logits[MLP_SAIDAS]);

/* int8 puro: sem float. Produto 8x8 -> 16 bits, acumulacao em 32 bits. */
void mlp_int8_inferir(const uint8_t pixels[MLP_ENTRADAS], int32_t logits[MLP_SAIDAS]);

/* int8 NEON: mesma conta do int8 puro, com os produtos escalares feitos por
 * intrinsics NEON (mlp_neon_int8.c). So compila para ARM (aarch64). */
void mlp_neon_int8_inferir(const uint8_t pixels[MLP_ENTRADAS], int32_t logits[MLP_SAIDAS]);

#endif /* MLP_H */
