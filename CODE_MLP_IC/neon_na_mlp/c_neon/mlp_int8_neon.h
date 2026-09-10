/*
 * mlp_int8_neon.h - Inferencia int8 da MLP 784 -> 64 -> 10 com kernel NEON.
 *
 * Mesma especificacao de c_modular/mlp_int8.c (ver c_modular/mlp.h), com os
 * produtos escalares das duas camadas feitos por dot_i8_neon. Usa os mesmos
 * pesos (modelo/exportado/mlp_pesos_int8.h) e deve produzir logits
 * identicos, bit a bit, a mlp_int8_inferir e a ref_int8.bin.
 */
#ifndef MLP_INT8_NEON_H
#define MLP_INT8_NEON_H

#include <stdint.h>

#include "mlp.h"   /* c_modular: MLP_ENTRADAS, MLP_OCULTA, MLP_SAIDAS */

/* Valores intermediarios de uma inferencia int8, na ordem do pipeline.
 * Usado para localizar o primeiro estagio em que duas implementacoes divergem. */
typedef struct {
    int8_t  x[MLP_ENTRADAS];        /* entrada quantizada (MLP8_LUT) */
    int32_t produto[MLP_OCULTA];    /* camada 784->64: dot(x, W0[j]), sem bias */
    int32_t a[MLP_OCULTA];          /* produto + B0[j] */
    int32_t relu[MLP_OCULTA];       /* max(a, 0) */
    int8_t  h[MLP_OCULTA];          /* ativacao requantizada (int8) */
    int32_t logits[MLP_SAIDAS];     /* camada 64->10: B1[k] + dot(h, W1[k]) */
} Mlp8Estagios;

/* Inferencia completa: escreve os 10 logits int32. */
void mlp_int8_neon_inferir(const uint8_t pixels[MLP_ENTRADAS], int32_t logits[MLP_SAIDAS]);

/* Mesma inferencia, registrando todos os estagios (usado so na validacao). */
void mlp_int8_neon_estagios(const uint8_t pixels[MLP_ENTRADAS], Mlp8Estagios *e);

#endif /* MLP_INT8_NEON_H */
