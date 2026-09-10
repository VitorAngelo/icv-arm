/*
 * dot_i8_ref.h - Produto escalar int8 de referencia (escalar).
 *
 * Referencia de correcao para os kernels NEON. Deve ser compilado com
 * -fno-tree-vectorize (ver Makefile), para que a referencia seja de fato
 * escalar e independente de qualquer vetorizacao.
 */
#ifndef DOT_I8_REF_H
#define DOT_I8_REF_H

#include <stdint.h>

/* soma_{i} a[i] * b[i] acumulada em int32 (mesma aritmetica da MLP int8). */
int32_t dot_i8_ref(const int8_t *a, const int8_t *b, int len);

/* Mesma soma acumulada em int64: resultado matematico exato, usado para
 * provar que a soma em int32 nao estourou. */
int64_t dot_i8_ref64(const int8_t *a, const int8_t *b, int len);

#endif /* DOT_I8_REF_H */
