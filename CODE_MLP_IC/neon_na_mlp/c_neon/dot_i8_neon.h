/*
 * dot_i8_neon.h - Interface do kernel NEON de produto escalar int8.
 *
 * Operacao: soma_{i=0}^{len-1} a[i] * b[i], com a e b em int8 e resultado int32.
 *
 * Restricoes de uso (responsabilidade de quem chama):
 *   - len deve ser multiplo de DOT_I8_NEON_BLOCO (16): o kernel nao trata
 *     elementos restantes (tail) e leria alem do vetor;
 *   - o resultado exato deve caber em int32. Cada produto vale no maximo
 *     |-128 * -128| = 16.384; para len <= 131.072 a soma nunca estoura.
 *
 * Na MLP 784 -> 64 -> 10 os tamanhos sao 784 = 16 * 49 e 64 = 16 * 4.
 */
#ifndef DOT_I8_NEON_H
#define DOT_I8_NEON_H

#include <stdint.h>

#define DOT_I8_NEON_BLOCO 16

int32_t dot_i8_neon(const int8_t *a, const int8_t *b, int len);

#endif /* DOT_I8_NEON_H */
