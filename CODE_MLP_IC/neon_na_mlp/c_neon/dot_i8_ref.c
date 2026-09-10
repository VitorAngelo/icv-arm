/*
 * dot_i8_ref.c - Produto escalar int8 de referencia (escalar).
 *
 * ORIGEM: funcao dot_i8 de experimentos_neon_2/homogeneidade_v2.c (codigo
 * de referencia do professor, dot.c). Corpo sem alteracao; nome padronizado.
 */
#include "dot_i8_ref.h"

int32_t dot_i8_ref(const int8_t *a, const int8_t *b, int len)
{
    int32_t acc = 0;
    for (int i = 0; i < len; i++)
        acc += (int32_t)a[i] * (int32_t)b[i];
    return acc;
}

int64_t dot_i8_ref64(const int8_t *a, const int8_t *b, int len)
{
    int64_t acc = 0;
    for (int i = 0; i < len; i++)
        acc += (int64_t)a[i] * (int64_t)b[i];
    return acc;
}
