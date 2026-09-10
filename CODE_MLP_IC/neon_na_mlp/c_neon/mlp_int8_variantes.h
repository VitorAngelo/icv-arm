/*
 * mlp_int8_variantes.h - Variantes escalares da inferencia int8 para o benchmark.
 *
 * As duas funcoes abaixo sao c_modular/mlp_int8.c SEM NENHUMA ALTERACAO no
 * codigo-fonte, compilado duas vezes pelo Makefile:
 *
 *   mlp_int8_escalar_inferir  compilado com -O2 -fno-tree-vectorize:
 *                             escalar puro, sem nenhuma instrucao NEON.
 *   mlp_int8_autovet_inferir  compilado com -O2: o GCC vetoriza os lacos
 *                             automaticamente (smull/saddw), como mostrado no
 *                             relatorio_neon.txt.
 *
 * Em cada objeto, o simbolo mlp_int8_inferir e renomeado com objcopy
 * --redefine-sym, para que as duas versoes coexistam no mesmo binario e
 * sejam medidas nas mesmas condicoes que mlp_int8_neon_inferir.
 */
#ifndef MLP_INT8_VARIANTES_H
#define MLP_INT8_VARIANTES_H

#include <stdint.h>

#include "mlp.h"

void mlp_int8_escalar_inferir(const uint8_t pixels[MLP_ENTRADAS], int32_t logits[MLP_SAIDAS]);
void mlp_int8_autovet_inferir(const uint8_t pixels[MLP_ENTRADAS], int32_t logits[MLP_SAIDAS]);

#endif /* MLP_INT8_VARIANTES_H */
