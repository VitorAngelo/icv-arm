/*
 * dot_neon.c
 *
 * Produto interno (dot product) em 3 precisões:
 *   - int8, int32, float32
 *
 * Para cada precisão: implementação escalar (referência)
 * e implementação vetorial com intrinsics NEON (ARMv8 / aarch64).
 *
 * Objetivo: verificar corretude das versões NEON contra as escalares.
 * Medição de desempenho será adicionada em etapa posterior.
 *
 * Compilar:
 *   aarch64-linux-gnu-gcc -O2 -static -o dot_neon dot_neon.c
 *
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <arm_neon.h>

/* ================================================================== */
/* ESCALARES — referência, sem modificação               */
/* ================================================================== */

int32_t dot_product_int8(const int8_t *a, const int8_t *b, int len)
{
    int32_t ret = 0;
    for (int i = 0; i < len; i++)
        ret += a[i] * b[i];
    return ret;
}

int32_t dot_product_int32(const int32_t *a, const int32_t *b, int len)
{
    int32_t ret = 0;
    for (int i = 0; i < len; i++)
        ret += a[i] * b[i];
    return ret;
}

float dot_product_float32(const float *a, const float *b, int len)
{
    float ret = 0.0f;
    for (int i = 0; i < len; i++)
        ret += a[i] * b[i];
    return ret;
}

/* ================================================================== */
/* NEON                                                                */
/* ================================================================== */

/*
 * int8 — 16 elementos por iteração (128 bits).
 *
 * int8x8 × int8x8 → int16x8  (vmull_s8): necessário pois
 * int8 × int8 pode estourar int8, resultado cabe em int16.
 *
 * vpadalq_s16: acumula pares de int16 em int32 (widening pair-wise add),
 * evitando overflow no acumulador.
 */
int32_t dot_product_int8_neon(const int8_t *a, const int8_t *b, int len)
{
    int32x4_t acc = vdupq_n_s32(0);

    for (int i = 0; i < len; i += 16) {
        int8x16_t va = vld1q_s8(&a[i]);
        int8x16_t vb = vld1q_s8(&b[i]);

        /* Multiplica metades inferior e superior separadamente */
        int16x8_t prod_lo = vmull_s8(vget_low_s8(va),  vget_low_s8(vb));
        int16x8_t prod_hi = vmull_s8(vget_high_s8(va), vget_high_s8(vb));

        /* Acumula em int32 via widening add */
        acc = vpadalq_s16(acc, prod_lo);
        acc = vpadalq_s16(acc, prod_hi);
    }

    /* Redução horizontal: int32x4 → escalar */
    int32x2_t s = vadd_s32(vget_low_s32(acc), vget_high_s32(acc));
    s = vpadd_s32(s, s);
    return vget_lane_s32(s, 0);
}

/*
 * int32 — 4 elementos por iteração.
 *
 * vmlaq_s32: acc += a * b (multiply-accumulate vetorial).
 * Sem necessidade de widening pois o acumulador já é int32.
 */
int32_t dot_product_int32_neon(const int32_t *a, const int32_t *b, int len)
{
    int32x4_t acc = vdupq_n_s32(0);

    for (int i = 0; i < len; i += 4) {
        int32x4_t va = vld1q_s32(&a[i]);
        int32x4_t vb = vld1q_s32(&b[i]);
        acc = vmlaq_s32(acc, va, vb);
    }

    int32x2_t s = vadd_s32(vget_low_s32(acc), vget_high_s32(acc));
    s = vpadd_s32(s, s);
    return vget_lane_s32(s, 0);
}

/*
 * float32 — 4 elementos por iteração.
 *
 * vmlaq_f32: acc += a * b.
 *
 * NOTA: o resultado pode diferir levemente da versão escalar.
 * Ponto flutuante não é associativo — a ordem de acumulação
 * vetorial difere da escalar, gerando erros de arredondamento
 * distintos. Isso é comportamento correto, não um bug.
 */
float dot_product_float32_neon(const float *a, const float *b, int len)
{
    float32x4_t acc = vdupq_n_f32(0.0f);

    for (int i = 0; i < len; i += 4) {
        float32x4_t va = vld1q_f32(&a[i]);
        float32x4_t vb = vld1q_f32(&b[i]);
        acc = vmlaq_f32(acc, va, vb);
    }

    float32x2_t s = vadd_f32(vget_low_f32(acc), vget_high_f32(acc));
    s = vpadd_f32(s, s);
    return vget_lane_f32(s, 0);
}

/* ================================================================== */
/* MAIN                                                                */
/* ================================================================== */

#define LEN (16 * 1024)

int main(void)
{
    int8_t  *ai8  = malloc(LEN);
    int8_t  *bi8  = malloc(LEN);
    int32_t *ai32 = malloc(LEN * sizeof(int32_t));
    int32_t *bi32 = malloc(LEN * sizeof(int32_t));
    float   *af   = malloc(LEN * sizeof(float));
    float   *bf   = malloc(LEN * sizeof(float));

    if (!ai8 || !bi8 || !ai32 || !bi32 || !af || !bf) {
        perror("malloc");
        return 1;
    }

    /* Mesma semente do professor — resultados reproduzíveis */
    srand(0);
    for (int i = 0; i < LEN; i++) {
        ai8[i]  = rand();
        bi8[i]  = rand();
        ai32[i] = ai8[i];
        bi32[i] = bi8[i];
        af[i]   = ai8[i];
        bf[i]   = bi8[i];
    }

    /* Escalares */
    int32_t r_i8  = dot_product_int8   (ai8,  bi8,  LEN);
    int32_t r_i32 = dot_product_int32  (ai32, bi32, LEN);
    float   r_f   = dot_product_float32(af,   bf,   LEN);

    /* NEON */
    int32_t r_i8n  = dot_product_int8_neon   (ai8,  bi8,  LEN);
    int32_t r_i32n = dot_product_int32_neon  (ai32, bi32, LEN);
    float   r_fn   = dot_product_float32_neon(af,   bf,   LEN);

    /* Verificação de corretude */
    printf("%-20s %12s %12s %6s\n", "funcao", "escalar", "neon", "ok?");
    printf("%-20s %12d %12d %6s\n", "int8",    r_i8,  r_i8n,  r_i8n  == r_i8  ? "OK" : "DIFF");
    printf("%-20s %12d %12d %6s\n", "int32",   r_i32, r_i32n, r_i32n == r_i32 ? "OK" : "DIFF");
    printf("%-20s %12.1f %12.1f %6s\n", "float32", r_f, r_fn, "ver");

    free(ai8);  free(bi8);
    free(ai32); free(bi32);
    free(af);   free(bf);
    return 0;
}
