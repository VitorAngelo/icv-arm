/*
 * dot_neon_timing.c
 *
 * Produto interno (dot product) em 3 precisoes: int8, int32, float32.
 * Para cada precisao: implementacao escalar (referencia) e NEON.
 * Medicao de tempo via clock_gettime, fixacao de nucleo via pthread.
 *
 * Compilar na placa:
 *   gcc -O2 -o dot_neon_timing dot_neon_timing.c -lpthread
 *
 * Usar:
 *   ./dot_neon_timing <core_id> <tamanho_kb> [semente]
 *   ./dot_neon_timing 0 16 0
 */

#define _GNU_SOURCE

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <arm_neon.h>


/* ================================================================== */
/* ESCALARES                                                           */
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
 * int8 — 16 elementos por iteracao (128 bits).
 *
 * int8x8 x int8x8 -> int16x8  (vmull_s8): necessario pois
 * int8 x int8 pode estourar int8, resultado cabe em int16.
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

        int16x8_t prod_lo = vmull_s8(vget_low_s8(va),  vget_low_s8(vb));
        int16x8_t prod_hi = vmull_s8(vget_high_s8(va), vget_high_s8(vb));

        acc = vpadalq_s16(acc, prod_lo);
        acc = vpadalq_s16(acc, prod_hi);
    }

    int32x2_t s = vadd_s32(vget_low_s32(acc), vget_high_s32(acc));
    s = vpadd_s32(s, s);
    return vget_lane_s32(s, 0);
}

/*
 * int32 — 4 elementos por iteracao.
 *
 * vmlaq_s32: acc += a * b (multiply-accumulate vetorial).
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
 * float32 — 4 elementos por iteracao.
 *
 * vmlaq_f32: acc += a * b.
 *
 * NOTA: resultado pode divergir levemente da versao escalar.
 * Ponto flutuante nao e associativo — ordem de acumulacao vetorial
 * difere da escalar, gerando erros de arredondamento distintos.
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
/* CORE AFFINITY                                                       */
/* ================================================================== */

void core_affinity(int core_id)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0) {
        perror("pthread_setaffinity_np failed");
        exit(1);
    }
}


/* ================================================================== */
/* MAIN                                                                */
/* ================================================================== */

int main(int argc, char *argv[])
{
    struct timespec t1, t2, t3, t4, t5, t6, t7;
    long long ns_i8s, ns_i32s, ns_f32s;
    long long ns_i8n, ns_i32n, ns_f32n;
    double macs_i8s, macs_i32s, macs_f32s;
    double macs_i8n, macs_i32n, macs_f32n;

    int i, rseed, core_id, klen;

    if ((argc < 3) ||
        (sscanf(argv[1], "%d", &core_id) != 1) ||
        (sscanf(argv[2], "%d", &klen)    != 1)) {
        fprintf(stderr, "%s core_num vec_len(kb)\n", argv[0]);
        exit(0);
    }
    if ((argc != 4) || (sscanf(argv[3], "%d", &rseed) != 1)) {
        rseed = 0;
    }

    fprintf(stderr, "core_id=%d klen=%d rseed=%d\n", core_id, klen, rseed);

    core_affinity(core_id);

    klen *= 1024;
    srand(rseed);

    int8_t  *ai8  = aligned_alloc(64, klen);
    int8_t  *bi8  = aligned_alloc(64, klen);
    int32_t *ai32 = aligned_alloc(64, klen * sizeof(int32_t));
    int32_t *bi32 = aligned_alloc(64, klen * sizeof(int32_t));
    float   *af   = aligned_alloc(64, klen * sizeof(float));
    float   *bf   = aligned_alloc(64, klen * sizeof(float));

    int32_t r_i8s, r_i32s, r_i8n, r_i32n;
    float   r_f32s, r_f32n;

    if (!ai8 || !bi8 || !ai32 || !bi32 || !af || !bf) {
        perror("malloc");
        return 1;
    }

    for (i = 0; i < klen; i++) {
        ai8[i]  = rand();
        bi8[i]  = rand();
        ai32[i] = ai8[i];
        bi32[i] = bi8[i];
        af[i]   = ai8[i];
        bf[i]   = bi8[i];
    }

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t1);
    r_i8s  = dot_product_int8   (ai8,  bi8,  klen);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t2);
    r_i32s = dot_product_int32  (ai32, bi32, klen);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t3);
    r_f32s = dot_product_float32(af,   bf,   klen);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t4);
    r_i8n  = dot_product_int8_neon   (ai8,  bi8,  klen);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t5);
    r_i32n = dot_product_int32_neon  (ai32, bi32, klen);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t6);
    r_f32n = dot_product_float32_neon(af,   bf,   klen);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t7);

    ns_i8s  = (t2.tv_sec-t1.tv_sec)*1000000000LL + (t2.tv_nsec-t1.tv_nsec);
    ns_i32s = (t3.tv_sec-t2.tv_sec)*1000000000LL + (t3.tv_nsec-t2.tv_nsec);
    ns_f32s = (t4.tv_sec-t3.tv_sec)*1000000000LL + (t4.tv_nsec-t3.tv_nsec);
    ns_i8n  = (t5.tv_sec-t4.tv_sec)*1000000000LL + (t5.tv_nsec-t4.tv_nsec);
    ns_i32n = (t6.tv_sec-t5.tv_sec)*1000000000LL + (t6.tv_nsec-t5.tv_nsec);
    ns_f32n = (t7.tv_sec-t6.tv_sec)*1000000000LL + (t7.tv_nsec-t6.tv_nsec);

    macs_i8s  = (double)klen / ((double)ns_i8s  * 1.0e-9);
    macs_i32s = (double)klen / ((double)ns_i32s * 1.0e-9);
    macs_f32s = (double)klen / ((double)ns_f32s * 1.0e-9);
    macs_i8n  = (double)klen / ((double)ns_i8n  * 1.0e-9);
    macs_i32n = (double)klen / ((double)ns_i32n * 1.0e-9);
    macs_f32n = (double)klen / ((double)ns_f32n * 1.0e-9);

    printf("core ; klen ; "
           "i8s_ns ; i32s_ns ; f32s_ns ; i8n_ns ; i32n_ns ; f32n_ns ; "
           "i8s_macs ; i32s_macs ; f32s_macs ; i8n_macs ; i32n_macs ; f32n_macs\n");

    printf("%d ; %d ; "
           "%lld ; %lld ; %lld ; %lld ; %lld ; %lld ; "
           "%.3e ; %.3e ; %.3e ; %.3e ; %.3e ; %.3e\n",
           core_id, klen,
           ns_i8s,  ns_i32s,  ns_f32s,  ns_i8n,  ns_i32n,  ns_f32n,
           macs_i8s, macs_i32s, macs_f32s, macs_i8n, macs_i32n, macs_f32n);

    fprintf(stderr, "escalar: %d %d %.1f\n", (int)r_i8s, r_i32s, r_f32s);
    fprintf(stderr, "neon:    %d %d %.1f\n", (int)r_i8n, r_i32n, r_f32n);
    fprintf(stderr, "i8  %s\n", r_i8n  == r_i8s  ? "OK" : "DIFF");
    fprintf(stderr, "i32 %s\n", r_i32n == r_i32s ? "OK" : "DIFF");

    free(ai8);  free(bi8);
    free(ai32); free(bi32);
    free(af);   free(bf);
    return 0;
}
