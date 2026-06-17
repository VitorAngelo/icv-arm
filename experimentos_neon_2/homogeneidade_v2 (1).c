/*
 * homogeneidade_simples.c
 *
 * Experimento desenvolvido a partir do experimento anterior (homogeneidade_v2.c)
 * e do codigo de referencia dot.c, incorporando o feedback recebido sobre
 * metodologia de medicao: warmup antes de cronometrar, loop de multiplas
 * execucoes dentro de um unico intervalo e divisao pelo numero de runs.
 * Objetivo: medir homogeneidade entre os 4 nucleos do Cortex-A72 com
 * maior rigor, cobrindo tres tipos de dado (int8, int32, float32)
 * em kernels scalar e NEON.
 */

#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <arm_neon.h>

#define NCORES 4

/* --- kernels escalares (baseados em dot.c) --- */

static int32_t dot_i8(const int8_t *a, const int8_t *b, int len)
{
    int32_t acc = 0;
    for (int i = 0; i < len; i++)
        acc += (int32_t)a[i] * (int32_t)b[i];
    return acc;
}

static int32_t dot_i32(const int32_t *a, const int32_t *b, int len)
{
    int32_t acc = 0;
    for (int i = 0; i < len; i++)
        acc += a[i] * b[i];
    return acc;
}

static float dot_f32(const float *a, const float *b, int len)
{
    float acc = 0.0f;
    for (int i = 0; i < len; i++)
        acc += a[i] * b[i];
    return acc;
}

/*
 * Kernels NEON (novos).
 * int8: vmull_s8 produz int16x8 a partir de 8 pares int8;
 * vpadalq_s16 soma pares adjacentes e acumula em int32x4.
 * 16 elementos por iteracao. len deve ser multiplo de 16.
 */
static int32_t neon_dot_i8(const int8_t *a, const int8_t *b, int len)
{
    int32x4_t acc = vdupq_n_s32(0);
    for (int i = 0; i < len; i += 16) {
        int8x16_t va = vld1q_s8(a + i);
        int8x16_t vb = vld1q_s8(b + i);
        acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(va),  vget_low_s8(vb)));
        acc = vpadalq_s16(acc, vmull_s8(vget_high_s8(va), vget_high_s8(vb)));
    }
    int32x2_t s = vadd_s32(vget_low_s32(acc), vget_high_s32(acc));
    return vget_lane_s32(vpadd_s32(s, s), 0);
}

/*
 * int32: vmlaq_s32 faz multiply-accumulate de 4 elementos por iteracao.
 * len deve ser multiplo de 4.
 */
static int32_t neon_dot_i32(const int32_t *a, const int32_t *b, int len)
{
    int32x4_t acc = vdupq_n_s32(0);
    for (int i = 0; i < len; i += 4) {
        acc = vmlaq_s32(acc, vld1q_s32(a + i), vld1q_s32(b + i));
    }
    int32x2_t s = vadd_s32(vget_low_s32(acc), vget_high_s32(acc));
    return vget_lane_s32(vpadd_s32(s, s), 0);
}

/*
 * float32: vmlaq_f32 faz multiply-accumulate de 4 floats por iteracao.
 * len deve ser multiplo de 4.
 */
static float neon_dot_f32(const float *a, const float *b, int len)
{
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (int i = 0; i < len; i += 4) {
        acc = vmlaq_f32(acc, vld1q_f32(a + i), vld1q_f32(b + i));
    }
    float32x2_t s = vadd_f32(vget_low_f32(acc), vget_high_f32(acc));
    return vget_lane_f32(vpadd_f32(s, s), 0);
}

/* Afinidade de nucleo: baseada em dot.c. */
static void core_affinity(int core_id)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0) {
        perror("pthread_setaffinity_np failed");
        exit(1);
    }
}

/* Estatistica simples (nova): media e mediana por nucleo. */
static double media(const long long *v, int n)
{
    double s = 0.0;
    for (int i = 0; i < n; i++) s += (double)v[i];
    return s / n;
}

static int cmp_ll(const void *x, const void *y)
{
    long long a = *(const long long *)x;
    long long b = *(const long long *)y;
    return (a > b) - (a < b);
}

static double mediana(long long *v, int n)
{
    long long *tmp = malloc((size_t)n * sizeof(long long));
    memcpy(tmp, v, (size_t)n * sizeof(long long));
    qsort(tmp, (size_t)n, sizeof(long long), cmp_ll);
    double med = (double)tmp[n / 2];
    free(tmp);
    return med;
}

int main(int argc, char *argv[])
{
    int kb     = (argc > 1) ? atoi(argv[1]) : 16;
    int runs   = (argc > 2) ? atoi(argv[2]) : 100;
    int trials = (argc > 3) ? atoi(argv[3]) : 500;
    int seed   = (argc > 4) ? atoi(argv[4]) : 1;

    if (runs < 1) runs = 1;
    int klen = kb * 1024;
    srand((unsigned)seed);

    fprintf(stderr, "kb=%d  klen=%d  runs=%d  trials=%d  seed=%d\n",
            kb, klen, runs, trials, seed);
    fprintf(stderr, "Cada medicao = %d execucoes cronometradas juntas, depois /runs.\n", runs);
    fprintf(stderr, "Nucleo sorteado aleatoriamente (rand() %% 4) a cada trial.\n\n");

    int8_t  *ai8  = aligned_alloc(64, (size_t)klen);
    int8_t  *bi8  = aligned_alloc(64, (size_t)klen);
    int32_t *ai32 = aligned_alloc(64, (size_t)klen * sizeof(int32_t));
    int32_t *bi32 = aligned_alloc(64, (size_t)klen * sizeof(int32_t));
    float   *af   = aligned_alloc(64, (size_t)klen * sizeof(float));
    float   *bf   = aligned_alloc(64, (size_t)klen * sizeof(float));

    if (!ai8 || !bi8 || !ai32 || !bi32 || !af || !bf) { perror("alloc"); return 1; }

    for (int i = 0; i < klen; i++) {
        ai8[i]  = (int8_t)(rand() & 0xFF);
        bi8[i]  = (int8_t)(rand() & 0xFF);
        ai32[i] = (int32_t)ai8[i];
        bi32[i] = (int32_t)bi8[i];
        af[i]   = (float)ai8[i];
        bf[i]   = (float)bi8[i];
    }

    long long *s_i8[NCORES],  *n_i8[NCORES];
    long long *s_i32[NCORES], *n_i32[NCORES];
    long long *s_f32[NCORES], *n_f32[NCORES];
    int count[NCORES];
    memset(count, 0, sizeof(count));

    for (int c = 0; c < NCORES; c++) {
        s_i8[c]  = malloc((size_t)trials * sizeof(long long));
        n_i8[c]  = malloc((size_t)trials * sizeof(long long));
        s_i32[c] = malloc((size_t)trials * sizeof(long long));
        n_i32[c] = malloc((size_t)trials * sizeof(long long));
        s_f32[c] = malloc((size_t)trials * sizeof(long long));
        n_f32[c] = malloc((size_t)trials * sizeof(long long));
        if (!s_i8[c] || !n_i8[c] || !s_i32[c] || !n_i32[c] || !s_f32[c] || !n_f32[c]) {
            perror("malloc"); return 1;
        }
    }

    printf("trial,core,si8_ns,ni8_ns,si32_ns,ni32_ns,sf32_ns,nf32_ns\n");

    struct timespec t1, t2;
    volatile int32_t sink_i = 0;
    volatile float   sink_f = 0.0f;

    for (int t = 0; t < trials; t++) {
        int core = rand() % NCORES;
        core_affinity(core);

        long long ns_si8, ns_ni8, ns_si32, ns_ni32, ns_sf32, ns_nf32;

        sink_i += dot_i8(ai8, bi8, klen);
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t1);
        for (int r = 0; r < runs; r++) sink_i += dot_i8(ai8, bi8, klen);
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t2);
        ns_si8 = ((t2.tv_sec - t1.tv_sec) * 1000000000LL + (t2.tv_nsec - t1.tv_nsec)) / runs;

        sink_i += neon_dot_i8(ai8, bi8, klen);
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t1);
        for (int r = 0; r < runs; r++) sink_i += neon_dot_i8(ai8, bi8, klen);
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t2);
        ns_ni8 = ((t2.tv_sec - t1.tv_sec) * 1000000000LL + (t2.tv_nsec - t1.tv_nsec)) / runs;

        sink_i += dot_i32(ai32, bi32, klen);
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t1);
        for (int r = 0; r < runs; r++) sink_i += dot_i32(ai32, bi32, klen);
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t2);
        ns_si32 = ((t2.tv_sec - t1.tv_sec) * 1000000000LL + (t2.tv_nsec - t1.tv_nsec)) / runs;

        sink_i += neon_dot_i32(ai32, bi32, klen);
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t1);
        for (int r = 0; r < runs; r++) sink_i += neon_dot_i32(ai32, bi32, klen);
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t2);
        ns_ni32 = ((t2.tv_sec - t1.tv_sec) * 1000000000LL + (t2.tv_nsec - t1.tv_nsec)) / runs;

        sink_f += dot_f32(af, bf, klen);
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t1);
        for (int r = 0; r < runs; r++) sink_f += dot_f32(af, bf, klen);
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t2);
        ns_sf32 = ((t2.tv_sec - t1.tv_sec) * 1000000000LL + (t2.tv_nsec - t1.tv_nsec)) / runs;

        sink_f += neon_dot_f32(af, bf, klen);
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t1);
        for (int r = 0; r < runs; r++) sink_f += neon_dot_f32(af, bf, klen);
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t2);
        ns_nf32 = ((t2.tv_sec - t1.tv_sec) * 1000000000LL + (t2.tv_nsec - t1.tv_nsec)) / runs;

        int idx = count[core]++;
        s_i8[core][idx]  = ns_si8;   n_i8[core][idx]  = ns_ni8;
        s_i32[core][idx] = ns_si32;  n_i32[core][idx] = ns_ni32;
        s_f32[core][idx] = ns_sf32;  n_f32[core][idx] = ns_nf32;

        printf("%d,%d,%lld,%lld,%lld,%lld,%lld,%lld\n",
               t, core, ns_si8, ns_ni8, ns_si32, ns_ni32, ns_sf32, ns_nf32);
    }

    (void)sink_i; (void)sink_f;

    fprintf(stderr, "\n%-8s %4s | %14s %14s | %14s %14s | %14s %14s\n",
            "nucleo", "n",
            "si8(med/med)", "ni8(med/med)",
            "si32(med/med)", "ni32(med/med)",
            "sf32(med/med)", "nf32(med/med)");
    fprintf(stderr, "---------|-----|----------------|----------------|----------------|----------------\n");

    for (int c = 0; c < NCORES; c++) {
        int n = count[c];
        if (n == 0) { fprintf(stderr, "core%d: sem amostras.\n", c); continue; }
        fprintf(stderr,
            "core%-4d %4d | %6.0f/%-6.0f  %6.0f/%-6.0f | %6.0f/%-6.0f  %6.0f/%-6.0f | %6.0f/%-6.0f  %6.0f/%-6.0f\n",
            c, n,
            media(s_i8[c], n),  mediana(s_i8[c], n),
            media(n_i8[c], n),  mediana(n_i8[c], n),
            media(s_i32[c], n), mediana(s_i32[c], n),
            media(n_i32[c], n), mediana(n_i32[c], n),
            media(s_f32[c], n), mediana(s_f32[c], n),
            media(n_f32[c], n), mediana(n_f32[c], n));
    }

    double mn_min = 1e18, mn_max = 0.0;
    for (int c = 0; c < NCORES; c++) {
        if (count[c] == 0) continue;
        double m = mediana(n_i8[c], count[c]);
        if (m < mn_min) mn_min = m;
        if (m > mn_max) mn_max = m;
    }
    double spread = (mn_max - mn_min) / mn_min * 100.0;
    fprintf(stderr, "\nSpread de mediana NEON i8 entre nucleos: %.1f%%\n", spread);
    if (spread < 5.0)
        fprintf(stderr, "-> Nucleos homogeneos (spread < 5%%).\n");
    else
        fprintf(stderr, "-> Assimetria detectada (spread >= 5%%).\n"
                        "   Sem counterbalancing, pode ser artefato de DVFS/ordem.\n");

    for (int c = 0; c < NCORES; c++) {
        free(s_i8[c]);  free(n_i8[c]);
        free(s_i32[c]); free(n_i32[c]);
        free(s_f32[c]); free(n_f32[c]);
    }
    free(ai8); free(bi8);
    free(ai32); free(bi32);
    free(af); free(bf);
    return 0;
}
