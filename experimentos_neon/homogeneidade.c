/*
 * homogeneidade.c
 *
 * Pergunta: os 4 nucleos Cortex-A72 do Pi 4 tem performance
 * igual na pratica, ou existe assimetria empirica?
 *
 * Compilar:
 *   gcc -O2 -march=armv8-a+simd homogeneidade.c -lpthread -o homogeneidade
 *
 * Uso:
 *   ./homogeneidade <nucleo_inicial> <tamanho_kb>
 *   ./homogeneidade 0 16
 *
 * Roda o mesmo dot product escalar e NEON i8 nos 4 nucleos
 * em execucoes separadas com cache despejado antes de cada medicao.
 */

#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <arm_neon.h>

#define NCORES   4
#define FLUSH_SZ (2 * 1024 * 1024)   /* maior que o L2 (1 MB) do Pi 4 */

int32_t dot_scalar_i8(const int8_t *a, const int8_t *b, int len)
{
    int32_t ret = 0;
    for (int i = 0; i < len; i++)
        ret += a[i] * b[i];
    return ret;
}

int32_t dot_neon_i8(const int8_t *a, const int8_t *b, int len)
{
    int32x4_t acc = vdupq_n_s32(0);
    for (int i = 0; i < len; i += 16) {
        int8x16_t va = vld1q_s8(&a[i]);
        int8x16_t vb = vld1q_s8(&b[i]);
        acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(va),  vget_low_s8(vb)));
        acc = vpadalq_s16(acc, vmull_s8(vget_high_s8(va), vget_high_s8(vb)));
    }
    int32x2_t s = vadd_s32(vget_low_s32(acc), vget_high_s32(acc));
    s = vpadd_s32(s, s);
    return vget_lane_s32(s, 0);
}

void core_affinity(int core_id)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0) {
        perror("pthread_setaffinity_np failed");
        exit(1);
    }
    sched_yield();
}

void despejar_cache(void)
{
    static volatile uint8_t buf[FLUSH_SZ];
    volatile uint64_t s = 0;
    for (int i = 0; i < FLUSH_SZ; i++) s += buf[i];
    (void)s;
}

int main(int argc, char *argv[])
{
    struct timespec t1, t2, t3;
    long long ns_scalar, ns_neon;
    long long resultados_scalar[NCORES], resultados_neon[NCORES];
    int klen, rseed = 0;

    if (argc < 3) {
        fprintf(stderr, "%s <nucleo_inicial> <tamanho_kb>\n", argv[0]);
        exit(1);
    }
    sscanf(argv[2], "%d", &klen);
    klen *= 1024;
    srand(rseed);

    int8_t *a = aligned_alloc(64, klen);
    int8_t *b = aligned_alloc(64, klen);
    if (!a || !b) { perror("alloc"); return 1; }

    for (int i = 0; i < klen; i++) {
        a[i] = rand();
        b[i] = rand();
    }

    fprintf(stderr, "klen=%d  working_set=%dkb  L1D=32kb  L2=1MB\n\n",
            klen / 1024, (2 * klen) / 1024);

    printf("nucleo ; klen ; scalar_ns ; neon_ns ; speedup\n");

    for (int c = 0; c < NCORES; c++) {
        core_affinity(c);
        despejar_cache();

        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t1);
        volatile int32_t rs = dot_scalar_i8(a, b, klen);
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t2);
        volatile int32_t rn = dot_neon_i8(a, b, klen);
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t3);

        ns_scalar = (t2.tv_sec - t1.tv_sec) * 1000000000LL + (t2.tv_nsec - t1.tv_nsec);
        ns_neon   = (t3.tv_sec - t2.tv_sec) * 1000000000LL + (t3.tv_nsec - t2.tv_nsec);
        resultados_scalar[c] = ns_scalar;
        resultados_neon[c]   = ns_neon;

        printf("%d ; %d ; %lld ; %lld ; %.2f\n",
               c, klen, ns_scalar, ns_neon,
               (double)ns_scalar / ns_neon);

        (void)rs; (void)rn;
    }

    fprintf(stderr, "\nvariancia NEON relativa ao nucleo 0:\n");
    for (int c = 0; c < NCORES; c++)
        fprintf(stderr, "  nucleo %d: %+.1f%%\n",
                c, (resultados_neon[c] - resultados_neon[0]) * 100.0 / resultados_neon[0]);

    free(a); free(b);
    return 0;
}
