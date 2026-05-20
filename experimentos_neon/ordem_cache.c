/*
 * ordem_cache.c
 *
 * Pergunta: medir i8, i32 e f32 em sequencia contamina os resultados
 * porque cada funcao aquece o cache para a proxima?
 *
 * Metodo: para cada tipo, mede frio (pos-flush) e imediatamente
 * quente (sem flush). O ratio frio/quente quantifica o quanto o
 * cache estava interferindo na metodologia original do professor.
 *
 * Compilar:
 *   gcc -O2 -march=armv8-a+simd ordem_cache.c -lpthread -o ordem_cache
 *
 * Uso:
 *   ./ordem_cache <nucleo> <tamanho_kb>
 *   ./ordem_cache 0 16      (L1D - working set 32kb)
 *   ./ordem_cache 0 256     (L2  - working set 512kb)
 *   ./ordem_cache 0 2048    (DRAM)
 */

#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define FLUSH_SZ (2 * 1024 * 1024)

int32_t dot_scalar_i8(const int8_t *a, const int8_t *b, int len)
{
    int32_t ret = 0;
    for (int i = 0; i < len; i++)
        ret += a[i] * b[i];
    return ret;
}

int32_t dot_scalar_i32(const int32_t *a, const int32_t *b, int len)
{
    int32_t ret = 0;
    for (int i = 0; i < len; i++)
        ret += a[i] * b[i];
    return ret;
}

float dot_scalar_f32(const float *a, const float *b, int len)
{
    float ret = 0.0f;
    for (int i = 0; i < len; i++)
        ret += a[i] * b[i];
    return ret;
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
    long long ns_frio, ns_quente;
    int core_id, klen;

    if (argc < 3) {
        fprintf(stderr, "%s <nucleo> <tamanho_kb>\n", argv[0]);
        exit(1);
    }
    sscanf(argv[1], "%d", &core_id);
    sscanf(argv[2], "%d", &klen);
    klen *= 1024;

    core_affinity(core_id);

    int8_t  *ai8  = aligned_alloc(64, klen);
    int8_t  *bi8  = aligned_alloc(64, klen);
    int32_t *ai32 = aligned_alloc(64, klen * sizeof(int32_t));
    int32_t *bi32 = aligned_alloc(64, klen * sizeof(int32_t));
    float   *af   = aligned_alloc(64, klen * sizeof(float));
    float   *bf   = aligned_alloc(64, klen * sizeof(float));
    if (!ai8 || !bi8 || !ai32 || !bi32 || !af || !bf) { perror("alloc"); return 1; }

    for (int i = 0; i < klen; i++) {
        ai8[i]  = rand();
        bi8[i]  = rand();
        ai32[i] = ai8[i];
        bi32[i] = bi8[i];
        af[i]   = ai8[i];
        bf[i]   = bi8[i];
    }

    fprintf(stderr, "nucleo=%d  ws_i8=%dkb  ws_i32/f32=%dkb  L1D=32kb  L2=1MB\n\n",
            core_id, (2 * klen) / 1024, (8 * klen) / 1024);

    printf("tipo ; frio_ns ; quente_ns ; frio/quente\n");

    /* i8 */
    despejar_cache();
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t1);
    volatile int32_t r0 = dot_scalar_i8(ai8, bi8, klen);
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t2);
    volatile int32_t r1 = dot_scalar_i8(ai8, bi8, klen);
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t3);
    ns_frio   = (t2.tv_sec - t1.tv_sec) * 1000000000LL + (t2.tv_nsec - t1.tv_nsec);
    ns_quente = (t3.tv_sec - t2.tv_sec) * 1000000000LL + (t3.tv_nsec - t2.tv_nsec);
    printf("i8  ; %lld ; %lld ; %.2f\n", ns_frio, ns_quente, (double)ns_frio / ns_quente);
    (void)r0; (void)r1;

    /* i32 */
    despejar_cache();
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t1);
    volatile int32_t r2 = dot_scalar_i32(ai32, bi32, klen);
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t2);
    volatile int32_t r3 = dot_scalar_i32(ai32, bi32, klen);
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t3);
    ns_frio   = (t2.tv_sec - t1.tv_sec) * 1000000000LL + (t2.tv_nsec - t1.tv_nsec);
    ns_quente = (t3.tv_sec - t2.tv_sec) * 1000000000LL + (t3.tv_nsec - t2.tv_nsec);
    printf("i32 ; %lld ; %lld ; %.2f\n", ns_frio, ns_quente, (double)ns_frio / ns_quente);
    (void)r2; (void)r3;

    /* f32 */
    despejar_cache();
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t1);
    volatile float r4 = dot_scalar_f32(af, bf, klen);
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t2);
    volatile float r5 = dot_scalar_f32(af, bf, klen);
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t3);
    ns_frio   = (t2.tv_sec - t1.tv_sec) * 1000000000LL + (t2.tv_nsec - t1.tv_nsec);
    ns_quente = (t3.tv_sec - t2.tv_sec) * 1000000000LL + (t3.tv_nsec - t2.tv_nsec);
    printf("f32 ; %lld ; %lld ; %.2f\n", ns_frio, ns_quente, (double)ns_frio / ns_quente);
    (void)r4; (void)r5;

    free(ai8); free(bi8); free(ai32); free(bi32); free(af); free(bf);
    return 0;
}
