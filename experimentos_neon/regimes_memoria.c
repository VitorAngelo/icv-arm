/*
 * regimes_memoria.c
 *
 * Pergunta: o speedup do NEON e constante ou depende de onde
 * o dado esta na hierarquia de memoria (L1D, L2, DRAM)?
 *
 * Metodo: varre klen de 1 KB a 4 MB em progressao x2.
 * Para cada tamanho: warmup + 50 repeticoes de escalar e NEON i8.
 * Reporta throughput (MACs/s) e speedup por regime.
 *
 * Compilar:
 *   gcc -O2 -march=armv8-a+simd regimes_memoria.c -lpthread -o regimes_memoria
 *
 * Uso:
 *   ./regimes_memoria <nucleo>
 *   ./regimes_memoria 0
 */

#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <arm_neon.h>

#define REPS    50
#define WARMUP   5
#define MAX_KB  4096

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

/* Retorna tempo medio por chamada em ns apos REPS repeticoes */
double medir_media_ns(int usar_neon, const int8_t *a, const int8_t *b, int n)
{
    struct timespec t1, t2;
    volatile int32_t sumidouro = 0;

    for (int r = 0; r < WARMUP; r++)
        sumidouro += usar_neon ? dot_neon_i8(a, b, n) : dot_scalar_i8(a, b, n);

    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t1);
    for (int r = 0; r < REPS; r++)
        sumidouro += usar_neon ? dot_neon_i8(a, b, n) : dot_scalar_i8(a, b, n);
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t2);

    (void)sumidouro;
    long long ns = (t2.tv_sec - t1.tv_sec) * 1000000000LL + (t2.tv_nsec - t1.tv_nsec);
    return (double)ns / REPS;
}

const char *regiao(int kb)
{
    int ws = 2 * kb;
    if (ws <=   32) return "L1D ";
    if (ws <=  512) return "L2  ";
    if (ws <= 1024) return "L2+ ";
    return                  "DRAM";
}

int main(int argc, char *argv[])
{
    int core_id = 0;
    if (argc >= 2) sscanf(argv[1], "%d", &core_id);

    core_affinity(core_id);

    int8_t *a = aligned_alloc(64, MAX_KB * 1024);
    int8_t *b = aligned_alloc(64, MAX_KB * 1024);
    if (!a || !b) { perror("alloc"); return 1; }

    for (int i = 0; i < MAX_KB * 1024; i++) {
        a[i] = rand();
        b[i] = rand();
    }

    fprintf(stderr, "nucleo=%d  reps=%d  warmup=%d  L1D=32kb  L2=1MB\n\n",
            core_id, REPS, WARMUP);

    printf("klen_kb ; ws_kb ; regiao ; scalar_ns ; neon_ns ; speedup ; macs_scalar ; macs_neon\n");

    for (int kb = 1; kb <= MAX_KB; kb *= 2) {
        int n = kb * 1024;
        double ns_s = medir_media_ns(0, a, b, n);
        double ns_n = medir_media_ns(1, a, b, n);
        double macs_s = n / (ns_s * 1.0e-9);
        double macs_n = n / (ns_n * 1.0e-9);

        printf("%d ; %d ; %s ; %.1f ; %.1f ; %.2f ; %.3e ; %.3e\n",
               kb, 2*kb, regiao(kb),
               ns_s, ns_n, ns_s / ns_n,
               macs_s, macs_n);
    }

    free(a); free(b);
    return 0;
}
