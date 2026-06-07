/*
 * homogeneidade_v2.c
 *
 * Mede se os 4 nucleos Cortex-A72 do Pi 4 sao homogeneos na pratica,
 * separando o efeito de NUCLEO do efeito de POSICAO na sequencia de medicao.
 *
 * Correcoes em relacao a versao anterior:
 *   - A ordem dos nucleos e sorteada a cada trial (counterbalancing completo
 *     das 24 permutacoes), entao efeito de tempo / temperatura / rampa de
 *     frequencia nao fica confundido com efeito de nucleo.
 *   - 10000 trials => 10000 amostras por nucleo => mediana, desvio, percentis.
 *   - Cache despejado antes de cada medicao (metodo do professor): cada amostra
 *     comeca fria. Isso torna a medicao limitada por memoria, nao por compute,
 *     mas com a ordem sorteada e 10000 trials o efeito do flush se distribui
 *     igualmente entre nucleos e posicoes, mantendo a comparacao relativa justa.
 *
 * Nao trava frequencia de proposito: a ordem sorteada distribui qualquer rampa
 * de DVFS ou deriva termica igualmente entre nucleos e posicoes, entao o spread
 * entre nucleos nao fica enviesado. O custo e variancia um pouco maior, sem vies.
 *
 * Compilar:
 *   gcc -O2 -march=armv8-a+simd homogeneidade_v2.c -lpthread -lm -o homogeneidade_v2
 *
 * Uso:
 *   ./homogeneidade_v2 [tamanho_kb] [trials] [seed] > dados.csv
 *   ./homogeneidade_v2 16 10000 1 > dados.csv
 *
 * stdout : CSV (trial,position,core,scalar_ns,neon_ns) para o grafico
 * stderr : resumo estatistico por nucleo e por posicao + veredito
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
#define NPERM  24                    /* 4! */
#define FLUSH_SZ (2 * 1024 * 1024)   /* maior que o L2 (1 MB) do Pi 4 */

/* ---------- kernels ---------- */

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

/* ---------- permutacoes (Heap) ---------- */

static int perms[NPERM][NCORES];
static int nperm = 0;

static void gen_perms(int *arr, int k)
{
    if (k == 1) {
        memcpy(perms[nperm++], arr, NCORES * sizeof(int));
        return;
    }
    for (int i = 0; i < k; i++) {
        gen_perms(arr, k - 1);
        if (k % 2 == 0) { int t = arr[i];   arr[i]   = arr[k-1]; arr[k-1] = t; }
        else            { int t = arr[0];   arr[0]   = arr[k-1]; arr[k-1] = t; }
    }
}

/* ---------- estatistica ---------- */

typedef struct { double mean, sd, median, p05, p95; long long min; } Stat;

static int cmpll(const void *x, const void *y)
{
    long long a = *(const long long *)x, b = *(const long long *)y;
    return (a > b) - (a < b);
}

static Stat compute(const long long *src, int n)
{
    long long *t = malloc((size_t)n * sizeof(long long));
    memcpy(t, src, (size_t)n * sizeof(long long));

    double s = 0.0;
    for (int i = 0; i < n; i++) s += (double)t[i];
    double m = s / n;
    double v = 0.0;
    for (int i = 0; i < n; i++) { double d = (double)t[i] - m; v += d * d; }

    qsort(t, n, sizeof(long long), cmpll);

    Stat st;
    st.mean   = m;
    st.sd     = sqrt(v / n);
    st.min    = t[0];
    st.median = (double)t[n / 2];
    st.p05    = (double)t[(int)(0.05 * n)];
    st.p95    = (double)t[(int)(0.95 * n)];
    free(t);
    return st;
}

/* ---------- grafico ascii ---------- */

static void barra(const char *rotulo, double val, double maxv, int width)
{
    int n = (maxv > 0.0) ? (int)(val / maxv * width + 0.5) : 0;
    fprintf(stderr, "  %-6s |", rotulo);
    for (int i = 0; i < n; i++)      fputc('#', stderr);
    for (int i = n; i < width; i++)  fputc(' ', stderr);
    fprintf(stderr, "| %.0f ns\n", val);
}

int main(int argc, char *argv[])
{
    int kb     = (argc > 1) ? atoi(argv[1]) : 16;
    int trials = (argc > 2) ? atoi(argv[2]) : 10000;
    int seed   = (argc > 3) ? atoi(argv[3]) : 1;
    int klen   = kb * 1024;                 /* multiplo de 16 e de 64 */

    srand((unsigned)seed);

    /* gera as 24 permutacoes */
    int base[NCORES] = {0, 1, 2, 3};
    gen_perms(base, NCORES);

    /* schedule balanceado: cada permutacao ~trials/24 vezes, depois embaralha */
    int *permidx = malloc((size_t)trials * sizeof(int));
    for (int t = 0; t < trials; t++) permidx[t] = t % NPERM;
    for (int t = trials - 1; t > 0; t--) {      /* Fisher-Yates */
        int j = rand() % (t + 1);
        int tmp = permidx[t]; permidx[t] = permidx[j]; permidx[j] = tmp;
    }

    /* amostras: cada nucleo e cada posicao aparecem 1x por trial => trials amostras */
    long long *neon_core[NCORES], *scal_core[NCORES];
    long long *neon_pos[NCORES],  *scal_pos[NCORES];
    for (int i = 0; i < NCORES; i++) {
        neon_core[i] = malloc((size_t)trials * sizeof(long long));
        scal_core[i] = malloc((size_t)trials * sizeof(long long));
        neon_pos[i]  = malloc((size_t)trials * sizeof(long long));
        scal_pos[i]  = malloc((size_t)trials * sizeof(long long));
    }

    int8_t *a = aligned_alloc(64, klen);
    int8_t *b = aligned_alloc(64, klen);
    if (!a || !b) { perror("alloc"); return 1; }
    for (int i = 0; i < klen; i++) { a[i] = rand(); b[i] = rand(); }

    fprintf(stderr,
        "kb=%d  klen=%d  working_set=%dkb  L1D=32kb  trials=%d  perms=%d (~%d cada)\n",
        kb, klen, (2 * klen) / 1024, trials, NPERM, trials / NPERM);
    fprintf(stderr, "metodo: flush de cache antes de cada medicao (frio).\n\n");

    printf("trial,position,core,scalar_ns,neon_ns\n");

    struct timespec t1, t2, t3;
    volatile int32_t sink = 0;

    for (int t = 0; t < trials; t++) {
        int *order = perms[permidx[t]];
        for (int pos = 0; pos < NCORES; pos++) {
            int core = order[pos];
            core_affinity(core);
            despejar_cache();      /* metodo do professor: mede frio */

            clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t1);
            sink += dot_scalar_i8(a, b, klen);
            clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t2);
            sink += dot_neon_i8(a, b, klen);
            clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t3);

            long long ns_s = (t2.tv_sec - t1.tv_sec) * 1000000000LL + (t2.tv_nsec - t1.tv_nsec);
            long long ns_n = (t3.tv_sec - t2.tv_sec) * 1000000000LL + (t3.tv_nsec - t2.tv_nsec);

            scal_core[core][t] = ns_s;   neon_core[core][t] = ns_n;
            scal_pos[pos][t]   = ns_s;   neon_pos[pos][t]   = ns_n;

            printf("%d,%d,%d,%lld,%lld\n", t, pos, core, ns_s, ns_n);
        }
    }
    (void)sink;

    /* ---------- resumo ---------- */
    Stat sc[NCORES], sp[NCORES];
    for (int c = 0; c < NCORES; c++) sc[c] = compute(neon_core[c], trials);
    for (int p = 0; p < NCORES; p++) sp[p] = compute(neon_pos[p], trials);

    fprintf(stderr, "POR NUCLEO (NEON, ns)\n");
    fprintf(stderr, "core ; mean ; median ; sd ; min ; p05 ; p95\n");
    for (int c = 0; c < NCORES; c++)
        fprintf(stderr, "%d ; %.0f ; %.0f ; %.0f ; %lld ; %.0f ; %.0f\n",
                c, sc[c].mean, sc[c].median, sc[c].sd, sc[c].min, sc[c].p05, sc[c].p95);

    fprintf(stderr, "\nPOR POSICAO DE MEDICAO (NEON, ns)  [1a, 2a, 3a, 4a medida do trial]\n");
    fprintf(stderr, "pos ; mean ; median ; sd ; min ; p05 ; p95\n");
    for (int p = 0; p < NCORES; p++)
        fprintf(stderr, "%d ; %.0f ; %.0f ; %.0f ; %lld ; %.0f ; %.0f\n",
                p, sp[p].mean, sp[p].median, sp[p].sd, sp[p].min, sp[p].p05, sp[p].p95);

    /* spread sobre a mediana (robusta a outlier de escalonador) */
    double cmin = sc[0].median, cmax = sc[0].median;
    double pmin = sp[0].median, pmax = sp[0].median;
    for (int i = 1; i < NCORES; i++) {
        if (sc[i].median < cmin) cmin = sc[i].median;
        if (sc[i].median > cmax) cmax = sc[i].median;
        if (sp[i].median < pmin) pmin = sp[i].median;
        if (sp[i].median > pmax) pmax = sp[i].median;
    }
    double spread_core = (cmax - cmin) / cmin * 100.0;
    double spread_pos  = (pmax - pmin) / pmin * 100.0;

    /* ---------- grafico ascii (mediana) ---------- */
    double maxv = (cmax > pmax) ? cmax : pmax;   /* mesma escala nos dois */
    char rot[8];

    fprintf(stderr, "\nNEON por nucleo (mediana)\n");
    for (int c = 0; c < NCORES; c++) {
        snprintf(rot, sizeof(rot), "core%d", c);
        barra(rot, sc[c].median, maxv, 50);
    }

    fprintf(stderr, "\nNEON por posicao de medicao (mediana)\n");
    for (int p = 0; p < NCORES; p++) {
        snprintf(rot, sizeof(rot), "%da", p + 1);
        barra(rot, sp[p].median, maxv, 50);
    }

    fprintf(stderr, "\nVEREDITO (NEON, sobre mediana)\n");
    fprintf(stderr, "  spread entre nucleos  : %.1f%%\n", spread_core);
    fprintf(stderr, "  spread entre posicoes : %.1f%%\n", spread_pos);
    if (spread_pos > spread_core)
        fprintf(stderr, "  -> a variacao segue a POSICAO, nao o nucleo. Assimetria e artefato de ordem.\n");
    else
        fprintf(stderr, "  -> a variacao segue o NUCLEO mesmo com ordem sorteada. Assimetria pode ser real.\n");

    for (int i = 0; i < NCORES; i++) {
        free(neon_core[i]); free(scal_core[i]);
        free(neon_pos[i]);  free(scal_pos[i]);
    }
    free(permidx); free(a); free(b);
    return 0;
}
