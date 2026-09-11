/*
 * comparador.c
 *
 * Roda as 10.000 imagens de teste do MNIST nas MLPs 784-64-10 em C:
 *   float32, int16, int8 e int8 com NEON.
 *
 * Acuracia: como em testar_acuracia.c (acertos, acuracia, decisoes alteradas,
 *           matriz de confusao).
 * Tempo:    como em homogeneidade_simples.c (warmup antes de cronometrar,
 *           runs execucoes cronometradas juntas com CLOCK_PROCESS_CPUTIME_ID,
 *           divisao pelo numero de runs, varios trials, media e mediana).
 *
 * Mostra no terminal a tabela das 4 versoes e a matriz de confusao comparando
 * int8 e int8 NEON. No final grava dois txt:
 *   resultados/comparacao_float_int16_int8.txt
 *   resultados/comparacao_int8_int8neon.txt
 *
 * Compilar com o compilador ARM (ver Makefile). Uso:
 *   ./comparador [pasta_dados] [pasta_resultados] [execucao] [trials] [runs]
 *   padrao: ../dados ../resultados nativa 3 1
 */

#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "mlp.h"

#define N_IMAGENS 10000
#define N_VERSOES 4
#define REGISTRO  (1 + MLP_ENTRADAS)

enum { FLOAT32, INT16, INT8, INT8NEON };
static const char *NOME[N_VERSOES] = {"float32", "int16", "int8", "int8 NEON"};

static uint8_t registros[N_IMAGENS][REGISTRO];   /* rotulo + 784 pixels */

/* --- uma inferencia: devolve a classe prevista e os logits int8 (para comparar) --- */
static int prever(int versao, const uint8_t *pixels, int32_t *logits_int8)
{
    int k = 0;
    if (versao == FLOAT32) {
        float z[MLP_SAIDAS];
        mlp_float_inferir(pixels, z);
        for (int j = 1; j < MLP_SAIDAS; j++) if (z[j] > z[k]) k = j;
    } else if (versao == INT16) {
        int64_t z[MLP_SAIDAS];
        mlp_int16_inferir(pixels, z);
        for (int j = 1; j < MLP_SAIDAS; j++) if (z[j] > z[k]) k = j;
    } else {
        int32_t z[MLP_SAIDAS];
        if (versao == INT8) mlp_int8_inferir(pixels, z);
        else                mlp_neon_int8_inferir(pixels, z);
        for (int j = 1; j < MLP_SAIDAS; j++) if (z[j] > z[k]) k = j;
        if (logits_int8) memcpy(logits_int8, z, sizeof(z));
    }
    return k;
}

/* uma passagem pelas 10.000 imagens; devolve a soma das classes (sumidouro) */
static int32_t passagem(int versao)
{
    int32_t acc = 0;
    for (int i = 0; i < N_IMAGENS; i++) acc += prever(versao, registros[i] + 1, NULL);
    return acc;
}

/* Afinidade de nucleo: igual a homogeneidade_simples.c. */
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

/* Estatistica simples: media e mediana, iguais a homogeneidade_simples.c. */
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

/* --- impressao em tela e em arquivo ------------------------------------------------- */

static void tabela(FILE *o, const int *versoes, int nv, const int *acertos, const int *dif_float,
                   long long **ns, int trials, int ref_acel)
{
    fprintf(o, "%-10s %8s %9s %10s %16s %14s %12s\n",
            "versao", "acertos", "acuracia", "dif_float", "mediana_us/img", "media_us/img", "aceleracao");
    for (int a = 0; a < nv; a++) {
        int v = versoes[a];
        double med = mediana(ns[v], trials) / 1e3;
        double acel = mediana(ns[ref_acel], trials) / mediana(ns[v], trials);
        fprintf(o, "%-10s %8d %9.4f %10d %16.3f %14.3f %11.2fx\n",
                NOME[v], acertos[v], acertos[v] / (double)N_IMAGENS, dif_float[v],
                med, media(ns[v], trials) / 1e3, acel);
    }
}

static void matriz(FILE *o, const char *titulo, int m[MLP_SAIDAS][MLP_SAIDAS])
{
    fprintf(o, "\nMatriz de confusao - %s (linha = digito real, coluna = digito previsto)\n", titulo);
    fprintf(o, "real |");
    for (int p = 0; p < MLP_SAIDAS; p++) fprintf(o, "%6d", p);
    fprintf(o, "\n-----+------------------------------------------------------------\n");
    for (int r = 0; r < MLP_SAIDAS; r++) {
        fprintf(o, "%4d |", r);
        for (int p = 0; p < MLP_SAIDAS; p++) fprintf(o, "%6d", m[r][p]);
        fprintf(o, "\n");
    }
}

/* matriz com as duas versoes lado a lado em cada celula: int8 / int8 NEON */
static int matriz_comparada(FILE *o, int a[MLP_SAIDAS][MLP_SAIDAS], int b[MLP_SAIDAS][MLP_SAIDAS])
{
    int diferentes = 0;
    fprintf(o, "\nMatriz de confusao comparada: cada celula = int8 / int8 NEON\n");
    fprintf(o, "real |");
    for (int p = 0; p < MLP_SAIDAS; p++) fprintf(o, "%10d", p);
    fprintf(o, "\n-----+----------------------------------------------------------------------------------------------------\n");
    for (int r = 0; r < MLP_SAIDAS; r++) {
        fprintf(o, "%4d |", r);
        for (int p = 0; p < MLP_SAIDAS; p++) {
            fprintf(o, "%5d/%-4d", a[r][p], b[r][p]);
            diferentes += (a[r][p] != b[r][p]);
        }
        fprintf(o, "\n");
    }
    fprintf(o, "celulas diferentes entre int8 e int8 NEON: %d\n", diferentes);
    return diferentes;
}

static void teorica(FILE *o)
{
    long mult = (long)MLP_OCULTA * MLP_ENTRADAS + (long)MLP_SAIDAS * MLP_OCULTA;
    long iter = (long)MLP_OCULTA * (MLP_ENTRADAS / 16) + (long)MLP_SAIDAS * (MLP_OCULTA / 16);
    fprintf(o, "\nACELERACAO NEON TEORICA (ARM, MLP 784-64-10)\n");
    fprintf(o, "  multiplicacoes por imagem (784x64 + 64x10)          : %ld\n", mult);
    fprintf(o, "  int8 em C: 1 multiplicacao por iteracao             : %ld iteracoes\n", mult);
    fprintf(o, "  int8 NEON: 16 multiplicacoes por iteracao (128 bits) : %ld iteracoes\n", iter);
    fprintf(o, "  aceleracao teorica maxima dos produtos              : %ldx\n", mult / iter);
    fprintf(o, "  lei de Amdahl, S = 1/((1-f) + f/16), f = fracao do tempo nos produtos:\n");
    for (int i = 0; i < 3; i++) {
        double f = (i == 0) ? 0.80 : (i == 1) ? 0.90 : 0.95;
        fprintf(o, "    f = %.2f -> S = %.1fx\n", f, 1.0 / ((1.0 - f) + f / 16.0));
    }
}

int main(int argc, char *argv[])
{
    const char *dados  = (argc > 1) ? argv[1] : "../dados";
    const char *saida  = (argc > 2) ? argv[2] : "../resultados";
    const char *execucao = (argc > 3) ? argv[3] : "nativa";
    int trials = (argc > 4) ? atoi(argv[4]) : 3;
    int runs   = (argc > 5) ? atoi(argv[5]) : 1;
    if (trials < 1) trials = 1;
    if (runs < 1) runs = 1;

    char caminho[1024];
    snprintf(caminho, sizeof(caminho), "%s/teste_u8.bin", dados);
    FILE *f = fopen(caminho, "rb");
    if (!f) { perror(caminho); return 1; }
    if (fread(registros, REGISTRO, N_IMAGENS, f) != N_IMAGENS) { fprintf(stderr, "arquivo curto\n"); return 1; }
    fclose(f);

    core_affinity(0);

    /* 1. Acuracia (como testar_acuracia.c) */
    int acertos[N_VERSOES] = {0}, dif_float[N_VERSOES] = {0};
    int m[N_VERSOES][MLP_SAIDAS][MLP_SAIDAS];
    memset(m, 0, sizeof(m));
    int logits_diferentes = 0;

    fprintf(stderr, "Rodando as %d imagens nas 4 versoes...\n", N_IMAGENS);
    for (int i = 0; i < N_IMAGENS; i++) {
        const uint8_t *px = registros[i] + 1;
        int real = registros[i][0];
        int32_t z8[MLP_SAIDAS], zn[MLP_SAIDAS];
        int classe[N_VERSOES];
        classe[FLOAT32]  = prever(FLOAT32, px, NULL);
        classe[INT16]    = prever(INT16, px, NULL);
        classe[INT8]     = prever(INT8, px, z8);
        classe[INT8NEON] = prever(INT8NEON, px, zn);
        for (int v = 0; v < N_VERSOES; v++) {
            if (classe[v] == real) acertos[v]++;
            if (classe[v] != classe[FLOAT32]) dif_float[v]++;
            m[v][real][classe[v]]++;
        }
        if (memcmp(z8, zn, sizeof(z8)) != 0) logits_diferentes++;
    }

    /* 2. Tempo (como homogeneidade_simples.c) */
    long long *ns[N_VERSOES];
    struct timespec t1, t2;
    volatile int32_t sink = 0;
    long long div = (long long)runs * N_IMAGENS;

    for (int v = 0; v < N_VERSOES; v++) ns[v] = malloc((size_t)trials * sizeof(long long));
    fprintf(stderr, "Medindo tempo: trials=%d runs=%d (cada run = %d imagens)...\n", trials, runs, N_IMAGENS);

    for (int t = 0; t < trials; t++) {
        for (int v = 0; v < N_VERSOES; v++) {
            sink += passagem(v);
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t1);
            for (int r = 0; r < runs; r++) sink += passagem(v);
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t2);
            ns[v][t] = ((t2.tv_sec - t1.tv_sec) * 1000000000LL + (t2.tv_nsec - t1.tv_nsec)) / div;
        }
        fprintf(stderr, "  trial %d/%d: float32 %lld | int16 %lld | int8 %lld | int8 NEON %lld ns/img\n",
                t + 1, trials, ns[0][t], ns[1][t], ns[2][t], ns[3][t]);
    }
    (void)sink;

    /* 3. Terminal: tabela das 4 versoes e matriz comparando int8 e int8 NEON */
    const int todas[4] = {FLOAT32, INT16, INT8, INT8NEON};
    const int tres[3]  = {FLOAT32, INT16, INT8};
    const int duas[2]  = {INT8, INT8NEON};

    printf("\n==================================================================================\n");
    printf(" MLP 784-64-10 em C - 10.000 imagens - execucao: %s\n", execucao);
    printf("==================================================================================\n");
    tabela(stdout, todas, 4, acertos, dif_float, ns, trials, INT8);
    printf("(aceleracao = mediana do int8 / mediana da versao)\n");
    int dif_celulas = matriz_comparada(stdout, m[INT8], m[INT8NEON]);
    printf("imagens com logits diferentes entre int8 e int8 NEON : %d\n", logits_diferentes);
    teorica(stdout);

    /* 4. txt: comparacao dos 3 em C */
    snprintf(caminho, sizeof(caminho), "%s/comparacao_float_int16_int8.txt", saida);
    FILE *o = fopen(caminho, "w");
    if (!o) { perror(caminho); return 1; }
    fprintf(o, "COMPARACAO float32 x int16 x int8 - MLP 784-64-10 em C - 10.000 imagens\n");
    fprintf(o, "compilador: gcc %s | execucao: %s | trials=%d runs=%d\n\n", __VERSION__, execucao, trials, runs);
    tabela(o, tres, 3, acertos, dif_float, ns, trials, FLOAT32);
    fprintf(o, "\nLEGENDA\n");
    fprintf(o, "  acertos         imagens classificadas corretamente, de 10000\n");
    fprintf(o, "  acuracia        acertos / 10000\n");
    fprintf(o, "  dif_float       imagens com classe diferente da prevista pelo float32\n");
    fprintf(o, "  mediana/media   tempo por imagem em microssegundos, sobre %d trials\n", trials);
    fprintf(o, "  aceleracao      mediana do float32 / mediana da versao\n");
    fprintf(o, "  execucao        nativa = na placa | emulada_qemu = QEMU no PC (tempo nao representa a placa)\n");
    for (int a = 0; a < 3; a++) matriz(o, NOME[tres[a]], m[tres[a]]);
    fclose(o);
    printf("\nGravado: %s\n", caminho);

    /* 5. txt: int8 x int8 NEON */
    snprintf(caminho, sizeof(caminho), "%s/comparacao_int8_int8neon.txt", saida);
    o = fopen(caminho, "w");
    if (!o) { perror(caminho); return 1; }
    fprintf(o, "COMPARACAO int8 x int8 NEON - MLP 784-64-10 em C - 10.000 imagens\n");
    fprintf(o, "compilador ARM: gcc %s | execucao: %s | trials=%d runs=%d\n\n", __VERSION__, execucao, trials, runs);
    tabela(o, duas, 2, acertos, dif_float, ns, trials, INT8);
    fprintf(o, "\nimagens com logits diferentes entre int8 e int8 NEON : %d\n", logits_diferentes);
    matriz_comparada(o, m[INT8], m[INT8NEON]);
    teorica(o);
    fprintf(o, "\nLEGENDA\n");
    fprintf(o, "  int8            mlp_int8.c: produtos escalares em laco C, sem vetorizacao\n");
    fprintf(o, "  int8 NEON       mlp_neon_int8.c: produtos com intrinsics NEON (vmull_s8 + vpadalq_s16)\n");
    fprintf(o, "  aceleracao      mediana do int8 / mediana da versao (medida nesta execucao)\n");
    fprintf(o, "  logits          os 10 valores de saida; iguais = mesma conta, bit a bit\n");
    fprintf(o, "  celula a / b    quantidade no int8 / quantidade no int8 NEON\n");
    fprintf(o, "  execucao        nativa = na placa | emulada_qemu = QEMU no PC (tempo nao representa a placa)\n");
    fclose(o);
    printf("Gravado: %s\n", caminho);

    int ok = (logits_diferentes == 0 && dif_celulas == 0);
    printf("\nRESULTADO: %s\n", ok ? "OK - int8 NEON identico ao int8 nas 10.000 imagens"
                                   : "FALHA - int8 NEON diferente do int8");

    for (int v = 0; v < N_VERSOES; v++) free(ns[v]);
    return ok ? 0 : 1;
}
