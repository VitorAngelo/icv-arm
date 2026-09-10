/*
 * bench_mlp_int8_neon.c - Benchmark da inferencia int8: escalar x autovetorizado x NEON.
 *
 * METODOLOGIA (a mesma de experimentos_neon_2/homogeneidade_v2.c, definida
 * pelo professor):
 *   - thread fixada em um nucleo (pthread_setaffinity_np);
 *   - warmup antes de cada bloco cronometrado;
 *   - cada medicao = RUNS execucoes cronometradas juntas com
 *     CLOCK_PROCESS_CPUTIME_ID, depois dividido pelo numero de execucoes;
 *   - TRIALS repeticoes, com media, mediana e minimo.
 * Adaptacoes para a MLP:
 *   - uma execucao = inferencia de IMAGENS imagens de teste; o tempo reportado
 *     e por imagem (ns / (RUNS * IMAGENS));
 *   - todas as imagens sao carregadas na RAM antes de medir (sem I/O no tempo);
 *   - as tres variantes usam exatamente as mesmas imagens;
 *   - a ordem das variantes gira a cada trial, para que nenhuma se beneficie
 *     sistematicamente do cache aquecido pela anterior (ordem_cache.c);
 *   - antes de medir, confere que as tres variantes produzem logits identicos;
 *     durante a medicao, uma soma de verificacao das classes confirma que o
 *     trabalho cronometrado e o mesmo.
 *
 * VARIANTES:
 *   escalar  c_modular/mlp_int8.c com -fno-tree-vectorize (sem NEON)
 *   autovet  c_modular/mlp_int8.c com -O2 (vetorizacao automatica do GCC)
 *   neon     c_neon/mlp_int8_neon.c (kernel dot_i8_neon, intrinsics manuais)
 *
 * Uso:
 *   ./bench_mlp_int8_neon [nucleo] [imagens] [runs] [trials] [pasta_dados]
 *   padrao: nucleo 0, imagens 10000, runs 3, trials 21, pasta ../modelo/exportado
 *
 * Saida: CSV por trial em stdout; parametros e resumo em stderr.
 * No QEMU os tempos sao de emulacao: servem so para conferir o funcionamento.
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
#include "mlp_int8_neon.h"
#include "mlp_int8_variantes.h"

#define N_VARIANTES 3
#define BYTES_REGISTRO (1 + MLP_ENTRADAS)

typedef void (*FuncaoInferencia)(const uint8_t *pixels, int32_t *logits);

typedef struct {
    const char *nome;
    FuncaoInferencia inferir;
} Variante;

static const Variante VARIANTES[N_VARIANTES] = {
    {"escalar", mlp_int8_escalar_inferir},
    {"autovet", mlp_int8_autovet_inferir},
    {"neon",    mlp_int8_neon_inferir},
};

/* --- utilitarios ------------------------------------------------------------ */

/* Afinidade de nucleo: igual a homogeneidade_v2.c. */
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

static long long diferenca_ns(const struct timespec *t1, const struct timespec *t2)
{
    return (t2->tv_sec - t1->tv_sec) * 1000000000LL + (t2->tv_nsec - t1->tv_nsec);
}

static int argmax(const int32_t *z)
{
    int k = 0;
    for (int j = 1; j < MLP_SAIDAS; j++) {
        if (z[j] > z[k]) k = j;
    }
    return k;
}

static int cmp_double(const void *x, const void *y)
{
    double a = *(const double *)x, b = *(const double *)y;
    return (a > b) - (a < b);
}

static void estatisticas(const double *v, int n, double *media, double *mediana, double *minimo)
{
    double *tmp = malloc((size_t)n * sizeof(double));
    memcpy(tmp, v, (size_t)n * sizeof(double));
    qsort(tmp, (size_t)n, sizeof(double), cmp_double);
    double s = 0.0;
    for (int i = 0; i < n; i++) s += v[i];
    *media = s / n;
    *mediana = (n % 2) ? tmp[n / 2] : 0.5 * (tmp[n / 2 - 1] + tmp[n / 2]);
    *minimo = tmp[0];
    free(tmp);
}

/* Le a primeira linha de um arquivo do sysfs (governador, frequencia, temperatura). */
static void ler_sysfs(const char *caminho, char *destino, size_t tam)
{
    FILE *f = fopen(caminho, "r");
    if (f == NULL || fgets(destino, (int)tam, f) == NULL) {
        snprintf(destino, tam, "indisponivel");
    } else {
        destino[strcspn(destino, "\n")] = '\0';
    }
    if (f) fclose(f);
}

static void registrar_ambiente(int nucleo, const char *momento)
{
    char caminho[256], gov[64], freq[64], temp[64];
    snprintf(caminho, sizeof(caminho), "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_governor", nucleo);
    ler_sysfs(caminho, gov, sizeof(gov));
    snprintf(caminho, sizeof(caminho), "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq", nucleo);
    ler_sysfs(caminho, freq, sizeof(freq));
    ler_sysfs("/sys/class/thermal/thermal_zone0/temp", temp, sizeof(temp));
    fprintf(stderr, "ambiente (%s): governador=%s  freq_khz=%s  temp_milicelsius=%s\n", momento, gov, freq, temp);
}

/* --- medicao ------------------------------------------------------------------ */

/* Executa RUNS passagens sobre as imagens; devolve ns por imagem e a soma de verificacao. */
static double medir_ns_por_imagem(FuncaoInferencia inferir, const uint8_t *pixels, int imagens, int runs,
                                  uint32_t *verificacao)
{
    struct timespec t1, t2;
    int32_t logits[MLP_SAIDAS];
    uint32_t soma = 0;

    inferir(pixels, logits);   /* warmup, como em homogeneidade_v2.c */

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t1);
    for (int r = 0; r < runs; r++) {
        for (int i = 0; i < imagens; i++) {
            inferir(&pixels[(size_t)i * MLP_ENTRADAS], logits);
            soma = soma * 31u + (uint32_t)argmax(logits);
        }
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t2);

    *verificacao = soma;
    return (double)diferenca_ns(&t1, &t2) / ((double)runs * (double)imagens);
}

int main(int argc, char *argv[])
{
    int nucleo  = (argc > 1) ? atoi(argv[1]) : 0;
    int imagens = (argc > 2) ? atoi(argv[2]) : 10000;
    int runs    = (argc > 3) ? atoi(argv[3]) : 3;
    int trials  = (argc > 4) ? atoi(argv[4]) : 21;
    const char *dir = (argc > 5) ? argv[5] : "../modelo/exportado";
    if (imagens < 1 || runs < 1 || trials < 1) {
        fprintf(stderr, "Uso: %s [nucleo] [imagens] [runs] [trials] [pasta_dados]\n", argv[0]);
        return 2;
    }

    /* 1. Carrega as imagens na RAM (fora de qualquer medicao). */
    char caminho[4096];
    snprintf(caminho, sizeof(caminho), "%s/teste_u8.bin", dir);
    FILE *f = fopen(caminho, "rb");
    if (f == NULL) {
        fprintf(stderr, "ERRO: nao foi possivel abrir %s\n", caminho);
        return 1;
    }
    uint8_t *pixels = malloc((size_t)imagens * MLP_ENTRADAS);
    uint8_t *rotulos = malloc((size_t)imagens);
    if (pixels == NULL || rotulos == NULL) {
        fprintf(stderr, "ERRO: memoria insuficiente\n");
        return 1;
    }
    uint8_t registro[BYTES_REGISTRO];
    for (int i = 0; i < imagens; i++) {
        if (fread(registro, 1, BYTES_REGISTRO, f) != BYTES_REGISTRO) {
            fprintf(stderr, "ERRO: o arquivo tem menos de %d imagens\n", imagens);
            return 1;
        }
        rotulos[i] = registro[0];
        memcpy(&pixels[(size_t)i * MLP_ENTRADAS], &registro[1], MLP_ENTRADAS);
    }
    fclose(f);

    core_affinity(nucleo);

    fprintf(stderr, "==================================================================\n");
    fprintf(stderr, " Benchmark MLP int8 784->64->10: escalar x autovet x neon\n");
    fprintf(stderr, "==================================================================\n");
    fprintf(stderr, "nucleo=%d  imagens=%d  runs=%d  trials=%d\n", nucleo, imagens, runs, trials);
    fprintf(stderr, "relogio=CLOCK_PROCESS_CPUTIME_ID  compilador=gcc %s\n", __VERSION__);
    registrar_ambiente(nucleo, "inicio");

    /* 2. Portao de correcao: as tres variantes devem gerar logits identicos. */
    int acertos[N_VARIANTES] = {0};
    for (int i = 0; i < imagens; i++) {
        int32_t z[N_VARIANTES][MLP_SAIDAS];
        const uint8_t *px = &pixels[(size_t)i * MLP_ENTRADAS];
        for (int v = 0; v < N_VARIANTES; v++) {
            VARIANTES[v].inferir(px, z[v]);
            acertos[v] += (argmax(z[v]) == rotulos[i]);
        }
        if (memcmp(z[0], z[1], sizeof(z[0])) != 0 || memcmp(z[0], z[2], sizeof(z[0])) != 0) {
            fprintf(stderr, "ERRO: logits diferentes entre variantes na imagem %d. Benchmark cancelado.\n", i);
            return 1;
        }
    }
    fprintf(stderr, "correcao: logits identicos nas 3 variantes em %d imagens; acuracia %.4f (%d/%d)\n",
            imagens, (double)acertos[2] / imagens, acertos[2], imagens);

    /* 3. Warmup global: uma passagem completa por variante. */
    for (int v = 0; v < N_VARIANTES; v++) {
        uint32_t lixo;
        medir_ns_por_imagem(VARIANTES[v].inferir, pixels, imagens, 1, &lixo);
    }

    /* 4. Trials. */
    double *tempos[N_VARIANTES];
    for (int v = 0; v < N_VARIANTES; v++) {
        tempos[v] = malloc((size_t)trials * sizeof(double));
        if (tempos[v] == NULL) {
            fprintf(stderr, "ERRO: memoria insuficiente\n");
            return 1;
        }
    }
    uint32_t verificacao_referencia = 0;
    int verificacao_ok = 1;

    printf("trial,ordem,escalar_ns_img,autovet_ns_img,neon_ns_img\n");
    for (int t = 0; t < trials; t++) {
        char ordem[8] = {0};
        for (int p = 0; p < N_VARIANTES; p++) {
            int v = (t + p) % N_VARIANTES;
            uint32_t verificacao;
            tempos[v][t] = medir_ns_por_imagem(VARIANTES[v].inferir, pixels, imagens, runs, &verificacao);
            if (t == 0 && p == 0) verificacao_referencia = verificacao;
            if (verificacao != verificacao_referencia) verificacao_ok = 0;
            ordem[p] = VARIANTES[v].nome[0];
        }
        printf("%d,%s,%.1f,%.1f,%.1f\n", t, ordem, tempos[0][t], tempos[1][t], tempos[2][t]);
    }

    registrar_ambiente(nucleo, "fim");

    /* 5. Resumo. */
    double med[N_VARIANTES], mediana[N_VARIANTES], minimo[N_VARIANTES];
    fprintf(stderr, "\n%-8s %14s %14s %14s   (tempo por imagem, microssegundos)\n", "variante", "media", "mediana", "minimo");
    for (int v = 0; v < N_VARIANTES; v++) {
        estatisticas(tempos[v], trials, &med[v], &mediana[v], &minimo[v]);
        fprintf(stderr, "%-8s %14.3f %14.3f %14.3f\n", VARIANTES[v].nome, med[v] / 1e3, mediana[v] / 1e3, minimo[v] / 1e3);
    }
    fprintf(stderr, "\nspeedup pela mediana:\n");
    fprintf(stderr, "  escalar / neon    = %.2fx   (NEON manual contra escalar puro)\n", mediana[0] / mediana[2]);
    fprintf(stderr, "  autovet / neon    = %.2fx   (NEON manual contra vetorizacao automatica do GCC)\n", mediana[1] / mediana[2]);
    fprintf(stderr, "  escalar / autovet = %.2fx   (ganho da vetorizacao automatica)\n", mediana[0] / mediana[1]);
    fprintf(stderr, "soma de verificacao das classes identica em todas as medicoes: %s\n", verificacao_ok ? "sim" : "NAO");

    for (int v = 0; v < N_VARIANTES; v++) free(tempos[v]);
    free(pixels);
    free(rotulos);
    return verificacao_ok ? 0 : 1;
}
