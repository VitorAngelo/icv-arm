/*
 * orquestrador.c - Validacao, metricas e medicao dos modulos de inferencia.
 *
 * Le o conjunto de teste em streaming (um registro de 785 bytes por vez),
 * executa os modulos escolhidos e:
 *   - compara os logits, bit a bit, com a referencia gerada pelo Python;
 *   - compara cada decisao com a do modelo Python original (float64, que
 *     reproduz o predict do scikit-learn);
 *   - calcula acuracia, matriz de confusao, precisao, revocacao e F1;
 *   - aplica o teste de McNemar exato para dizer se diferencas de acuracia
 *     sao compativeis com o acaso;
 *   - mede apenas o tempo de inferencia (a leitura de arquivos fica fora).
 *
 * Uso:
 *   ./orquestrador <modulo> [n_amostras] [dir_dados]
 *     modulo      float | int16 | int8 | todos (apenas os modulos compilados)
 *     n_amostras  quantidade de imagens a processar (padrao: todas)
 *     dir_dados   pasta com teste_u8.bin e ref_*.bin (padrao: ../modelo/exportado)
 *
 * Codigo de saida: 0 somente se todos os logits coincidirem com a referencia.
 * math.h e usado apenas aqui, para a estatistica; os modulos int nao usam float.
 *
 * ARQUIVOS LIDOS (gerados por python/exporta_modelo.py em modelo/exportado/):
 *   teste_u8.bin         10.000 registros [rotulo][784 pixels], lidos um por vez
 *   ref_<modulo>.bin     logits que o modulo C deve produzir, bit a bit
 *   ref_python_pred.bin  classe prevista pelo modelo Python original (scikit-learn)
 *
 * COMO LER O RELATORIO:
 *   Logits diferentes da referencia  deve ser 0: prova que o C calcula
 *                                    exatamente o que o Python especificou.
 *   Matriz de confusao               celula [real][previsto] conta quantas
 *                                    imagens do digito "real" foram classificadas
 *                                    como "previsto"; a diagonal sao os acertos.
 *   Precisao da classe k             dos que o modelo chamou de k, fracao correta.
 *   Revocacao da classe k            dos que eram k, fracao reconhecida.
 *   F1                               media harmonica de precisao e revocacao.
 *   Python errava, modulo acerta     amostras corrigidas pelo modulo.
 *   Python acertava, modulo erra     erros introduzidos pelo modulo.
 *   p-valor McNemar                  probabilidade de um desequilibrio igual ou
 *                                    maior entre as duas linhas acima se os dois
 *                                    classificadores tivessem a mesma taxa de erro.
 *                                    p < 0,05: diferenca de acuracia dificilmente e acaso.
 */
#define _POSIX_C_SOURCE 199309L

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "mlp.h"

#define DIR_PADRAO     "../modelo/exportado"
#define ARQUIVO_TESTE  "teste_u8.bin"
#define ARQUIVO_PYTHON "ref_python_pred.bin"
#define TAM_CAMINHO    4096
#define MAX_MODULOS    3
#define ALFA           0.05

/* Uniao para armazenar logits de qualquer tipo (float32, int32 ou int64) no mesmo
 * espaco, sem violar as regras de aliasing do C ao comparar bytes com memcmp. */
typedef union {
    float   f32[MLP_SAIDAS];
    int32_t i32[MLP_SAIDAS];
    int64_t i64[MLP_SAIDAS];
} Logits;

/* Descricao de um modulo: como chama-lo e onde esta sua referencia. */
typedef struct {
    const char *nome;
    const char *descricao;
    const char *arquivo_ref;
    size_t bytes_logits;
    void (*inferir)(const uint8_t *pixels, Logits *z);
    int (*argmax)(const Logits *z);
    void (*imprimir)(const char *rotulo, const Logits *z);
} Modulo;

/* Tudo o que se mede em uma execucao de um modulo. */
typedef struct {
    long amostras;
    long acertos;
    long logits_diferentes;
    long decisoes_diferentes_ref;
    long confusao[MLP_SAIDAS][MLP_SAIDAS];   /* [real][previsto] */
    double segundos;
    double seg_min;
    double seg_max;
    uint8_t *pred;                           /* classe prevista por amostra */
} Resultado;

typedef struct {
    long n;               /* amostras avaliadas */
    uint8_t *rotulo;      /* classe real por amostra */
    Resultado python;     /* modelo Python original */
} Base;

typedef struct {
    long discordantes;    /* decisoes diferentes entre A e B */
    long b_corrige;       /* A erra e B acerta */
    long b_erra_novo;     /* A acerta e B erra */
    long ambos_errados;   /* ambos erram, com classes diferentes */
    double p;             /* McNemar exato bilateral */
} Pareado;

/* ------------------------------------------------------------------------ */
/* Adaptadores por tipo                                                     */
/* ------------------------------------------------------------------------ */
static void inferir_float(const uint8_t *p, Logits *z) { mlp_float_inferir(p, z->f32); }

static int argmax_f32(const Logits *z)
{
    int k = 0;
    for (int j = 1; j < MLP_SAIDAS; j++) {
        if (z->f32[j] > z->f32[k]) k = j;
    }
    return k;
}

static void imprimir_float(const char *rotulo, const Logits *z)
{
    printf("    %-8s", rotulo);
    for (int j = 0; j < MLP_SAIDAS; j++) printf(" %.9g", (double)z->f32[j]);
    printf("\n");
}

#ifdef MLP_COM_INT16
static void inferir_int16(const uint8_t *p, Logits *z) { mlp_int16_inferir(p, z->i64); }

static int argmax_i64(const Logits *z)
{
    int k = 0;
    for (int j = 1; j < MLP_SAIDAS; j++) {
        if (z->i64[j] > z->i64[k]) k = j;
    }
    return k;
}

static void imprimir_int16(const char *rotulo, const Logits *z)
{
    printf("    %-8s", rotulo);
    for (int j = 0; j < MLP_SAIDAS; j++) printf(" %" PRId64, z->i64[j]);
    printf("\n");
}
#endif

#ifdef MLP_COM_INT8
static void inferir_int8(const uint8_t *p, Logits *z) { mlp_int8_inferir(p, z->i32); }

static int argmax_i32(const Logits *z)
{
    int k = 0;
    for (int j = 1; j < MLP_SAIDAS; j++) {
        if (z->i32[j] > z->i32[k]) k = j;
    }
    return k;
}

static void imprimir_int8(const char *rotulo, const Logits *z)
{
    printf("    %-8s", rotulo);
    for (int j = 0; j < MLP_SAIDAS; j++) printf(" %" PRId32, z->i32[j]);
    printf("\n");
}
#endif

static const Modulo MODULOS[] = {
    {"float", "float32 (pesos originais)", "ref_float32.bin",
     sizeof(float) * MLP_SAIDAS, inferir_float, argmax_f32, imprimir_float},
#ifdef MLP_COM_INT16
    {"int16", "int16 (produto 32 bits, acumulacao 64 bits)", "ref_int16.bin",
     sizeof(int64_t) * MLP_SAIDAS, inferir_int16, argmax_i64, imprimir_int16},
#endif
#ifdef MLP_COM_INT8
    {"int8", "int8 (produto 16 bits, acumulacao 32 bits)", "ref_int8.bin",
     sizeof(int32_t) * MLP_SAIDAS, inferir_int8, argmax_i32, imprimir_int8},
#endif
};

#define N_MODULOS ((int)(sizeof(MODULOS) / sizeof(MODULOS[0])))

/* ------------------------------------------------------------------------ */
/* Utilitarios                                                              */
/* ------------------------------------------------------------------------ */
static double agora_segundos(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static int sistema_little_endian(void)
{
    const uint16_t teste = 1;
    return *(const uint8_t *)&teste == 1;
}

static void listar_modulos(void)
{
    fprintf(stderr, "Modulos compilados:");
    for (int m = 0; m < N_MODULOS; m++) fprintf(stderr, " %s", MODULOS[m].nome);
    fprintf(stderr, "\n");
}

static FILE *abrir(const char *dir, const char *arquivo)
{
    char caminho[TAM_CAMINHO];
    int n = snprintf(caminho, TAM_CAMINHO, "%s/%s", dir, arquivo);
    if (n <= 0 || n >= TAM_CAMINHO) {
        fprintf(stderr, "ERRO: caminho longo demais para %s.\n", arquivo);
        return NULL;
    }
    FILE *f = fopen(caminho, "rb");
    if (f == NULL) fprintf(stderr, "ERRO: nao foi possivel abrir %s\n", caminho);
    return f;
}

static long tamanho_arquivo(FILE *f)
{
    if (fseek(f, 0, SEEK_END) != 0) return -1;
    long t = ftell(f);
    if (fseek(f, 0, SEEK_SET) != 0) return -1;
    return t;
}

static double razao(long num, long den)
{
    return (den > 0) ? (double)num / (double)den : 0.0;
}

/*
 * Teste de McNemar exato (comparacao pareada de dois classificadores no mesmo teste).
 *   b = amostras em que so o classificador B acerta
 *   c = amostras em que so o classificador A acerta
 * As amostras em que ambos acertam ou ambos erram nao informam qual e melhor.
 * Se A e B tivessem a mesma taxa de erro, cada uma das b + c amostras
 * discordantes cairia em b ou c com probabilidade 1/2 (moeda justa).
 * p-valor bilateral = 2 * P(X <= min(b, c)), X ~ Binomial(b + c, 1/2).
 * As combinacoes sao calculadas em logaritmo (lgamma) para nao estourar.
 */
static double mcnemar_exato(long b, long c)
{
    long n = b + c;
    if (n == 0) return 1.0;
    long k = (b < c) ? b : c;
    double soma = 0.0;
    for (long i = 0; i <= k; i++) {
        double log_termo = lgamma((double)n + 1.0) - lgamma((double)i + 1.0) -
                           lgamma((double)(n - i) + 1.0) - (double)n * log(2.0);
        soma += exp(log_termo);
    }
    double p = 2.0 * soma;
    return (p > 1.0) ? 1.0 : p;
}

static Pareado comparar(const uint8_t *pred_a, const uint8_t *pred_b, const uint8_t *rotulo, long n)
{
    Pareado c = {0, 0, 0, 0, 1.0};
    for (long i = 0; i < n; i++) {
        int a_ok = (pred_a[i] == rotulo[i]);
        int b_ok = (pred_b[i] == rotulo[i]);
        if (pred_a[i] != pred_b[i]) c.discordantes++;
        if (!a_ok && b_ok) c.b_corrige++;
        if (a_ok && !b_ok) c.b_erra_novo++;
        if (!a_ok && !b_ok && pred_a[i] != pred_b[i]) c.ambos_errados++;
    }
    c.p = mcnemar_exato(c.b_corrige, c.b_erra_novo);
    return c;
}

/* ------------------------------------------------------------------------ */
/* Base: rotulos e modelo Python original                                   */
/* ------------------------------------------------------------------------ */
/* Le os rotulos reais e as previsoes do modelo Python original, e monta a
 * matriz de confusao do Python para servir de base de comparacao. */
static int carregar_base(const char *dir, long n_max, Base *base)
{
    memset(base, 0, sizeof(*base));
    FILE *f_teste = abrir(dir, ARQUIVO_TESTE);
    FILE *f_py = abrir(dir, ARQUIVO_PYTHON);
    int status = 0;
    if (f_teste == NULL || f_py == NULL) {
        status = -1;
        goto fim;
    }

    long bytes_teste = tamanho_arquivo(f_teste);
    long bytes_py = tamanho_arquivo(f_py);
    if (bytes_teste <= 0 || bytes_teste % MLP_BYTES_REGISTRO != 0) {
        fprintf(stderr, "ERRO: %s nao contem registros de %d bytes.\n", ARQUIVO_TESTE, MLP_BYTES_REGISTRO);
        status = -1;
        goto fim;
    }
    long total = bytes_teste / MLP_BYTES_REGISTRO;
    if (bytes_py != total) {
        fprintf(stderr, "ERRO: %s tem %ld bytes; esperado %ld.\n", ARQUIVO_PYTHON, bytes_py, total);
        status = -1;
        goto fim;
    }
    base->n = (n_max > 0 && n_max < total) ? n_max : total;
    base->rotulo = malloc((size_t)base->n);
    base->python.pred = malloc((size_t)base->n);
    if (base->rotulo == NULL || base->python.pred == NULL) {
        fprintf(stderr, "ERRO: memoria insuficiente.\n");
        status = -1;
        goto fim;
    }

    uint8_t registro[MLP_BYTES_REGISTRO];
    for (long i = 0; i < base->n; i++) {
        if (fread(registro, 1, MLP_BYTES_REGISTRO, f_teste) != MLP_BYTES_REGISTRO ||
            fread(&base->python.pred[i], 1, 1, f_py) != 1) {
            fprintf(stderr, "ERRO: leitura incompleta na amostra %ld.\n", i);
            status = -1;
            goto fim;
        }
        if (registro[0] >= MLP_SAIDAS || base->python.pred[i] >= MLP_SAIDAS) {
            fprintf(stderr, "ERRO: classe invalida na amostra %ld.\n", i);
            status = -1;
            goto fim;
        }
        base->rotulo[i] = registro[0];
        base->python.confusao[registro[0]][base->python.pred[i]]++;
        if (registro[0] == base->python.pred[i]) base->python.acertos++;
    }
    base->python.amostras = base->n;

fim:
    if (f_teste) fclose(f_teste);
    if (f_py) fclose(f_py);
    return status;
}

/* ------------------------------------------------------------------------ */
/* Execucao de um modulo                                                    */
/* ------------------------------------------------------------------------ */
/* Executa um modulo sobre o teste, em streaming: le uma imagem, infere,
 * compara com a referencia e acumula as metricas. RAM usada: um registro. */
static int executar(const Modulo *m, const char *dir, const Base *base, Resultado *r)
{
    FILE *f_teste = abrir(dir, ARQUIVO_TESTE);
    FILE *f_ref = abrir(dir, m->arquivo_ref);
    int status = 0;

    memset(r, 0, sizeof(*r));
    r->seg_min = 1e30;
    r->pred = malloc((size_t)base->n);
    if (f_teste == NULL || f_ref == NULL || r->pred == NULL) {
        status = -1;
        goto fim;
    }
    if (tamanho_arquivo(f_ref) != (long)(m->bytes_logits * (size_t)(tamanho_arquivo(f_teste) / MLP_BYTES_REGISTRO))) {
        fprintf(stderr, "ERRO: %s nao corresponde ao numero de registros de teste.\n", m->arquivo_ref);
        status = -1;
        goto fim;
    }

    uint8_t registro[MLP_BYTES_REGISTRO];
    Logits z, ref;

    for (long i = 0; i < base->n; i++) {
        if (fread(registro, 1, MLP_BYTES_REGISTRO, f_teste) != MLP_BYTES_REGISTRO ||
            fread(&ref, m->bytes_logits, 1, f_ref) != 1) {
            fprintf(stderr, "ERRO: leitura incompleta na amostra %ld.\n", i);
            status = -1;
            break;
        }
        const int real = registro[0];
        if (real != base->rotulo[i]) {
            fprintf(stderr, "ERRO: rotulo inconsistente na amostra %ld.\n", i);
            status = -1;
            break;
        }

        memset(&z, 0, sizeof(z));
        double t0 = agora_segundos();
        m->inferir(&registro[1], &z);
        double dt = agora_segundos() - t0;
        r->segundos += dt;
        if (dt < r->seg_min) r->seg_min = dt;
        if (dt > r->seg_max) r->seg_max = dt;

        const int previsto = m->argmax(&z);
        r->pred[i] = (uint8_t)previsto;
        r->confusao[real][previsto]++;
        if (previsto == real) r->acertos++;

        if (memcmp(&z, &ref, m->bytes_logits) != 0) {
            if (r->logits_diferentes == 0) {
                printf("  Primeira divergencia de logits na amostra %ld (rotulo %d):\n", i, real);
                m->imprimir("C:", &z);
                m->imprimir("Python:", &ref);
            }
            r->logits_diferentes++;
        }
        if (previsto != m->argmax(&ref)) r->decisoes_diferentes_ref++;
        r->amostras++;
    }

fim:
    if (f_teste) fclose(f_teste);
    if (f_ref) fclose(f_ref);
    return status;
}

/* ------------------------------------------------------------------------ */
/* Relatorios                                                               */
/* ------------------------------------------------------------------------ */
static void relatar_matriz(const char *titulo, const Resultado *r)
{
    printf("\n  %s (linhas = classe real, colunas = classe prevista)\n", titulo);
    printf("  real |");
    for (int p = 0; p < MLP_SAIDAS; p++) printf("%6d", p);
    printf(" | total\n  -----+");
    for (int p = 0; p < MLP_SAIDAS; p++) printf("------");
    printf("-+------\n");
    for (int c = 0; c < MLP_SAIDAS; c++) {
        long total = 0;
        printf("  %4d |", c);
        for (int p = 0; p < MLP_SAIDAS; p++) {
            printf("%6ld", r->confusao[c][p]);
            total += r->confusao[c][p];
        }
        printf(" | %5ld\n", total);
    }
}

/* Matriz do modulo menos a do Python: mostra exatamente quais confusoes mudaram.
 * Na diagonal, +n = n acertos a mais naquele digito; fora dela, +n = n erros a mais. */
static void relatar_matriz_diferenca(const Resultado *r, const Resultado *py)
{
    long alteradas = 0;
    printf("\n  Diferenca da matriz: modulo - Python original ('.' = igual)\n");
    printf("  real |");
    for (int p = 0; p < MLP_SAIDAS; p++) printf("%6d", p);
    printf("\n  -----+");
    for (int p = 0; p < MLP_SAIDAS; p++) printf("------");
    printf("\n");
    for (int c = 0; c < MLP_SAIDAS; c++) {
        printf("  %4d |", c);
        for (int p = 0; p < MLP_SAIDAS; p++) {
            long d = r->confusao[c][p] - py->confusao[c][p];
            if (d == 0) {
                printf("%6s", ".");
            } else {
                printf("%+6ld", d);
                alteradas++;
            }
        }
        printf("\n");
    }
    printf("  Celulas alteradas: %ld (na diagonal, positivo = mais acertos)\n", alteradas);
}

/*
 * Metricas da classe c a partir da matriz de confusao M[real][previsto]:
 *   verdadeiros positivos  VP = M[c][c]
 *   suporte   = soma da linha c   (imagens que realmente sao c)
 *   previstos = soma da coluna c  (imagens que o modelo chamou de c)
 *   precisao  = VP / previstos
 *   revocacao = VP / suporte
 *   F1        = 2 * precisao * revocacao / (precisao + revocacao)
 * Media macro: media simples das 10 classes. Media ponderada: pesa cada
 * classe pelo suporte (a revocacao ponderada coincide com a acuracia).
 */
static void metricas_classe(const Resultado *r, int c, long *suporte, long *previstos,
                            double *p, double *rv, double *f1)
{
    long vp = r->confusao[c][c];
    *suporte = 0;
    *previstos = 0;
    for (int k = 0; k < MLP_SAIDAS; k++) {
        *suporte += r->confusao[c][k];
        *previstos += r->confusao[k][c];
    }
    *p = razao(vp, *previstos);
    *rv = razao(vp, *suporte);
    *f1 = (*p + *rv > 0.0) ? 2.0 * *p * *rv / (*p + *rv) : 0.0;
}

static void relatar_classes(const Resultado *r, const Resultado *py)
{
    double soma[3] = {0.0, 0.0, 0.0};
    double pond[3] = {0.0, 0.0, 0.0};

    printf("\n  Metricas por classe%s\n", py ? " (dF1 = F1 do modulo - F1 do Python original)" : "");
    printf("  classe  suporte  previstos  precisao  revocacao      F1%s\n", py ? "      dF1" : "");
    for (int c = 0; c < MLP_SAIDAS; c++) {
        long sup, prev;
        double p, rv, f1;
        metricas_classe(r, c, &sup, &prev, &p, &rv, &f1);
        printf("  %6d  %7ld  %9ld  %8.4f  %9.4f  %6.4f", c, sup, prev, p, rv, f1);
        if (py) {
            long sup_py, prev_py;
            double p_py, rv_py, f1_py;
            metricas_classe(py, c, &sup_py, &prev_py, &p_py, &rv_py, &f1_py);
            printf("  %+7.4f", f1 - f1_py);
        }
        printf("\n");
        soma[0] += p;
        soma[1] += rv;
        soma[2] += f1;
        pond[0] += p * (double)sup;
        pond[1] += rv * (double)sup;
        pond[2] += f1 * (double)sup;
    }
    printf("  %-17s  %9s  %8.4f  %9.4f  %6.4f\n", "media macro", "", soma[0] / MLP_SAIDAS,
           soma[1] / MLP_SAIDAS, soma[2] / MLP_SAIDAS);
    if (r->amostras > 0) {
        double n = (double)r->amostras;
        printf("  %-17s  %9s  %8.4f  %9.4f  %6.4f\n", "media ponderada", "", pond[0] / n, pond[1] / n, pond[2] / n);
    }
}

static void relatar_python(const Base *base)
{
    const Resultado *py = &base->python;
    printf("==================================================================\n");
    printf(" Python original: float64, equivalente ao predict do scikit-learn\n");
    printf("==================================================================\n");
    printf("  Amostras : %ld\n", py->amostras);
    printf("  Acuracia : %.4f (%ld/%ld)\n", razao(py->acertos, py->amostras), py->acertos, py->amostras);
    relatar_matriz("Matriz de confusao", py);
    relatar_classes(py, NULL);
}

static void relatar_modulo(const Modulo *m, const Resultado *r, const Base *base)
{
    const Resultado *py = &base->python;
    Pareado c = comparar(py->pred, r->pred, base->rotulo, base->n);
    int validado = (r->amostras > 0 && r->logits_diferentes == 0);

    printf("  Amostras                           : %ld\n", r->amostras);
    printf("  Acuracia                           : %.4f (%ld/%ld)\n",
           razao(r->acertos, r->amostras), r->acertos, r->amostras);
    printf("  Diferenca vs Python original       : %+.4f (%+ld amostras)\n",
           razao(r->acertos - py->acertos, r->amostras), r->acertos - py->acertos);
    printf("  Logits diferentes da referencia    : %ld\n", r->logits_diferentes);
    printf("  Decisoes diferentes da referencia  : %ld\n", r->decisoes_diferentes_ref);
    printf("  Decisoes diferentes do Python      : %ld\n", c.discordantes);
    printf("    Python errava, modulo acerta     : %ld\n", c.b_corrige);
    printf("    Python acertava, modulo erra     : %ld\n", c.b_erra_novo);
    printf("    ambos erram (classes diferentes) : %ld\n", c.ambos_errados);
    printf("    p-valor McNemar exato            : %.4f (%s)\n", c.p,
           (c.p < ALFA) ? "diferenca significativa a 5%" : "compativel com o acaso a 5%");
    printf("  Tempo de inferencia (medio)        : %.3f us/imagem\n",
           (r->amostras > 0) ? r->segundos * 1e6 / (double)r->amostras : 0.0);
    printf("  Tempo de inferencia (min / max)    : %.3f / %.3f us\n",
           (r->amostras > 0) ? r->seg_min * 1e6 : 0.0, r->seg_max * 1e6);
    printf("  Vazao                              : %.1f imagens/s\n",
           (r->segundos > 0.0) ? (double)r->amostras / r->segundos : 0.0);
    relatar_matriz("Matriz de confusao", r);
    relatar_matriz_diferenca(r, py);
    relatar_classes(r, py);
    printf("\n  Resultado %-5s: %s\n", m->nome, validado ? "VALIDADO (bit a bit)" : "FALHA");
}

static void relatar_comparacao(const Modulo *exec[], const Resultado res[], int n, const Base *base)
{
    const Resultado *py = &base->python;
    double us_float = 0.0;
    for (int i = 0; i < n; i++) {
        if (strcmp(exec[i]->nome, "float") == 0 && res[i].amostras > 0) {
            us_float = res[i].segundos * 1e6 / (double)res[i].amostras;
        }
    }

    printf("\n==================================================================\n");
    printf(" COMPARACAO COM O PYTHON ORIGINAL\n");
    printf("==================================================================\n");
    printf("  modulo  acuracia  liquido  corrige  erra novo  p McNemar  bit a bit  us/imagem  aceleracao\n");
    printf("  %-6s  %8.4f  %7s  %7s  %9s  %9s  %9s  %9s  %10s\n", "python",
           razao(py->acertos, py->amostras), "-", "-", "-", "-", "-", "-", "-");
    for (int i = 0; i < n; i++) {
        const Resultado *r = &res[i];
        Pareado c = comparar(py->pred, r->pred, base->rotulo, base->n);
        double us = (r->amostras > 0) ? r->segundos * 1e6 / (double)r->amostras : 0.0;
        printf("  %-6s  %8.4f  %+7ld  %7ld  %9ld  %9.4f  %9s  %9.3f", exec[i]->nome,
               razao(r->acertos, r->amostras), r->acertos - py->acertos, c.b_corrige, c.b_erra_novo, c.p,
               (r->amostras > 0 && r->logits_diferentes == 0) ? "sim" : "NAO", us);
        if (us_float > 0.0 && us > 0.0) {
            printf("  %9.2fx\n", us_float / us);
        } else {
            printf("  %10s\n", "-");
        }
    }

    if (n > 1) {
        printf("\n  Comparacao pareada entre modulos (A x B)\n");
        printf("  par              discordantes  liquido B  B corrige  B erra novo  ambos erram  p McNemar\n");
        for (int a = 0; a < n; a++) {
            for (int b = a + 1; b < n; b++) {
                Pareado c = comparar(res[a].pred, res[b].pred, base->rotulo, base->n);
                char par[32];
                snprintf(par, sizeof(par), "%s x %s", exec[a]->nome, exec[b]->nome);
                printf("  %-15s  %12ld  %+9ld  %9ld  %11ld  %11ld  %9.4f\n", par, c.discordantes,
                       c.b_corrige - c.b_erra_novo, c.b_corrige, c.b_erra_novo, c.ambos_errados, c.p);
            }
        }
    }
    printf("\n  p McNemar < %.2f: diferenca de acuracia improvavel por acaso; caso contrario, compativel com o acaso.\n",
           ALFA);
}

/* ------------------------------------------------------------------------ */
/* Principal                                                                */
/* ------------------------------------------------------------------------ */
int main(int argc, char **argv)
{
    if (argc < 2 || argc > 4) {
        fprintf(stderr, "Uso: %s <float|int16|int8|todos> [n_amostras] [dir_dados]\n", argv[0]);
        listar_modulos();
        return 2;
    }
    if (!sistema_little_endian()) {
        fprintf(stderr, "ERRO: os arquivos de referencia sao little-endian.\n");
        return 2;
    }

    long n_max = -1;
    if (argc >= 3) {
        char *fim = NULL;
        n_max = strtol(argv[2], &fim, 10);
        if (fim == argv[2] || *fim != '\0' || n_max <= 0) {
            fprintf(stderr, "ERRO: n_amostras deve ser um inteiro positivo.\n");
            return 2;
        }
    }
    const char *dir = (argc == 4) ? argv[3] : DIR_PADRAO;
    const int todos = (strcmp(argv[1], "todos") == 0);

    int encontrado = todos;
    for (int i = 0; i < N_MODULOS && !encontrado; i++) {
        encontrado = (strcmp(argv[1], MODULOS[i].nome) == 0);
    }
    if (!encontrado) {
        fprintf(stderr, "ERRO: modulo '%s' nao existe ou nao foi compilado.\n", argv[1]);
        listar_modulos();
        return 2;
    }

    Base base;
    if (carregar_base(dir, n_max, &base) != 0) {
        free(base.rotulo);
        free(base.python.pred);
        return 1;
    }
    relatar_python(&base);

    const Modulo *exec[MAX_MODULOS];
    Resultado res[MAX_MODULOS];
    int n_exec = 0;
    int falhas = 0;

    for (int i = 0; i < N_MODULOS; i++) {
        const Modulo *m = &MODULOS[i];
        if (!todos && strcmp(argv[1], m->nome) != 0) continue;

        printf("==================================================================\n");
        printf(" Modulo %s: %s\n", m->nome, m->descricao);
        printf("==================================================================\n");
        Resultado *r = &res[n_exec];
        if (executar(m, dir, &base, r) != 0 || r->amostras != base.n) {
            falhas++;
            printf("  Resultado %-5s: FALHA (erro de execucao)\n", m->nome);
            free(r->pred);
            continue;
        }
        relatar_modulo(m, r, &base);
        if (r->logits_diferentes != 0) falhas++;
        exec[n_exec++] = m;
    }

    if (n_exec > 0) {
        relatar_comparacao(exec, res, n_exec, &base);
    }

    printf("==================================================================\n");
    printf(" %s\n", (falhas == 0) ? "SUCESSO: todos os modulos executados foram validados."
                                  : "FALHA: ha modulo com divergencia ou erro.");
    printf("==================================================================\n");

    for (int i = 0; i < n_exec; i++) free(res[i].pred);
    free(base.rotulo);
    free(base.python.pred);
    return (falhas == 0) ? 0 : 1;
}
