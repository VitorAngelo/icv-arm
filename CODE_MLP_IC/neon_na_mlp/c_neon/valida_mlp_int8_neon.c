/*
 * valida_mlp_int8_neon.c - Validacao da MLP int8 com NEON contra o escalar.
 *
 * Para cada uma das 10.000 imagens de teste, com a mesma entrada:
 *   1. logits de mlp_int8_neon_inferir == logits de mlp_int8_inferir
 *      (c_modular/mlp_int8.c, compilado sem alteracao);
 *   2. logits de mlp_int8_neon_inferir == ref_int8.bin (especificacao Python);
 *   3. estagio a estagio, NEON x referencia escalar (dot_i8_ref):
 *        x, produto W0.x, produto + bias, ReLU, h requantizado, logits, argmax.
 *      Na primeira divergencia, imprime imagem, estagio, indice e valores.
 *   4. acuracia das duas versoes e numero de classificacoes divergentes.
 *
 * A referencia por estagio (estagios_referencia) repete as formulas de
 * c_modular/mlp_int8.c com dot_i8_ref. Para garantir que essa repeticao esta
 * certa, seus logits tambem sao comparados com mlp_int8_inferir.
 *
 * Uso: ./valida_mlp_int8_neon [pasta_dados]   (padrao: ../modelo/exportado)
 * Codigo de saida: 0 somente se todas as comparacoes forem identicas.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mlp.h"
#include "mlp_int8_neon.h"
#include "dot_i8_ref.h"
#include "mlp_pesos_int8.h"

#define BYTES_REGISTRO (1 + MLP_ENTRADAS)

enum {
    E_X, E_PRODUTO, E_A, E_RELU, E_H, E_LOGITS, E_ARGMAX,
    C_NEON_X_MODULAR, C_NEON_X_PYTHON, C_REF_X_MODULAR, N_COMPARACOES
};

static const char *NOMES[N_COMPARACOES] = {
    "784->64  entrada quantizada x",
    "784->64  produto W0.x (sem bias)",
    "784->64  produto + bias B0",
    "784->64  ReLU max(a, 0)",
    "784->64  h requantizado (int8)",
    "64->10   logits",
    "argmax   classe prevista",
    "logits NEON x mlp_int8_inferir (c_modular)",
    "logits NEON x ref_int8.bin (Python)",
    "autoverificacao: referencia x mlp_int8_inferir",
};

typedef struct {
    long comparacoes[N_COMPARACOES];
    long diferencas[N_COMPARACOES];
    int primeira_impressa;
} Contagem;

/* Referencia escalar por estagio: formulas de c_modular/mlp_int8.c com dot_i8_ref. */
static void estagios_referencia(const uint8_t *pixels, Mlp8Estagios *e)
{
    for (int i = 0; i < MLP_ENTRADAS; i++) {
        e->x[i] = MLP8_LUT[i * 256 + pixels[i]];
    }
    for (int j = 0; j < MLP_OCULTA; j++) {
        e->produto[j] = dot_i8_ref(e->x, &MLP8_W0[j * MLP_ENTRADAS], MLP_ENTRADAS);
        e->a[j] = MLP8_B0[j] + e->produto[j];
        e->relu[j] = (e->a[j] > 0) ? e->a[j] : 0;
        int64_t t = ((int64_t)e->relu[j] * MLP8_M + (INT64_C(1) << (MLP8_N - 1))) >> MLP8_N;
        e->h[j] = (int8_t)((t > MLP8_LIM) ? MLP8_LIM : t);
    }
    for (int k = 0; k < MLP_SAIDAS; k++) {
        e->logits[k] = MLP8_B1[k] + dot_i8_ref(e->h, &MLP8_W1[k * MLP_OCULTA], MLP_OCULTA);
    }
}

static int argmax(const int32_t *z)
{
    int k = 0;
    for (int j = 1; j < MLP_SAIDAS; j++) {
        if (z[j] > z[k]) k = j;
    }
    return k;
}

/* Compara n valores de um estagio; registra e imprime a primeira divergencia geral. */
static void comparar(Contagem *c, int estagio, long imagem, const int64_t *neon, const int64_t *ref, int n)
{
    for (int i = 0; i < n; i++) {
        c->comparacoes[estagio]++;
        if (neon[i] != ref[i]) {
            c->diferencas[estagio]++;
            if (!c->primeira_impressa) {
                printf("  PRIMEIRA DIVERGENCIA: imagem %ld, estagio '%s', indice %d: NEON=%lld referencia=%lld\n",
                       imagem, NOMES[estagio], i, (long long)neon[i], (long long)ref[i]);
                c->primeira_impressa = 1;
            }
        }
    }
}

#define COMPARAR_VETOR(c, est, img, va, vb, n)                      \
    do {                                                            \
        int64_t ta_[MLP_ENTRADAS], tb_[MLP_ENTRADAS];               \
        for (int q_ = 0; q_ < (n); q_++) {                          \
            ta_[q_] = (va)[q_];                                     \
            tb_[q_] = (vb)[q_];                                     \
        }                                                           \
        comparar((c), (est), (img), ta_, tb_, (n));                 \
    } while (0)

int main(int argc, char **argv)
{
    const char *dir = (argc > 1) ? argv[1] : "../modelo/exportado";
    char caminho[4096];

    snprintf(caminho, sizeof(caminho), "%s/teste_u8.bin", dir);
    FILE *f_teste = fopen(caminho, "rb");
    snprintf(caminho, sizeof(caminho), "%s/ref_int8.bin", dir);
    FILE *f_ref = fopen(caminho, "rb");
    if (f_teste == NULL || f_ref == NULL) {
        printf("ERRO: nao foi possivel abrir teste_u8.bin ou ref_int8.bin em %s\n", dir);
        return 1;
    }

    Contagem c;
    memset(&c, 0, sizeof(c));
    uint8_t registro[BYTES_REGISTRO];
    int32_t logits_modular[MLP_SAIDAS];
    int32_t logits_neon[MLP_SAIDAS];
    int32_t logits_python[MLP_SAIDAS];
    Mlp8Estagios est_neon, est_ref;
    long imagens = 0, acertos_escalar = 0, acertos_neon = 0, classes_divergentes = 0;

    printf("==================================================================\n");
    printf(" Validacao: mlp_int8_neon_inferir x escalar (c_modular) x Python\n");
    printf("==================================================================\n");

    while (fread(registro, 1, BYTES_REGISTRO, f_teste) == BYTES_REGISTRO) {
        if (fread(logits_python, sizeof(int32_t), MLP_SAIDAS, f_ref) != MLP_SAIDAS) {
            printf("ERRO: ref_int8.bin menor que o arquivo de teste.\n");
            return 1;
        }
        const uint8_t *pixels = &registro[1];
        const int rotulo = registro[0];

        mlp_int8_inferir(pixels, logits_modular);
        mlp_int8_neon_inferir(pixels, logits_neon);
        mlp_int8_neon_estagios(pixels, &est_neon);
        estagios_referencia(pixels, &est_ref);

        COMPARAR_VETOR(&c, E_X, imagens, est_neon.x, est_ref.x, MLP_ENTRADAS);
        COMPARAR_VETOR(&c, E_PRODUTO, imagens, est_neon.produto, est_ref.produto, MLP_OCULTA);
        COMPARAR_VETOR(&c, E_A, imagens, est_neon.a, est_ref.a, MLP_OCULTA);
        COMPARAR_VETOR(&c, E_RELU, imagens, est_neon.relu, est_ref.relu, MLP_OCULTA);
        COMPARAR_VETOR(&c, E_H, imagens, est_neon.h, est_ref.h, MLP_OCULTA);
        COMPARAR_VETOR(&c, E_LOGITS, imagens, est_neon.logits, est_ref.logits, MLP_SAIDAS);
        {
            int64_t cn = argmax(est_neon.logits), cr = argmax(est_ref.logits);
            comparar(&c, E_ARGMAX, imagens, &cn, &cr, 1);
        }
        COMPARAR_VETOR(&c, C_NEON_X_MODULAR, imagens, logits_neon, logits_modular, MLP_SAIDAS);
        COMPARAR_VETOR(&c, C_NEON_X_PYTHON, imagens, logits_neon, logits_python, MLP_SAIDAS);
        COMPARAR_VETOR(&c, C_REF_X_MODULAR, imagens, est_ref.logits, logits_modular, MLP_SAIDAS);

        int classe_escalar = argmax(logits_modular);
        int classe_neon = argmax(logits_neon);
        acertos_escalar += (classe_escalar == rotulo);
        acertos_neon += (classe_neon == rotulo);
        classes_divergentes += (classe_escalar != classe_neon);
        imagens++;
    }
    fclose(f_teste);
    fclose(f_ref);

    int falhas = (imagens != 10000);
    printf("\n  %-46s %12s %11s\n", "comparacao", "valores", "diferencas");
    for (int i = 0; i < N_COMPARACOES; i++) {
        if (i == C_NEON_X_MODULAR) printf("  ----------------------------------------------------------------------\n");
        printf("  %-46s %12ld %11ld   %s\n", NOMES[i], c.comparacoes[i], c.diferencas[i],
               c.diferencas[i] == 0 ? "OK" : "FALHA");
        falhas += (c.diferencas[i] != 0);
    }
    printf("\n  Imagens                         : %ld\n", imagens);
    printf("  Acuracia int8 escalar (c_modular): %.4f (%ld/%ld)\n", (double)acertos_escalar / (double)imagens,
           acertos_escalar, imagens);
    printf("  Acuracia int8 NEON               : %.4f (%ld/%ld)\n", (double)acertos_neon / (double)imagens,
           acertos_neon, imagens);
    printf("  Classificacoes divergentes       : %ld\n", classes_divergentes);
    falhas += (classes_divergentes != 0);

    printf("==================================================================\n");
    printf(" %s\n", falhas == 0 ? "SUCESSO: MLP int8 NEON identica ao escalar e ao Python em todos os estagios."
                                : "FALHA: ha divergencias; nao prosseguir.");
    printf("==================================================================\n");
    return falhas == 0 ? 0 : 1;
}
