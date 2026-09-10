/*
 * teste_dot_i8_neon.c - Teste isolado do kernel dot_i8_neon contra a referencia.
 *
 * O que prova: para todas as entradas testadas, dot_i8_neon devolve exatamente
 * o mesmo int32 que dot_i8_ref (escalar), e esse valor e o resultado
 * matematico exato (confirmado pela soma em int64, dot_i8_ref64).
 *
 * Grupos de teste:
 *   1. Resultado registrado na placa: srand(0), 16.384 elementos (dot_neon.c)
 *      -> esperado 576483.
 *   2. Todos os tamanhos multiplos de 16, de 16 a 4096, com vetores aleatorios.
 *   3. Casos extremos nos tamanhos da MLP (784 e 64): zeros, 127*127,
 *      -128*-128 (maior produto), -128*127 (menor produto), sinais alternados
 *      e impulsos (um unico elemento nao nulo em cada posicao).
 *   4. Desalinhamento: vetores comecando em 16 deslocamentos de memoria.
 *   5. Vetores reais da MLP: entrada quantizada (MLP8_LUT) de cada imagem de
 *      teste contra cada linha de MLP8_W0 (784 -> 64), e ativacao oculta
 *      contra cada linha de MLP8_W1 (64 -> 10).
 *
 * Nao testa tamanhos que nao sao multiplos de 16: o kernel nao trata tail
 * (restricao documentada em dot_i8_neon.h, satisfeita pela MLP).
 *
 * Uso: ./teste_dot_i8_neon [pasta_dados]   (padrao: ../modelo/exportado)
 * Codigo de saida: 0 somente se nao houver nenhuma diferenca.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dot_i8_neon.h"
#include "dot_i8_ref.h"
#include "mlp_pesos_int8.h"

#define ENTRADAS 784
#define OCULTA   64
#define SAIDAS   10
#define BYTES_REGISTRO (1 + ENTRADAS)
#define MAX_LEN  (16 * 1024)

typedef struct {
    const char *nome;
    long casos;
    long diferencas;
} Grupo;

/* Gerador pseudoaleatorio xorshift32: deterministico e igual em qualquer maquina. */
static uint32_t estado_rng = 2463534242u;

static int8_t aleatorio_i8(void)
{
    estado_rng ^= estado_rng << 13;
    estado_rng ^= estado_rng >> 17;
    estado_rng ^= estado_rng << 5;
    return (int8_t)(estado_rng & 0xFF);
}

/* Compara um caso; na primeira diferenca do grupo, imprime os detalhes. */
static void conferir(Grupo *g, const int8_t *a, const int8_t *b, int len, const char *descricao)
{
    int32_t ref = dot_i8_ref(a, b, len);
    int64_t exato = dot_i8_ref64(a, b, len);
    int32_t neon = dot_i8_neon(a, b, len);
    g->casos++;
    if (neon != ref || (int64_t)ref != exato) {
        if (g->diferencas == 0) {
            printf("    DIFERENCA [%s] len=%d: neon=%d ref=%d exato=%lld\n",
                   descricao, len, neon, ref, (long long)exato);
        }
        g->diferencas++;
    }
}

static void relatar(const Grupo *g)
{
    printf("  %-52s casos %8ld   diferencas %ld   %s\n", g->nome, g->casos, g->diferencas,
           g->diferencas == 0 ? "OK" : "FALHA");
}

/* Requantizacao identica a de c_modular/mlp_int8.c; usada aqui apenas para
 * gerar ativacoes ocultas realistas para o teste da camada 64 -> 10. */
static int8_t requantizar_para_teste(int32_t a)
{
    if (a < 0) a = 0;
    int64_t t = ((int64_t)a * MLP8_M + (INT64_C(1) << (MLP8_N - 1))) >> MLP8_N;
    return (int8_t)((t > MLP8_LIM) ? MLP8_LIM : t);
}

int main(int argc, char **argv)
{
    const char *dir = (argc > 1) ? argv[1] : "../modelo/exportado";
    static int8_t a[MAX_LEN + 16];
    static int8_t b[MAX_LEN + 16];
    int falhas = 0;

    (void)MLP8_B1;   /* nao usado neste teste */

    printf("==================================================================\n");
    printf(" Teste isolado: dot_i8_neon x dot_i8_ref (int8 x int8 -> int32)\n");
    printf("==================================================================\n");

    /* 1. Resultado registrado na placa (dot_neon.c: srand(0), LEN = 16*1024). */
    {
        Grupo g = {"1. resultado da placa (srand(0), 16384 elementos)", 0, 0};
        srand(0);
        for (int i = 0; i < MAX_LEN; i++) {
            a[i] = (int8_t)rand();
            b[i] = (int8_t)rand();
        }
        conferir(&g, a, b, MAX_LEN, "placa");
        int32_t neon = dot_i8_neon(a, b, MAX_LEN);
        printf("    dot_i8_neon = %d   (registrado na placa: 576483)\n", neon);
        if (neon != 576483) g.diferencas++;
        relatar(&g);
        falhas += (g.diferencas != 0);
    }

    /* 2. Todos os tamanhos multiplos de 16 ate 4096, 20 vetores cada. */
    {
        Grupo g = {"2. tamanhos 16..4096 (multiplos de 16), aleatorios", 0, 0};
        for (int len = 16; len <= 4096; len += 16) {
            for (int rep = 0; rep < 20; rep++) {
                for (int i = 0; i < len; i++) {
                    a[i] = aleatorio_i8();
                    b[i] = aleatorio_i8();
                }
                conferir(&g, a, b, len, "aleatorio");
            }
        }
        relatar(&g);
        falhas += (g.diferencas != 0);
    }

    /* 3. Casos extremos nos tamanhos da MLP. */
    {
        Grupo g = {"3. extremos e impulsos (784 e 64)", 0, 0};
        const int tamanhos[2] = {ENTRADAS, OCULTA};
        const int8_t pares[][2] = {{0, 0}, {127, 127}, {-128, -128}, {-128, 127},
                                   {127, -128}, {-127, -127}, {1, -1}};
        for (int t = 0; t < 2; t++) {
            int len = tamanhos[t];
            for (size_t p = 0; p < sizeof(pares) / sizeof(pares[0]); p++) {
                memset(a, pares[p][0], (size_t)len);
                memset(b, pares[p][1], (size_t)len);
                conferir(&g, a, b, len, "constante");
            }
            for (int i = 0; i < len; i++) {                    /* sinais alternados */
                a[i] = (i % 2) ? -128 : 127;
                b[i] = (i % 3) ? 127 : -128;
            }
            conferir(&g, a, b, len, "alternado");
            for (int pos = 0; pos < len; pos++) {              /* impulso em cada posicao */
                memset(a, 0, (size_t)len);
                memset(b, 0, (size_t)len);
                a[pos] = (int8_t)(-128 + (pos % 256));
                b[pos] = -128;
                conferir(&g, a, b, len, "impulso");
            }
        }
        relatar(&g);
        falhas += (g.diferencas != 0);
    }

    /* 4. Desalinhamento de memoria (vld1q_s8 nao exige alinhamento). */
    {
        Grupo g = {"4. desalinhamento (16 deslocamentos, len 784)", 0, 0};
        for (int off = 0; off < 16; off++) {
            for (int rep = 0; rep < 50; rep++) {
                for (int i = 0; i < ENTRADAS; i++) {
                    a[off + i] = aleatorio_i8();
                    b[off + i] = aleatorio_i8();
                }
                conferir(&g, a + off, b + off, ENTRADAS, "desalinhado");
            }
        }
        relatar(&g);
        falhas += (g.diferencas != 0);
    }

    /* 5. Vetores reais da MLP (10.000 imagens de teste). */
    {
        char caminho[4096];
        snprintf(caminho, sizeof(caminho), "%s/teste_u8.bin", dir);
        FILE *f = fopen(caminho, "rb");
        if (f == NULL) {
            printf("  ERRO: nao foi possivel abrir %s\n", caminho);
            return 1;
        }
        Grupo g1 = {"5a. MLP camada 784->64 (x real . linhas de MLP8_W0)", 0, 0};
        Grupo g2 = {"5b. MLP camada 64->10 (h real . linhas de MLP8_W1)", 0, 0};
        uint8_t registro[BYTES_REGISTRO];
        int8_t x[ENTRADAS];
        int8_t h[OCULTA];
        while (fread(registro, 1, BYTES_REGISTRO, f) == BYTES_REGISTRO) {
            for (int i = 0; i < ENTRADAS; i++) {
                x[i] = MLP8_LUT[i * 256 + registro[1 + i]];
            }
            for (int j = 0; j < OCULTA; j++) {
                const int8_t *w = &MLP8_W0[j * ENTRADAS];
                conferir(&g1, x, w, ENTRADAS, "camada 784->64");
                h[j] = requantizar_para_teste(MLP8_B0[j] + dot_i8_ref(x, w, ENTRADAS));
            }
            for (int k = 0; k < SAIDAS; k++) {
                conferir(&g2, h, &MLP8_W1[k * OCULTA], OCULTA, "camada 64->10");
            }
        }
        fclose(f);
        relatar(&g1);
        relatar(&g2);
        falhas += (g1.diferencas != 0) + (g2.diferencas != 0);
        if (g1.casos != 10000L * OCULTA || g2.casos != 10000L * SAIDAS) {
            printf("  ERRO: numero de imagens lidas diferente de 10000.\n");
            falhas++;
        }
    }

    printf("==================================================================\n");
    printf(" %s\n", falhas == 0 ? "SUCESSO: dot_i8_neon identico a referencia em todos os casos."
                                : "FALHA: ha diferencas; nao prosseguir.");
    printf("==================================================================\n");
    return falhas == 0 ? 0 : 1;
}
