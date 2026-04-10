#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int32_t dot_product_int8(const int8_t *a, const int8_t *b, int len)
{
    int32_t ret = 0;
    for (int i = 0; i < len; i++) {
        ret += a[i] * b[i];
    }
    return ret;
}

int32_t dot_product_int32(const int32_t *a, const int32_t *b, int len)
{
    int32_t ret = 0;
    for (int i = 0; i < len; i++) {
        ret += a[i] * b[i];
    }
    return ret;
}

float dot_product_float32(const float *a, const float *b, int len)
{
    float ret = 0.0;
    for (int i = 0; i < len; i++) {
        ret += a[i] * b[i];
    }
    return ret;
}

#define LEN (16*1024)

int main(int argc, char *argv[])
{
    int8_t *ai8 = malloc(LEN);
    int8_t *bi8 = malloc(LEN);
    int32_t *ai32 = malloc(LEN * sizeof(int32_t));
    int32_t *bi32 = malloc(LEN * sizeof(int32_t));
    int32_t doti8, doti32;
    float *af = malloc(LEN * sizeof(float));
    float *bf = malloc(LEN * sizeof(float));
    float dotf;
    int i;

    if (!ai8 || !bi8 || !ai32 || !bi32 || !af || !bf) {
        perror("malloc");
        return 1;
    }
    srand(0);//time(NULL));
    for (i = 0; i < LEN; i++) {
        ai8[i] = rand();
        bi8[i] = rand();
        ai32[i] = ai8[i];
        bi32[i] = bi8[i];
        af[i] = ai8[i];
        bf[i] = bi8[i];
    }
    doti8 = dot_product_int8(ai8, bi8, LEN);
    doti32 = dot_product_int32(ai32, bi32, LEN);
    dotf = dot_product_float32(af, bf, LEN);
    printf("%d %d %f\n", (int)doti8, doti32, dotf);
    free(ai8); free(bi8);
    free(ai32); free(bi32);
    free(af); free(bf);
    return 0;
}
