#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "stats.h"

#define VEC_ALIGN 32 /* bytes: alineacion requerida por AVX2 (256 bits) */

static double elapsed_ms(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) * 1000.0 +
           (end.tv_nsec - start.tv_nsec) / 1e6;
}

/* Reserva 'count' floats alineados a VEC_ALIGN bytes (aligned_alloc
 * exige que el tamano solicitado sea multiplo del alineamiento, por
 * eso se redondea hacia arriba). */
static float *alloc_aligned_floats(size_t count) {
    size_t bytes = count * sizeof(float);
    size_t padded = ((bytes + VEC_ALIGN - 1) / VEC_ALIGN) * VEC_ALIGN;
    if (padded == 0) padded = VEC_ALIGN;

    float *p = aligned_alloc(VEC_ALIGN, padded);
    if (!p) {
        fprintf(stderr, "Error: no se pudo reservar memoria alineada.\n");
        exit(EXIT_FAILURE);
    }
    memset(p, 0, padded);
    return p;
}

/*
 * Formato de input.dat (little endian):
 *   int32_t n
 *   float   arr[n]
 */
static float *read_input(const char *path, int *out_n) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Error: no se pudo abrir '%s'\n", path);
        exit(EXIT_FAILURE);
    }

    int32_t n = 0;
    if (fread(&n, sizeof(int32_t), 1, f) != 1) {
        fprintf(stderr, "Error: archivo de entrada invalido (falta N)\n");
        fclose(f);
        exit(EXIT_FAILURE);
    }
    if (n < 0) {
        fprintf(stderr, "Error: N invalido (%d)\n", n);
        fclose(f);
        exit(EXIT_FAILURE);
    }

    float *arr = alloc_aligned_floats((size_t)(n > 0 ? n : 1));
    if (n > 0 && fread(arr, sizeof(float), (size_t)n, f) != (size_t)n) {
        fprintf(stderr, "Error: archivo de entrada truncado\n");
        fclose(f);
        exit(EXIT_FAILURE);
    }

    fclose(f);
    *out_n = n;
    return arr;
}

static void write_output(const char *path, const float *arr, int n) {
    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "Error: no se pudo crear '%s'\n", path);
        exit(EXIT_FAILURE);
    }
    fwrite(&n, sizeof(int32_t), 1, f);
    if (n > 0) fwrite(arr, sizeof(float), (size_t)n, f);
    fclose(f);
}

/* Resumen en texto plano (para que tools/verify_reference.py no
 * tenga que parsear el binario de salida). */
static void write_stats_summary(const char *path, int n, float sum,
                                 float mean, float var, float stddev,
                                 float min, float max, double ms) {
    FILE *f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "Aviso: no se pudo crear el resumen '%s'\n", path);
        return;
    }
    fprintf(f, "n=%d\n", n);
    fprintf(f, "sum=%.9g\n", sum);
    fprintf(f, "mean=%.9g\n", mean);
    fprintf(f, "var=%.9g\n", var);
    fprintf(f, "stddev=%.9g\n", stddev);
    fprintf(f, "min=%.9g\n", min);
    fprintf(f, "max=%.9g\n", max);
    fprintf(f, "kernel_ms=%.6f\n", ms);
    fclose(f);
}

static void usage(const char *prog) {
    fprintf(stderr,
        "Uso: %s <input.dat> <output.dat> [repeticiones]\n"
        "  input.dat      archivo binario de entrada (int32 N + N floats)\n"
        "  output.dat     archivo binario de salida (arreglo normalizado)\n"
        "  repeticiones   veces que se repite el kernel para promediar\n"
        "                 el tiempo medido (por defecto: 1)\n",
        prog);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    const char *input_path  = argv[1];
    const char *output_path = argv[2];
    int reps = (argc >= 4) ? atoi(argv[3]) : 1;
    if (reps < 1) reps = 1;

    int n = 0;
    float *in  = read_input(input_path, &n);
    float *out = alloc_aligned_floats((size_t)(n > 0 ? n : 1));

    float sum = 0.0f, mean = 0.0f, var = 0.0f, min = 0.0f, max = 0.0f;
    double total_ms = 0.0;
    struct timespec t0, t1;

    /* --- Seccion medida: sum_array + compute_stats + normalize_array --- */
    for (int r = 0; r < reps; r++) {
        clock_gettime(CLOCK_MONOTONIC, &t0);

        sum = sum_array(in, n);
        compute_stats(in, n, &mean, &var, &min, &max);
        float stddev_r = sqrtf(var);
        normalize_array(in, out, n, mean, stddev_r);

        clock_gettime(CLOCK_MONOTONIC, &t1);
        total_ms += elapsed_ms(t0, t1);
    }
    double avg_ms = total_ms / reps;
    float stddev = sqrtf(var);

    printf("N        = %d\n", n);
    printf("Suma     = %.6f\n", sum);
    printf("Media    = %.6f\n", mean);
    printf("Varianza = %.6f\n", var);
    printf("StdDev   = %.6f\n", stddev);
    printf("Minimo   = %.6f\n", min);
    printf("Maximo   = %.6f\n", max);
    printf("Tiempo promedio del kernel (%d rep.): %.6f ms\n", reps, avg_ms);

    write_output(output_path, out, n);

    char summary_path[1024];
    snprintf(summary_path, sizeof(summary_path), "%s.stats.txt", output_path);
    write_stats_summary(summary_path, n, sum, mean, var, stddev, min, max, avg_ms);

    free(in);
    free(out);
    return EXIT_SUCCESS;
}
