#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <math.h>
#include "v7_model_data.h"

static const int BINDER_IDX[23] = {
    1, 3, 4, 8, 9, 13, 14, 16, 17, 22, 24, 26,
    29, 31, 34, 37, 38, 39, 41, 42, 51, 53, 55
};

static double eval_tree(int base, const double *x) {
    int idx = 0;
    while (NODE_LEFT[base + idx] >= 0) {
        int feat = NODE_FEAT[base + idx];
        idx = (x[feat] <= NODE_THRESH[base + idx]) ? NODE_LEFT[base + idx] : NODE_RIGHT[base + idx];
    }
    return NODE_VAL[base + idx];
}

static int predict_class(const double *x) {
    double raw[NUM_CLASSES];
    for (int c = 0; c < NUM_CLASSES; ++c) raw[c] = INIT_SCORES[c];
    for (int t = 0; t < NUM_TREES; ++t) raw[t % NUM_CLASSES] += eval_tree(TREE_OFFSETS[t], x);
    int best = 0;
    for (int c = 1; c < NUM_CLASSES; ++c) if (raw[c] > raw[best]) best = c;
    return best + 1;
}

static char *next_csv(char **cur) {
    if (!*cur) return NULL;
    char *s = *cur, *c = strchr(s, ',');
    if (c) { *c = '\0'; *cur = c + 1; } else { *cur = NULL; }
    return s;
}

int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1] : "holdout_2320_top60_with_ref.csv";
    FILE *fp = fopen(path, "r");
    if (!fp) { perror("fopen"); return 1; }
    char line[65536];
    fgets(line, sizeof(line), fp);
    int total = 0, clean_ok = 0, masked_ok = 0;
    while (fgets(line, sizeof(line), fp)) {
        double x[NUM_FEATURES];
        char *cur = line;
        for (int f = 0; f < NUM_FEATURES; ++f) x[f] = strtod(next_csv(&cur), NULL);
        int true_c = (int)strtol(next_csv(&cur), NULL, 10);
        if (predict_class(x) == true_c) clean_ok++;
        for (int b = 0; b < 23; ++b) x[BINDER_IDX[b]] = 0.0;
        if (predict_class(x) == true_c) masked_ok++;
        total++;
    }
    fclose(fp);
    printf("On-Device Clean (60-D)           : %.2f%% (%d / %d)\n", 100.0 * clean_ok / total, clean_ok, total);
    printf("On-Device 23-Binder Masked (37-D): %.2f%% (%d / %d)\n", 100.0 * masked_ok / total, masked_ok, total);
    return 0;
}
