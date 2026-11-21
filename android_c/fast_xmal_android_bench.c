#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <time.h>
#include <sys/resource.h>
#include "v7_model_data.h"
#define EXPECTED_HOLDOUT_N 2320
#define EXPECTED_SHAP_N 200

typedef struct { int feature_index; double zero_fraction; double one_fraction; } PathFeat;
static double eval_tree(int base, const double *x) {
    int idx = 0;
    while (NODE_LEFT[base + idx] >= 0) {
        int feat = NODE_FEAT[base + idx];
        idx = (x[feat] <= NODE_THRESH[base + idx]) ? NODE_LEFT[base + idx] : NODE_RIGHT[base + idx];
    }
    return NODE_VAL[base + idx];
}
static void predict_sample(const double *x, double *probs_out, int *pred_class_out, double *raw_out) {
    double raw[NUM_CLASSES];
    for (int c = 0; c < NUM_CLASSES; ++c) raw[c] = INIT_SCORES[c];
    for (int t = 0; t < NUM_TREES; ++t) raw[t % NUM_CLASSES] += eval_tree(TREE_OFFSETS[t], x);
    if (raw_out) for (int c = 0; c < NUM_CLASSES; ++c) raw_out[c] = raw[c];
    double max_r = raw[0];
    for (int c = 1; c < NUM_CLASSES; ++c) if (raw[c] > max_r) max_r = raw[c];
    double sum_exp = 0.0; int best_c = 0;
    for (int c = 0; c < NUM_CLASSES; ++c) {
        probs_out[c] = exp(raw[c] - max_r);
        sum_exp += probs_out[c];
        if (raw[c] > raw[best_c]) best_c = c;
    }
    for (int c = 0; c < NUM_CLASSES; ++c) probs_out[c] /= sum_exp;
    *pred_class_out = best_c + 1;
}
static void extend_path(PathFeat *pf, double *pw, int ud, double zf, double of, int fi) {
    pf[ud].feature_index = fi; pf[ud].zero_fraction = zf; pf[ud].one_fraction = of;
    pw[ud] = (ud == 0) ? 1.0 : 0.0;
    for (int i = ud - 1; i >= 0; --i) {
        pw[i + 1] += of * pw[i] * (double)(i + 1) / (double)(ud + 1);
        pw[i] = zf * pw[i] * (double)(ud - i) / (double)(ud + 1);
    }
}
static void unwind_path(PathFeat *pf, double *pw, int ud, int pi) {
    const double of = pf[pi].one_fraction, zf = pf[pi].zero_fraction;
    double n1 = pw[ud];
    for (int i = ud - 1; i >= 0; --i) {
        if (of != 0.0) {
            const double tmp = pw[i];
            pw[i] = n1 * (double)(ud + 1) / ((double)(i + 1) * of);
            n1 = tmp - pw[i] * zf * (double)(ud - i) / (double)(ud + 1);
        } else {
            pw[i] = (pw[i] * (double)(ud + 1)) / (zf * (double)(ud - i));
        }
    }
    for (int i = pi; i < ud; ++i) pf[i] = pf[i + 1];
}
static double unwound_sum(const PathFeat *pf, const double *pw, int ud, int pi) {
    const double of = pf[pi].one_fraction, zf = pf[pi].zero_fraction;
    double n1 = pw[ud], tot = 0.0;
    if (of != 0.0) {
        for (int i = ud - 1; i >= 0; --i) {
            const double tmp = n1 * (double)(ud + 1) / ((double)(i + 1) * of);
            tot += tmp;
            n1 = pw[i] - tmp * zf * (double)(ud - i) / (double)(ud + 1);
        }
    } else {
        for (int i = ud - 1; i >= 0; --i) tot += (pw[i] * (double)(ud + 1)) / (zf * (double)(ud - i));
    }
    return tot;
}
static void tree_shap_recurse(int base, int node_idx, const double *x, double *phi_cls,
                              const PathFeat *ppf, const double *ppw, int ud, double pzf, double pof, int pfi) {
    PathFeat pf[MAX_PATH_LEN]; double pw[MAX_PATH_LEN];
    for (int i = 0; i < ud; ++i) { pf[i] = ppf[i]; pw[i] = ppw[i]; }
    extend_path(pf, pw, ud, pzf, pof, pfi);
    const int left = NODE_LEFT[base + node_idx];
    if (left < 0) {
        const double val = NODE_VAL[base + node_idx];
        for (int i = 1; i <= ud; ++i) phi_cls[pf[i].feature_index] += unwound_sum(pf, pw, ud, i) * (pf[i].one_fraction - pf[i].zero_fraction) * val;
        return;
    }
    const int right = NODE_RIGHT[base + node_idx], sf = NODE_FEAT[base + node_idx];
    const double th = NODE_THRESH[base + node_idx], pc = NODE_COUNT[base + node_idx];
    const int hc = (x[sf] <= th) ? left : right, cc = (hc == left) ? right : left;
    double izf = 1.0, iof = 1.0; int nd = ud, ex_i = -1;
    for (int i = 1; i <= ud; ++i) if (pf[i].feature_index == sf) { ex_i = i; break; }
    if (ex_i >= 1) { izf = pf[ex_i].zero_fraction; iof = pf[ex_i].one_fraction; unwind_path(pf, pw, ud, ex_i); nd -= 1; }
    tree_shap_recurse(base, hc, x, phi_cls, pf, pw, nd + 1, izf * (NODE_COUNT[base + hc] / pc), iof, sf);
    tree_shap_recurse(base, cc, x, phi_cls, pf, pw, nd + 1, izf * (NODE_COUNT[base + cc] / pc), 0.0, sf);
}
static void compute_treeshap(const double *x, double *phi_out) {
    memset(phi_out, 0, sizeof(double) * NUM_FEATURES * NUM_CLASSES);
    PathFeat dpf[MAX_PATH_LEN]; double dpw[MAX_PATH_LEN];
    for (int t = 0; t < NUM_TREES; ++t) {
        int cls = t % NUM_CLASSES; double phi_cls[NUM_FEATURES]; memset(phi_cls, 0, sizeof(phi_cls));
        tree_shap_recurse(TREE_OFFSETS[t], 0, x, phi_cls, dpf, dpw, 0, 1.0, 1.0, -1);
        for (int f = 0; f < NUM_FEATURES; ++f) phi_out[f * NUM_CLASSES + cls] += phi_cls[f];
    }
}
static char *next_csv(char **cur) {
    if (!*cur) return NULL; char *s = *cur, *c = strchr(s, ',');
    if (c) { *c = '\0'; *cur = c + 1; } else { *cur = NULL; }
    return s;
}
static int p_dbl(const char *s, double *out) {
    if (!s) return 0; while (*s && isspace((unsigned char)*s)) s++; if (!*s) return 0;
    char *e = NULL; errno = 0; double v = strtod(s, &e);
    if (e == s || errno == ERANGE || !isfinite(v)) return 0;
    while (*e && isspace((unsigned char)*e)) e++; if (*e) return 0; *out = v; return 1;
}
static int p_int(const char *s, int *out) {
    if (!s) return 0; while (*s && isspace((unsigned char)*s)) s++; if (!*s) return 0;
    char *e = NULL; errno = 0; long v = strtol(s, &e, 10);
    if (e == s || errno == ERANGE) return 0;
    while (*e && isspace((unsigned char)*e)) e++; if (*e) return 0; *out = (int)v; return 1;
}
static int cmp_d(const void *a, const void *b) { double da=*(const double*)a, db=*(const double*)b; return (da>db)-(da<db); }
static void pstats(const char *lbl, double *arr, int n) {
    double *s = (double*)malloc(sizeof(double)*n); memcpy(s, arr, sizeof(double)*n); qsort(s, n, sizeof(double), cmp_d);
    double sum=0, sq=0; for(int i=0;i<n;++i) sum+=arr[i]; double mean=sum/n;
    for(int i=0;i<n;++i) sq+=(arr[i]-mean)*(arr[i]-mean); double sd=(n>1)?sqrt(sq/(n-1)):0;
    printf("%-28s | mean=%7.4f ms | std=%6.4f ms | P50=%7.4f ms | P95=%7.4f ms | P99=%7.4f ms\n",
           lbl, mean, sd, s[(int)(0.5*(n-1))], s[(int)(0.95*(n-1))], s[(int)(0.99*(n-1))]);
    free(s);
}
int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "Usage: %s <holdout_2320_csv> <shap_ref_200_csv>\n", argv[0]); return 1; }
    FILE *fp = fopen(argv[1], "r"); if (!fp) { perror("fopen"); return 2; }
    int cap = EXPECTED_HOLDOUT_N + 8, n = 0;
    double *X = malloc(sizeof(double)*cap*NUM_FEATURES), *py_prob = malloc(sizeof(double)*cap*NUM_CLASSES);
    int *true_c = malloc(sizeof(int)*cap), *py_pred = malloc(sizeof(int)*cap);
    char line[65536]; fgets(line, sizeof(line), fp);
    while (fgets(line, sizeof(line), fp)) {
        char *cur = line;
        for (int f=0; f<NUM_FEATURES; ++f) if (!p_dbl(next_csv(&cur), &X[n*NUM_FEATURES+f])) return 2;
        if (!p_int(next_csv(&cur), &true_c[n])) return 2;
        if (!p_int(next_csv(&cur), &py_pred[n])) return 2;
        for (int c=0; c<NUM_CLASSES; ++c) if (!p_dbl(next_csv(&cur), &py_prob[n*NUM_CLASSES+c])) return 2;
        n++;
    }
    fclose(fp);
    int pred_m = 0, acc_m = 0; double max_pd = 0.0;
    for (int i=0; i<n; ++i) {
        double probs[NUM_CLASSES], raw[NUM_CLASSES]; int pc = 0;
        predict_sample(&X[i*NUM_FEATURES], probs, &pc, raw);
        if (pc == py_pred[i]) pred_m++; if (pc == true_c[i]) acc_m++;
        for (int c=0; c<NUM_CLASSES; ++c) { double d = fabs(probs[c] - py_prob[i*NUM_CLASSES+c]); if (d > max_pd) max_pd = d; }
    }
    FILE *sfp = fopen(argv[2], "r"); if (!sfp) { perror("fopen shap"); return 2; }
    fgets(line, sizeof(line), sfp);
    double max_sd = -1.0, max_add = 0.0; int sr = 0;
    while (fgets(line, sizeof(line), sfp)) {
        double phi[NUM_FEATURES*NUM_CLASSES], probs[NUM_CLASSES], raw[NUM_CLASSES]; int pc=0;
        predict_sample(&X[sr*NUM_FEATURES], probs, &pc, raw);
        compute_treeshap(&X[sr*NUM_FEATURES], phi);
        char *cur = line;
        for (int k=0; k<NUM_FEATURES*NUM_CLASSES; ++k) {
            double rv=0; if (!p_dbl(next_csv(&cur), &rv)) return 2;
            double d = fabs(phi[k] - rv); if (max_sd < 0 || d > max_sd) max_sd = d;
        }
        for (int c=0; c<NUM_CLASSES; ++c) {
            double s = EXPECTED_SHAP[c];
            for (int f=0; f<NUM_FEATURES; ++f) s += phi[f*NUM_CLASSES+c];
            double ad = fabs(s - raw[c]); if (ad > max_add) max_add = ad;
        }
        sr++;
    }
    fclose(sfp);
    int bn = (n < 200) ? n : 200; double pw[NUM_CLASSES], phw[NUM_FEATURES*NUM_CLASSES], rw[NUM_CLASSES]; int pwc=0;
    for (int w=0; w<20; ++w) { predict_sample(X, pw, &pwc, rw); compute_treeshap(X, phw); }
    double *tp = malloc(sizeof(double)*bn), *ts = malloc(sizeof(double)*bn), *tt = malloc(sizeof(double)*bn);
    volatile double sink = 0.0; struct timespec ts0, ts1, ts2, ts3;
    for (int i=0; i<bn; ++i) {
        clock_gettime(CLOCK_MONOTONIC, &ts0); predict_sample(&X[i*NUM_FEATURES], pw, &pwc, rw); clock_gettime(CLOCK_MONOTONIC, &ts1);
        clock_gettime(CLOCK_MONOTONIC, &ts2); compute_treeshap(&X[i*NUM_FEATURES], phw); clock_gettime(CLOCK_MONOTONIC, &ts3);
        sink += pw[0] + phw[0];
        tp[i] = (ts1.tv_sec-ts0.tv_sec)*1e3 + (ts1.tv_nsec-ts0.tv_nsec)/1e6;
        ts[i] = (ts3.tv_sec-ts2.tv_sec)*1e3 + (ts3.tv_nsec-ts2.tv_nsec)/1e6;
        tt[i] = tp[i] + ts[i];
    }
    struct rusage ru; getrusage(RUSAGE_SELF, &ru);
    printf("Class Parity vs Python : %d / %d\n", pred_m, n);
    printf("Holdout Accuracy       : %.2f%% (%d / %d)\n", 100.0*acc_m/n, acc_m, n);
    printf("Max |prob_C - prob_Py| : %.3e\n", max_pd);
    printf("Max |SHAP_C - SHAP_Py| : %.3e (across %d rows)\n", max_sd, sr);
    printf("Max |Additivity Err|   : %.3e (across %d rows x 5 classes)\n", max_add, sr);
    printf("Process Peak RSS       : %ld KB (%.2f MB)\n", ru.ru_maxrss, ru.ru_maxrss/1024.0);
    pstats("Prediction Only (1T)", tp, bn); pstats("TreeSHAP Only (1T)", ts, bn); pstats("Pred + TreeSHAP (1T)", tt, bn);
    free(X); free(py_prob); free(true_c); free(py_pred); free(tp); free(ts); free(tt);
    return (pred_m == n && max_pd < 1e-6 && max_sd < 1e-5 && sink > -1e100) ? 0 : 2;
}
