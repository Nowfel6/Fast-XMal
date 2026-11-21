#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/resource.h>
#include "v7_model_data.h"

typedef struct { int feature_index; double zero_fraction; double one_fraction; } PathFeat;
static double eval_tree(int base, const double *x) {
    int idx = 0;
    while (NODE_LEFT[base + idx] >= 0) {
        int feat = NODE_FEAT[base + idx];
        idx = (x[feat] <= NODE_THRESH[base + idx]) ? NODE_LEFT[base + idx] : NODE_RIGHT[base + idx];
    }
    return NODE_VAL[base + idx];
}
static void predict_sample(const double *x, double *probs_out, int *pred_out, double *raw_out) {
    double raw[NUM_CLASSES];
    for (int c = 0; c < NUM_CLASSES; ++c) raw[c] = INIT_SCORES[c];
    for (int t = 0; t < NUM_TREES; ++t) raw[t % NUM_CLASSES] += eval_tree(TREE_OFFSETS[t], x);
    if (raw_out) for (int c = 0; c < NUM_CLASSES; ++c) raw_out[c] = raw[c];
    double max_r = raw[0];
    for (int c = 1; c < NUM_CLASSES; ++c) if (raw[c] > max_r) max_r = raw[c];
    double sum_exp = 0.0; int best = 0;
    for (int c = 0; c < NUM_CLASSES; ++c) {
        probs_out[c] = exp(raw[c] - max_r);
        sum_exp += probs_out[c];
        if (raw[c] > raw[best]) best = c;
    }
    for (int c = 0; c < NUM_CLASSES; ++c) probs_out[c] /= sum_exp;
    *pred_out = best + 1;
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

enum SyscallAction {
    ACT_NONE = 0,
    ACT_ACCESS, ACT_BRK, ACT_CLONE, ACT_CONNECT, ACT_EPOLL_CTL, ACT_EPOLL_WAIT,
    ACT_EXECVE, ACT_EXIT, ACT_FLOCK, ACT_FSTATAT64, ACT_GETDENTS64, ACT_GETTID,
    ACT_GETTIMEOFDAY, ACT_LLSEEK, ACT_LSEEK, ACT_LSTAT64, ACT_MADVISE, ACT_MKDIR,
    ACT_MMAP2, ACT_MPROTECT, ACT_MUNMAP, ACT_NEWSELECT, ACT_OPEN, ACT_PIPE,
    ACT_PRCTL, ACT_PREAD64, ACT_PWRITE64, ACT_READ, ACT_READLINK, ACT_RECVFROM,
    ACT_RENAME, ACT_SENDMSG, ACT_SENDTO, ACT_SIGPROCMASK, ACT_SOCKET, ACT_STAT64,
    ACT_WRITE, ACT_WRITEV
};

typedef struct { const char *name; int act; int is_bionic_alias; } ScEntry;
static const ScEntry SC_TABLE[] = {
    {"_llseek",       ACT_LLSEEK,      0},
    {"_newselect",    ACT_NEWSELECT,   0},
    {"access",        ACT_ACCESS,      0},
    {"brk",           ACT_BRK,         0},
    {"clone",         ACT_CLONE,       0},
    {"connect",       ACT_CONNECT,     0},
    {"epoll_ctl",     ACT_EPOLL_CTL,   0},
    {"epoll_pwait",   ACT_EPOLL_WAIT,  1},
    {"epoll_wait",    ACT_EPOLL_WAIT,  0},
    {"execve",        ACT_EXECVE,      0},
    {"execveat",      ACT_EXECVE,      1},
    {"exit",          ACT_EXIT,        0},
    {"exit_group",    ACT_EXIT,        1},
    {"faccessat",     ACT_ACCESS,      1},
    {"faccessat2",    ACT_ACCESS,      1},
    {"flock",         ACT_FLOCK,       0},
    {"fstatat64",     ACT_FSTATAT64,   1},
    {"getdents64",    ACT_GETDENTS64,  0},
    {"gettid",        ACT_GETTID,      0},
    {"gettimeofday",  ACT_GETTIMEOFDAY,0},
    {"lseek",         ACT_LSEEK,       0},
    {"lstat64",       ACT_LSTAT64,     0},
    {"madvise",       ACT_MADVISE,     0},
    {"mkdir",         ACT_MKDIR,       0},
    {"mkdirat",       ACT_MKDIR,       1},
    {"mmap2",         ACT_MMAP2,       0},
    {"mprotect",      ACT_MPROTECT,    0},
    {"munmap",        ACT_MUNMAP,      0},
    {"open",          ACT_OPEN,        0},
    {"openat",        ACT_OPEN,        1},
    {"pipe",          ACT_PIPE,        0},
    {"pipe2",         ACT_PIPE,        1},
    {"prctl",         ACT_PRCTL,       0},
    {"pread64",       ACT_PREAD64,     0},
    {"pselect6",      ACT_NEWSELECT,   1},
    {"pwrite64",      ACT_PWRITE64,    0},
    {"read",          ACT_READ,        0},
    {"readlink",      ACT_READLINK,    0},
    {"readlinkat",    ACT_READLINK,    1},
    {"recvfrom",      ACT_RECVFROM,    0},
    {"rename",        ACT_RENAME,      0},
    {"renameat",      ACT_RENAME,      1},
    {"renameat2",     ACT_RENAME,      1},
    {"rt_sigprocmask",ACT_SIGPROCMASK, 1},
    {"sendmsg",       ACT_SENDMSG,     0},
    {"sendto",        ACT_SENDTO,      0},
    {"sigprocmask",   ACT_SIGPROCMASK, 0},
    {"socket",        ACT_SOCKET,      0},
    {"stat64",        ACT_STAT64,      0},
    {"write",         ACT_WRITE,       0},
    {"writev",        ACT_WRITEV,      0},
};
static const int SC_TABLE_N = sizeof(SC_TABLE) / sizeof(SC_TABLE[0]);

static int cmp_sc(const void *key, const void *elem) {
    return strcmp((const char *)key, ((const ScEntry *)elem)->name);
}

typedef struct {
    long total_lines, syscalls, resumed, signals, exits;
    long cat_direct, cat_alias, cat_open_ok, cat_open_fail, cat_other;
    long cnt_access, cnt_pread64, cnt_read, cnt_flock, cnt_mkdir, cnt_mprotect;
    long cnt_sigprocmask, cnt_lstat64, cnt_create_folder, cnt_newselect, cnt_writev;
    long cnt_terminate_thread, cnt_munmap, cnt_recvfrom, cnt_net_conn, cnt_net_sock;
    long cnt_fs_fail, cnt_rename, cnt_epoll_wait, cnt_llseek, cnt_brk, cnt_write;
    long cnt_stat64, cnt_fs_read, cnt_mmap2, cnt_madvise, cnt_net_write, cnt_gettimeofday;
    long cnt_lseek, cnt_prctl, cnt_fs_pipe, cnt_gettid, cnt_fs_create, cnt_epoll_ctl, cnt_getdents64;
    long cnt_execve, cnt_open_ok, cnt_pipe, cnt_pwrite64, cnt_readlink, cnt_clone_thread;
} TraceCounts;

static int parse_strace_log(const char *path, TraceCounts *tc) {
    memset(tc, 0, sizeof(*tc));
    FILE *fp = fopen(path, "r");
    if (!fp) return 0;
    char line[4096];
    while (fgets(line, sizeof(line), fp)) {
        tc->total_lines++;
        char *p = line;
        if (strncmp(p, "[pid ", 5) == 0) {
            char *rb = strchr(p, ']');
            if (!rb) continue;
            p = rb + 1;
            while (*p == ' ' || *p == '\t') p++;
        } else if (*p >= '0' && *p <= '9') {
            while (*p >= '0' && *p <= '9') p++;
            while (*p == ' ' || *p == '\t') p++;
        }
        if (strncmp(p, "+++", 3) == 0) { tc->exits++; continue; }
        if (strncmp(p, "---", 3) == 0) { tc->signals++; continue; }
        if (strncmp(p, "<...", 4) == 0) { tc->resumed++; continue; }
        char *par = strchr(p, '(');
        if (!par) continue;
        size_t nlen = (size_t)(par - p);
        if (nlen == 0 || nlen >= 48) { tc->syscalls++; tc->cat_other++; continue; }
        char name[48];
        memcpy(name, p, nlen);
        name[nlen] = '\0';
        tc->syscalls++;
        const ScEntry *e = (const ScEntry *)bsearch(name, SC_TABLE, SC_TABLE_N, sizeof(ScEntry), cmp_sc);
        if (!e) { tc->cat_other++; continue; }

        int failed = (strstr(par, ") = -1 ") != NULL);
        switch (e->act) {
            case ACT_ACCESS:       tc->cnt_access++;       if (e->is_bionic_alias) tc->cat_alias++; else tc->cat_direct++; break;
            case ACT_PREAD64:      tc->cnt_pread64++;      tc->cat_direct++; break;
            case ACT_READ:         tc->cnt_read++;         tc->cat_direct++; break;
            case ACT_FLOCK:        tc->cnt_flock++;        tc->cat_direct++; break;
            case ACT_MKDIR:        tc->cnt_mkdir++; if (!failed) tc->cnt_create_folder++; if (e->is_bionic_alias) tc->cat_alias++; else tc->cat_direct++; break;
            case ACT_MPROTECT:     tc->cnt_mprotect++;     tc->cat_direct++; break;
            case ACT_SIGPROCMASK:  tc->cnt_sigprocmask++;  if (e->is_bionic_alias) tc->cat_alias++; else tc->cat_direct++; break;
            case ACT_LSTAT64:      tc->cnt_lstat64++;      tc->cat_direct++; break;
            case ACT_STAT64:       tc->cnt_stat64++;       tc->cat_direct++; break;
            case ACT_FSTATAT64:
                if (strstr(par, "AT_SYMLINK_NOFOLLOW")) tc->cnt_lstat64++; else tc->cnt_stat64++;
                tc->cat_alias++; break;
            case ACT_NEWSELECT:    tc->cnt_newselect++;    if (e->is_bionic_alias) tc->cat_alias++; else tc->cat_direct++; break;
            case ACT_WRITEV:       tc->cnt_writev++;       tc->cat_direct++; break;
            case ACT_EXIT:         tc->cnt_terminate_thread++; tc->cat_direct++; break;
            case ACT_MMAP2:        tc->cnt_mmap2++;        tc->cat_direct++; break;
            case ACT_MUNMAP:       tc->cnt_munmap++;       tc->cat_direct++; break;
            case ACT_RECVFROM:     tc->cnt_recvfrom++;     tc->cat_direct++; break;
            case ACT_CONNECT:      tc->cnt_net_conn++;     tc->cat_direct++; break;
            case ACT_SOCKET:       tc->cnt_net_sock++;     tc->cat_direct++; break;
            case ACT_SENDTO:
            case ACT_SENDMSG:      tc->cnt_net_write++;    tc->cat_direct++; break;
            case ACT_RENAME:       tc->cnt_rename++;       if (e->is_bionic_alias) tc->cat_alias++; else tc->cat_direct++; break;
            case ACT_EPOLL_WAIT:   tc->cnt_epoll_wait++;   if (e->is_bionic_alias) tc->cat_alias++; else tc->cat_direct++; break;
            case ACT_LLSEEK:       tc->cnt_llseek++;       tc->cat_direct++; break;
            case ACT_BRK:          tc->cnt_brk++;          tc->cat_direct++; break;
            case ACT_WRITE:        tc->cnt_write++;        tc->cat_direct++; break;
            case ACT_MADVISE:      tc->cnt_madvise++;      tc->cat_direct++; break;
            case ACT_GETTIMEOFDAY: tc->cnt_gettimeofday++; tc->cat_direct++; break;
            case ACT_LSEEK:        tc->cnt_lseek++;        tc->cat_direct++; break;
            case ACT_PRCTL:        tc->cnt_prctl++;        tc->cat_direct++; break;
            case ACT_PIPE:         tc->cnt_fs_pipe++; tc->cnt_pipe++; if (e->is_bionic_alias) tc->cat_alias++; else tc->cat_direct++; break;
            case ACT_GETTID:       tc->cnt_gettid++;       tc->cat_direct++; break;
            case ACT_EPOLL_CTL:    tc->cnt_epoll_ctl++;    tc->cat_direct++; break;
            case ACT_GETDENTS64:   tc->cnt_getdents64++;   tc->cat_direct++; break;
            case ACT_EXECVE:       tc->cnt_execve++;       tc->cat_other++; break;
            case ACT_PWRITE64:     tc->cnt_pwrite64++;     tc->cat_other++; break;
            case ACT_READLINK:     tc->cnt_readlink++;     tc->cat_other++; break;
            case ACT_CLONE:        if (strstr(par, "CLONE_THREAD")) tc->cnt_clone_thread++; tc->cat_other++; break;
            case ACT_OPEN:
                if (failed) { tc->cnt_fs_fail++; tc->cat_open_fail++; }
                else {
                    tc->cnt_open_ok++; tc->cat_open_ok++;
                    if (strstr(par, "O_CREAT")) tc->cnt_fs_create++;
                    else if (strstr(par, "O_RDONLY")) tc->cnt_fs_read++;
                }
                break;
            default: tc->cat_other++; break;
        }
    }
    fclose(fp);
    return 1;
}

static void vectorize_v7(const TraceCounts *tc, double *x) {
    memset(x, 0, sizeof(double) * NUM_FEATURES);
    x[0]  = log1p((double)tc->cnt_access);
    x[2]  = log1p((double)tc->cnt_pread64);
    x[5]  = log1p((double)tc->cnt_read);
    x[6]  = log1p((double)tc->cnt_flock);
    x[7]  = log1p((double)tc->cnt_mkdir);
    x[10] = log1p((double)tc->cnt_mprotect);
    x[11] = log1p((double)tc->cnt_sigprocmask);
    x[12] = log1p((double)tc->cnt_lstat64);
    x[15] = log1p((double)tc->cnt_create_folder);
    x[18] = log1p((double)tc->cnt_newselect);
    x[19] = log1p((double)tc->cnt_writev);
    x[20] = log1p((double)tc->cnt_terminate_thread);
    long mem_sum = tc->cnt_execve + tc->cnt_execve + tc->cnt_mmap2 + tc->cnt_mprotect + tc->cnt_munmap;
    x[21] = log1p((double)mem_sum);
    x[23] = log1p((double)tc->cnt_munmap);
    x[25] = log1p((double)tc->cnt_recvfrom);
    x[27] = log1p((double)tc->cnt_net_conn);
    x[28] = log1p((double)tc->cnt_net_sock);
    x[30] = log1p((double)tc->cnt_fs_fail);
    x[32] = log1p((double)tc->cnt_rename);
    x[33] = log1p((double)tc->cnt_epoll_wait);
    x[35] = log1p((double)tc->cnt_llseek);
    x[36] = log1p((double)tc->cnt_brk);
    x[40] = log1p((double)tc->cnt_write);
    long io_sum = tc->cnt_clone_thread + tc->cnt_fs_create + tc->cnt_fs_read + tc->cnt_fs_fail
                + tc->cnt_fs_pipe + tc->cnt_net_write + tc->cnt_terminate_thread
                + tc->cnt_open_ok + tc->cnt_pipe + tc->cnt_pread64 + tc->cnt_pwrite64
                + tc->cnt_read + tc->cnt_readlink + tc->cnt_write + tc->cnt_writev;
    x[43] = log1p((double)io_sum);
    x[44] = log1p((double)tc->cnt_stat64);
    x[45] = log1p((double)tc->cnt_fs_read);
    x[46] = log1p((double)tc->cnt_mmap2);
    x[47] = log1p((double)tc->cnt_madvise);
    x[48] = log1p((double)tc->cnt_net_write);
    x[49] = log1p((double)tc->cnt_gettimeofday);
    x[50] = log1p((double)tc->cnt_lseek);
    x[52] = log1p((double)tc->cnt_prctl);
    x[54] = log1p((double)tc->cnt_fs_pipe);
    x[56] = log1p((double)tc->cnt_gettid);
    x[57] = log1p((double)tc->cnt_fs_create);
    x[58] = log1p((double)tc->cnt_epoll_ctl);
    x[59] = log1p((double)tc->cnt_getdents64);
}

static int cmp_d(const void *a, const void *b) { double da=*(const double*)a, db=*(const double*)b; return (da>db)-(da<db); }
static void summarize(const char *lbl, double *arr, int n) {
    double *s = malloc(sizeof(double)*n); memcpy(s, arr, sizeof(double)*n); qsort(s, n, sizeof(double), cmp_d);
    double sum=0, sq=0; for(int i=0;i<n;++i) sum+=arr[i]; double m=sum/n;
    for(int i=0;i<n;++i) sq+=(arr[i]-m)*(arr[i]-m); double sd=sqrt(sq/(n-1));
    printf("  %-18s: mean=%7.3f ± %6.3f ms | P50=%7.3f | P95=%7.3f | P99=%7.3f\n",
           lbl, m, sd, s[(int)(0.50*(n-1))], s[(int)(0.95*(n-1))], s[(int)(0.99*(n-1))]);
    free(s);
}

int main(void) {
    const char *logs[3] = {"real_benign_app_v7.log", "real_tar_v7.log", "real_clang_v7.log"};
    const int WARM = 50, ROUNDS = 1000;
    double *t_parse[3], *t_vec[3], *t_pred[3], *t_shap[3], *t_det[3], *t_tot[3];
    for (int f = 0; f < 3; ++f) {
        t_parse[f] = malloc(sizeof(double)*ROUNDS); t_vec[f] = malloc(sizeof(double)*ROUNDS);
        t_pred[f]  = malloc(sizeof(double)*ROUNDS); t_shap[f]= malloc(sizeof(double)*ROUNDS);
        t_det[f]   = malloc(sizeof(double)*ROUNDS); t_tot[f] = malloc(sizeof(double)*ROUNDS);
    }
    TraceCounts tc; double x[NUM_FEATURES], probs[NUM_CLASSES], raw[NUM_CLASSES], phi[NUM_FEATURES*NUM_CLASSES];
    int pcls = 0; volatile double sink = 0.0;

    printf("=== V7 LOG AUDIT & RECONCILIATION ===\n");
    for (int f = 0; f < 3; ++f) {
        if (!parse_strace_log(logs[f], &tc)) { fprintf(stderr, "Missing %s\n", logs[f]); return 1; }
        vectorize_v7(&tc, x);
        predict_sample(x, probs, &pcls, raw);
        compute_treeshap(x, phi);
        int active = 0; for (int i = 0; i < NUM_FEATURES; ++i) if (x[i] > 0.0) active++;
        double max_add = 0.0;
        for (int c = 0; c < NUM_CLASSES; ++c) {
            double s = EXPECTED_SHAP[c];
            for (int i = 0; i < NUM_FEATURES; ++i) s += phi[i*NUM_CLASSES+c];
            double d = fabs(s - raw[c]); if (d > max_add) max_add = d;
        }
        long part_sum = tc.cat_direct + tc.cat_alias + tc.cat_open_ok + tc.cat_open_fail + tc.cat_other;
        long line_sum = tc.syscalls + tc.resumed + tc.signals + tc.exits;
        printf("[%s] lines=%ld (chk=%ld: sc=%ld+res=%ld+sig=%ld+ex=%ld) | sc_part=%ld (dir=%ld+als=%ld+openOK=%ld+openFail=%ld+oth=%ld) | act=%d/37 | pred=C%d (P=%.4f) | addErr=%.2e\n",
               logs[f], tc.total_lines, line_sum, tc.syscalls, tc.resumed, tc.signals, tc.exits,
               part_sum, tc.cat_direct, tc.cat_alias, tc.cat_open_ok, tc.cat_open_fail, tc.cat_other,
               active, pcls, probs[pcls-1], max_add);
    }

    for (int w = 0; w < WARM; ++w) {
        for (int f = 0; f < 3; ++f) {
            parse_strace_log(logs[f], &tc); vectorize_v7(&tc, x);
            predict_sample(x, probs, &pcls, raw); compute_treeshap(x, phi);
            sink += probs[0] + phi[0];
        }
    }
    struct timespec a, b, c, d, e;
    for (int r = 0; r < ROUNDS; ++r) {
        for (int k = 0; k < 3; ++k) {
            int f = (r + k) % 3;
            clock_gettime(CLOCK_MONOTONIC, &a); parse_strace_log(logs[f], &tc); clock_gettime(CLOCK_MONOTONIC, &b);
            vectorize_v7(&tc, x); clock_gettime(CLOCK_MONOTONIC, &c);
            predict_sample(x, probs, &pcls, raw); clock_gettime(CLOCK_MONOTONIC, &d);
            compute_treeshap(x, phi); clock_gettime(CLOCK_MONOTONIC, &e);
            sink += probs[0] + phi[0];
            double mp = (b.tv_sec-a.tv_sec)*1e3 + (b.tv_nsec-a.tv_nsec)/1e6;
            double mv = (c.tv_sec-b.tv_sec)*1e3 + (c.tv_nsec-b.tv_nsec)/1e6;
            double md = (d.tv_sec-c.tv_sec)*1e3 + (d.tv_nsec-c.tv_nsec)/1e6;
            double ms = (e.tv_sec-d.tv_sec)*1e3 + (e.tv_nsec-d.tv_nsec)/1e6;
            t_parse[f][r] = mp; t_vec[f][r] = mv; t_pred[f][r] = md; t_shap[f][r] = ms;
            t_det[f][r] = mv + md + ms; t_tot[f][r] = mp + mv + md + ms;
        }
    }
    struct rusage ru; getrusage(RUSAGE_SELF, &ru);
    printf("\n=== V7 FLASH LOG PARSING BENCHMARK (N=%d INTERLEAVED ROUNDS) | Peak RSS: %ld KB (%.2f MB) ===\n",
           ROUNDS, ru.ru_maxrss, ru.ru_maxrss / 1024.0);
    for (int f = 0; f < 3; ++f) {
        parse_strace_log(logs[f], &tc);
        double pm = 0; for (int r = 0; r < ROUNDS; ++r) pm += t_parse[f][r]; pm /= ROUNDS;
        printf("\n[%s] syscalls=%ld (%.1fk syscalls/s)\n", logs[f], tc.syscalls, (tc.syscalls / pm));
        summarize("Parse",            t_parse[f], ROUNDS);
        summarize("Vec",              t_vec[f],   ROUNDS);
        summarize("Pred",             t_pred[f],  ROUNDS);
        summarize("TreeSHAP",         t_shap[f],  ROUNDS);
        summarize("Post-Parse Det",   t_det[f],   ROUNDS);
        summarize("Total (All)",      t_tot[f],   ROUNDS);
    }
    return (sink < -1e100) ? 1 : 0;
}
