#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>
#include <sched.h>
#include <sys/wait.h>
#include <sys/ptrace.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <sys/resource.h>
#include <linux/elf.h>
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

typedef struct {
    long syscalls;
    long cat_direct, cat_alias, cat_open_ok, cat_open_fail, cat_other;
    long cnt_access, cnt_pread64, cnt_read, cnt_flock, cnt_mkdir, cnt_mprotect;
    long cnt_sigprocmask, cnt_lstat64, cnt_create_folder, cnt_newselect, cnt_writev;
    long cnt_terminate_thread, cnt_munmap, cnt_recvfrom, cnt_net_conn, cnt_net_sock;
    long cnt_fs_fail, cnt_rename, cnt_epoll_wait, cnt_llseek, cnt_brk, cnt_write;
    long cnt_stat64, cnt_fs_read, cnt_mmap2, cnt_madvise, cnt_net_write, cnt_gettimeofday;
    long cnt_lseek, cnt_prctl, cnt_fs_pipe, cnt_gettid, cnt_fs_create, cnt_epoll_ctl, cnt_getdents64;
    long cnt_execve, cnt_open_ok, cnt_pipe, cnt_pwrite64, cnt_readlink, cnt_clone_thread;
} PtraceCounts;

static void record_syscall(PtraceCounts *tc, int nr, const unsigned long *args, long ret) {
    int failed = (ret < 0 && ret >= -4095L);
    tc->syscalls++;
    switch (nr) {
        case __NR_access:        tc->cnt_access++;       tc->cat_direct++; break;
        case __NR_faccessat:
#ifdef __NR_faccessat2
        case __NR_faccessat2:
#endif
            tc->cnt_access++; tc->cat_alias++; break;
        case __NR_pread64:       tc->cnt_pread64++;      tc->cat_direct++; break;
        case __NR_read:          tc->cnt_read++;         tc->cat_direct++; break;
        case __NR_flock:         tc->cnt_flock++;        tc->cat_direct++; break;
        case __NR_mkdir:         tc->cnt_mkdir++; if (!failed) tc->cnt_create_folder++; tc->cat_direct++; break;
        case __NR_mkdirat:       tc->cnt_mkdir++; if (!failed) tc->cnt_create_folder++; tc->cat_alias++; break;
        case __NR_mprotect:      tc->cnt_mprotect++;     tc->cat_direct++; break;
        case __NR_sigprocmask:   tc->cnt_sigprocmask++;  tc->cat_direct++; break;
        case __NR_rt_sigprocmask:tc->cnt_sigprocmask++;  tc->cat_alias++;  break;
        case __NR_lstat64:       tc->cnt_lstat64++;      tc->cat_direct++; break;
        case __NR_stat64:        tc->cnt_stat64++;       tc->cat_direct++; break;
        case __NR_fstatat64:
            if (args[3] & AT_SYMLINK_NOFOLLOW) tc->cnt_lstat64++; else tc->cnt_stat64++;
            tc->cat_alias++; break;
        case __NR__newselect:    tc->cnt_newselect++;    tc->cat_direct++; break;
        case __NR_pselect6:
#ifdef __NR_pselect6_time64
        case __NR_pselect6_time64:
#endif
            tc->cnt_newselect++; tc->cat_alias++; break;
        case __NR_writev:        tc->cnt_writev++;       tc->cat_direct++; break;
        case __NR_exit:
        case __NR_exit_group:    tc->cnt_terminate_thread++; tc->cat_direct++; break;
        case __NR_mmap2:         tc->cnt_mmap2++;        tc->cat_direct++; break;
        case __NR_munmap:        tc->cnt_munmap++;       tc->cat_direct++; break;
        case __NR_recvfrom:      tc->cnt_recvfrom++;     tc->cat_direct++; break;
        case __NR_connect:       tc->cnt_net_conn++;     tc->cat_direct++; break;
        case __NR_socket:        tc->cnt_net_sock++;     tc->cat_direct++; break;
        case __NR_sendto:
        case __NR_sendmsg:       tc->cnt_net_write++;    tc->cat_direct++; break;
        case __NR_rename:        tc->cnt_rename++;       tc->cat_direct++; break;
        case __NR_renameat:
#ifdef __NR_renameat2
        case __NR_renameat2:
#endif
            tc->cnt_rename++; tc->cat_alias++; break;
        case __NR_epoll_wait:    tc->cnt_epoll_wait++;   tc->cat_direct++; break;
        case __NR_epoll_pwait:   tc->cnt_epoll_wait++;   tc->cat_alias++;  break;
        case __NR__llseek:       tc->cnt_llseek++;       tc->cat_direct++; break;
        case __NR_brk:           tc->cnt_brk++;          tc->cat_direct++; break;
        case __NR_write:         tc->cnt_write++;        tc->cat_direct++; break;
        case __NR_madvise:       tc->cnt_madvise++;      tc->cat_direct++; break;
        case __NR_gettimeofday:  tc->cnt_gettimeofday++; tc->cat_direct++; break;
        case __NR_lseek:         tc->cnt_lseek++;        tc->cat_direct++; break;
        case __NR_prctl:         tc->cnt_prctl++;        tc->cat_direct++; break;
        case __NR_pipe:          tc->cnt_fs_pipe++; tc->cnt_pipe++; tc->cat_direct++; break;
        case __NR_pipe2:         tc->cnt_fs_pipe++; tc->cnt_pipe++; tc->cat_alias++;  break;
        case __NR_gettid:        tc->cnt_gettid++;       tc->cat_direct++; break;
        case __NR_epoll_ctl:     tc->cnt_epoll_ctl++;    tc->cat_direct++; break;
        case __NR_getdents64:    tc->cnt_getdents64++;   tc->cat_direct++; break;
        case __NR_execve:
        case __NR_execveat:      tc->cnt_execve++;       tc->cat_other++;  break;
        case __NR_pwrite64:      tc->cnt_pwrite64++;     tc->cat_other++;  break;
        case __NR_readlink:
        case __NR_readlinkat:    tc->cnt_readlink++;     tc->cat_other++;  break;
        case __NR_clone:         if (args[0] & CLONE_THREAD) tc->cnt_clone_thread++; tc->cat_other++; break;
        case __NR_open:
        case __NR_openat: {
            unsigned long flags = (nr == __NR_open) ? args[1] : args[2];
            if (failed) { tc->cnt_fs_fail++; tc->cat_open_fail++; }
            else {
                tc->cnt_open_ok++; tc->cat_open_ok++;
                if (flags & O_CREAT) tc->cnt_fs_create++;
                else if ((flags & O_ACCMODE) == O_RDONLY) tc->cnt_fs_read++;
            }
            break;
        }
        default: tc->cat_other++; break;
    }
}

static void vectorize_v7(const PtraceCounts *tc, double *x) {
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

#define MAX_TIDS 512
typedef struct {
    pid_t tid;
    int active;
    int started;
    int in_syscall;
    int sc_nr;
    unsigned long args[6];
} TidState;

static TidState tids[MAX_TIDS];
static TidState *get_tid_state(pid_t tid, int create) {
    int free_i = -1;
    for (int i = 0; i < MAX_TIDS; ++i) {
        if (tids[i].active && tids[i].tid == tid) return &tids[i];
        if (!tids[i].active && free_i < 0) free_i = i;
    }
    if (create && free_i >= 0) {
        memset(&tids[free_i], 0, sizeof(TidState));
        tids[free_i].tid = tid;
        tids[free_i].active = 1;
        return &tids[free_i];
    }
    return NULL;
}

static int run_untraced(char *const argv[]) {
    pid_t pid = fork();
    if (pid == 0) {
        int devnull = open("/dev/null", O_WRONLY);
        if (devnull >= 0) { dup2(devnull, 1); dup2(devnull, 2); close(devnull); }
        execvp(argv[0], argv);
        _exit(127);
    }
    int st = 0;
    waitpid(pid, &st, 0);
    return WIFEXITED(st) ? WEXITSTATUS(st) : -1;
}

static int run_traced(char *const argv[], PtraceCounts *tc) {
    memset(tc, 0, sizeof(*tc));
    memset(tids, 0, sizeof(tids));
    pid_t root = fork();
    if (root == 0) {
        int devnull = open("/dev/null", O_WRONLY);
        if (devnull >= 0) { dup2(devnull, 1); dup2(devnull, 2); close(devnull); }
        ptrace(PTRACE_TRACEME, 0, NULL, NULL);
        raise(SIGSTOP);
        execvp(argv[0], argv);
        _exit(127);
    }
    int st = 0;
    if (waitpid(root, &st, 0) <= 0) return -1;
    unsigned long opts = PTRACE_O_TRACESYSGOOD | PTRACE_O_TRACECLONE |
                         PTRACE_O_TRACEFORK | PTRACE_O_TRACEVFORK | PTRACE_O_TRACEEXEC;
    ptrace(PTRACE_SETOPTIONS, root, 0, (void *)opts);
    TidState *rs = get_tid_state(root, 1);
    rs->started = 1;
    ptrace(PTRACE_SYSCALL, root, 0, 0);

    pid_t pid;
    while ((pid = waitpid(-1, &st, __WALL)) > 0) {
        if (WIFEXITED(st) || WIFSIGNALED(st)) {
            TidState *ts = get_tid_state(pid, 0);
            if (ts) ts->active = 0;
            continue;
        }
        if (!WIFSTOPPED(st)) continue;
        int sig = WSTOPSIG(st);
        unsigned int ev = (unsigned int)(st >> 16);
        TidState *ts = get_tid_state(pid, 1);

        if (sig == SIGSTOP && !ts->started) {
            ts->started = 1;
            ptrace(PTRACE_SYSCALL, pid, 0, 0);
            continue;
        }
        if (ev == PTRACE_EVENT_CLONE || ev == PTRACE_EVENT_FORK || ev == PTRACE_EVENT_VFORK) {
            unsigned long new_pid = 0;
            ptrace(PTRACE_GETEVENTMSG, pid, 0, &new_pid);
            if (new_pid > 0) get_tid_state((pid_t)new_pid, 1);
            ptrace(PTRACE_SYSCALL, pid, 0, 0);
            continue;
        }
        if (ev == PTRACE_EVENT_EXEC) {
            /* execve entry already occurred; keep in_syscall=1 for its exit stop */
            ptrace(PTRACE_SYSCALL, pid, 0, 0);
            continue;
        }
        if (sig == (SIGTRAP | 0x80)) {
            unsigned long uregs[18];
            struct iovec iov = { uregs, sizeof(uregs) };
            if (ptrace(PTRACE_GETREGSET, pid, (void *)(uintptr_t)NT_PRSTATUS, &iov) == 0) {
                if (!ts->in_syscall) {
                    ts->sc_nr = (int)uregs[7];
                    for (int a = 0; a < 6; ++a) ts->args[a] = uregs[a];
                    ts->in_syscall = 1;
                    if (ts->sc_nr == __NR_exit || ts->sc_nr == __NR_exit_group) {
                        record_syscall(tc, ts->sc_nr, ts->args, 0);
                    }
                } else {
                    long ret = (long)uregs[0];
                    record_syscall(tc, ts->sc_nr, ts->args, ret);
                    ts->in_syscall = 0;
                }
            }
            ptrace(PTRACE_SYSCALL, pid, 0, 0);
            continue;
        }
        ptrace(PTRACE_SYSCALL, pid, 0, (void *)(uintptr_t)((sig == SIGTRAP) ? 0 : sig));
    }
    return 0;
}

static int cmp_d(const void *a, const void *b) { double da=*(const double*)a, db=*(const double*)b; return (da>db)-(da<db); }
static void pstats(const char *lbl, double *arr, int n) {
    double *s = malloc(sizeof(double)*n); memcpy(s, arr, sizeof(double)*n); qsort(s, n, sizeof(double), cmp_d);
    double sum=0, sq=0; for(int i=0;i<n;++i) sum+=arr[i]; double m=sum/n;
    for(int i=0;i<n;++i) sq+=(arr[i]-m)*(arr[i]-m); double sd=(n>1)?sqrt(sq/(n-1)):0.0;
    printf("  %-22s: mean=%7.3f ± %6.3f ms | P50=%7.3f | P95=%7.3f | P99=%7.3f\n",
           lbl, m, sd, s[(int)(0.50*(n-1))], s[(int)(0.95*(n-1))], s[(int)(0.99*(n-1))]);
    free(s);
}

static void bench_cmd(const char *label, char *const argv[], int warm, int reps) {
    double *t_unt = malloc(sizeof(double)*reps), *t_cap = malloc(sizeof(double)*reps);
    double *t_vec = malloc(sizeof(double)*reps), *t_prd = malloc(sizeof(double)*reps);
    double *t_shp = malloc(sizeof(double)*reps), *t_det = malloc(sizeof(double)*reps), *t_e2e = malloc(sizeof(double)*reps);
    PtraceCounts tc; double x[NUM_FEATURES], probs[NUM_CLASSES], raw[NUM_CLASSES], phi[NUM_FEATURES*NUM_CLASSES];
    int pcls = 0; volatile double sink = 0.0; struct timespec a, b, c, d, e;

    for (int w = 0; w < warm; ++w) {
        run_untraced(argv); run_traced(argv, &tc);
        vectorize_v7(&tc, x); predict_sample(x, probs, &pcls, raw); compute_treeshap(x, phi);
        sink += probs[0] + phi[0];
    }
    double sc_sum = 0.0, act_sum = 0.0; PtraceCounts last_tc; double last_probs[NUM_CLASSES]; int last_pcls = 0;
    for (int r = 0; r < reps; ++r) {
        if ((r & 1) == 0) {
            clock_gettime(CLOCK_MONOTONIC, &a); run_untraced(argv); clock_gettime(CLOCK_MONOTONIC, &b);
            t_unt[r] = (b.tv_sec-a.tv_sec)*1e3 + (b.tv_nsec-a.tv_nsec)/1e6;
            clock_gettime(CLOCK_MONOTONIC, &a); run_traced(argv, &tc); clock_gettime(CLOCK_MONOTONIC, &b);
        } else {
            clock_gettime(CLOCK_MONOTONIC, &a); run_traced(argv, &tc); clock_gettime(CLOCK_MONOTONIC, &b);
            clock_gettime(CLOCK_MONOTONIC, &c); run_untraced(argv); clock_gettime(CLOCK_MONOTONIC, &d);
            t_unt[r] = (d.tv_sec-c.tv_sec)*1e3 + (d.tv_nsec-c.tv_nsec)/1e6;
        }
        double cap_ms = (b.tv_sec-a.tv_sec)*1e3 + (b.tv_nsec-a.tv_nsec)/1e6;
        clock_gettime(CLOCK_MONOTONIC, &b); vectorize_v7(&tc, x); clock_gettime(CLOCK_MONOTONIC, &c);
        predict_sample(x, probs, &pcls, raw); clock_gettime(CLOCK_MONOTONIC, &d);
        compute_treeshap(x, phi); clock_gettime(CLOCK_MONOTONIC, &e);
        sink += probs[0] + phi[0];
        t_cap[r] = cap_ms;
        t_vec[r] = (c.tv_sec-b.tv_sec)*1e3 + (c.tv_nsec-b.tv_nsec)/1e6;
        t_prd[r] = (d.tv_sec-c.tv_sec)*1e3 + (d.tv_nsec-c.tv_nsec)/1e6;
        t_shp[r] = (e.tv_sec-d.tv_sec)*1e3 + (e.tv_nsec-d.tv_nsec)/1e6;
        t_det[r] = t_vec[r] + t_prd[r] + t_shp[r];
        t_e2e[r] = cap_ms + t_det[r];
        sc_sum += tc.syscalls;
        int act = 0; for (int i=0; i<NUM_FEATURES; ++i) if (x[i] > 0.0) act++;
        act_sum += act; last_tc = tc; memcpy(last_probs, probs, sizeof(probs)); last_pcls = pcls;
    }
    double m_unt=0, m_cap=0; for(int i=0;i<reps;++i){m_unt+=t_unt[i]; m_cap+=t_cap[i];}
    m_unt/=reps; m_cap/=reps;
    double mean_sc = sc_sum / reps, ovh = m_cap - m_unt, us_sc = (ovh * 1000.0) / mean_sc;
    long psum = last_tc.cat_direct + last_tc.cat_alias + last_tc.cat_open_ok + last_tc.cat_open_fail + last_tc.cat_other;
    printf("\n=== [%s] (reps=%d, warm=%d) ===\n", label, reps, warm);
    printf("  Mean Syscalls=%.1f | Active=%.1f/37 | Last pred=C%d (P=%.4f)\n", mean_sc, act_sum/reps, last_pcls, last_probs[last_pcls-1]);
    printf("  Last Partition (sum=%ld/%ld): dir=%ld + alias=%ld + openOK=%ld + openFail=%ld + other=%ld\n",
           psum, last_tc.syscalls, last_tc.cat_direct, last_tc.cat_alias, last_tc.cat_open_ok, last_tc.cat_open_fail, last_tc.cat_other);
    printf("  Incremental ptrace overhead: %.3f ms (%.2f us/syscall)\n", ovh, us_sc);
    pstats("Untraced Session",     t_unt, reps);
    pstats("Traced Capture",       t_cap, reps);
    pstats("Vec (in-register)",    t_vec, reps);
    pstats("Pred",                 t_prd, reps);
    pstats("TreeSHAP",             t_shp, reps);
    pstats("Post-Capture Detect",  t_det, reps);
    pstats("E2E Traced Total",     t_e2e, reps);
    free(t_unt); free(t_cap); free(t_vec); free(t_prd); free(t_shp); free(t_det); free(t_e2e);
    if (sink < -1e100) fprintf(stderr, "sink\n");
}

int main(int argc, char **argv) {
    int mode = (argc > 1) ? atoi(argv[1]) : 0;
    char *app_argv[] = {"./benign_app_session_v7", NULL};
    char *tar_argv[] = {"tar", "-czf", "/data/data/com.termux/files/usr/tmp/v7_pt_tar.tgz",
                        "v7_fast_xmal_top60_ordered.txt", "v7_kernel_binder_partition.json", NULL};
    if (mode == 0) {
        bench_cmd("benign_app_session_v7 (AUDIT)", app_argv, 1, 3);
        bench_cmd("tar -czf (w/ gzip)", tar_argv, 2, 10);
    } else {
        bench_cmd("benign_app_session_v7 (N=1000)", app_argv, 50, mode);
    }
    unlink("/data/data/com.termux/files/usr/tmp/v7_pt_tar.tgz");
    struct rusage ru; getrusage(RUSAGE_SELF, &ru);
    printf("\nProcess Peak RSS: %ld KB (%.2f MB)\n", ru.ru_maxrss, ru.ru_maxrss / 1024.0);
    return 0;
}
