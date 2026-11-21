#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <dirent.h>
#include <pthread.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/file.h>
#include <sys/epoll.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/prctl.h>
#include <sys/uio.h>
#include <sys/wait.h>
#include <sys/syscall.h>
#include <netinet/in.h>
#include <arpa/inet.h>

static void *worker_fn(void *arg) {
    int id = *(int *)arg;
    char tname[16];
    snprintf(tname, sizeof(tname), "fx_wrk_%d", id);
    prctl(PR_SET_NAME, tname, 0, 0, 0);
    syscall(SYS_gettid);
    struct timeval tv;
    syscall(SYS_gettimeofday, &tv, NULL);
    int pfd[2];
    if (pipe2(pfd, O_NONBLOCK) == 0) {
        char msg[16] = "ping";
        write(pfd[1], msg, 4);
        read(pfd[0], msg, 4);
        close(pfd[0]);
        close(pfd[1]);
    }
    return NULL;
}

int main(void) {
    char base[128];
    snprintf(base, sizeof(base), "/data/data/com.termux/files/usr/tmp/fx_v7_%d", (int)getpid());
    mkdir(base, 0700);

    // 1. Signal mask + memory heap/mmap/mprotect/madvise/munmap
    sigset_t ss;
    sigemptyset(&ss);
    sigaddset(&ss, SIGUSR1);
    sigprocmask(SIG_BLOCK, &ss, NULL);
    sigprocmask(SIG_UNBLOCK, &ss, NULL);

    void *heap_p = malloc(64 * 1024);
    if (heap_p) { memset(heap_p, 0xA5, 64 * 1024); free(heap_p); }

    void *mp = mmap(NULL, 8192, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mp != MAP_FAILED) {
        mprotect(mp, 8192, PROT_READ);
        madvise(mp, 8192, MADV_DONTNEED);
        munmap(mp, 8192);
    }

    // 2. File I/O, pread64, writev, flock, stat/lstat, access, lseek/_llseek, rename, 5 temp cycles
    char p1[192], p2[192];
    snprintf(p1, sizeof(p1), "%s/cache_0.dat", base);
    snprintf(p2, sizeof(p2), "%s/cache_renamed.dat", base);

    for (int i = 0; i < 5; ++i) {
        char tmpf[192];
        snprintf(tmpf, sizeof(tmpf), "%s/tmp_%d.bin", base, i);
        int fd = open(tmpf, O_CREAT | O_RDWR | O_TRUNC, 0600);
        if (fd >= 0) {
            flock(fd, LOCK_EX);
            struct iovec iov[2] = { {"hdr:", 4}, {"payload_data", 12} };
            writev(fd, iov, 2);
            char buf[32];
            pread(fd, buf, 8, 4);
            syscall(SYS_lseek, fd, 0, SEEK_SET);
            loff_t res_off = 0;
            syscall(SYS__llseek, fd, 0, 0, &res_off, SEEK_END);
            flock(fd, LOCK_UN);
            close(fd);
        }
        int rfd = open(tmpf, O_RDONLY);
        if (rfd >= 0) { char rb[16]; read(rfd, rb, sizeof(rb)); close(rfd); }
        unlink(tmpf);
    }

    int fd1 = open(p1, O_CREAT | O_WRONLY | O_TRUNC, 0600);
    if (fd1 >= 0) { write(fd1, "fast_xmal_v7", 12); close(fd1); }
    rename(p1, p2);
    access(p2, R_OK);
    // Intentional missing-file probe (realistic app config check)
    int mfd = open("/data/data/com.termux/files/usr/tmp/nonexistent_fx_v7.cfg", O_RDONLY);
    if (mfd >= 0) close(mfd);

    struct stat st;
    stat(p2, &st);
    lstat(p2, &st);

    DIR *d = opendir(base);
    if (d) { while (readdir(d) != NULL) {} closedir(d); }

    // 3. epoll + select + loopback UDP/TCP sockets
    int ep = epoll_create1(0);
    int sp[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sp) == 0) {
        struct epoll_event ev = { .events = EPOLLIN, .data.fd = sp[0] };
        epoll_ctl(ep, EPOLL_CTL_ADD, sp[0], &ev);
        write(sp[1], "x", 1);
        struct epoll_event out_ev;
        epoll_wait(ep, &out_ev, 1, 5);
        fd_set rfds; FD_ZERO(&rfds); FD_SET(sp[0], &rfds);
        struct timeval sel_tv = {0, 1000};
        select(sp[0] + 1, &rfds, NULL, NULL, &sel_tv);
        char c; read(sp[0], &c, 1);
        close(sp[0]); close(sp[1]);
    }
    if (ep >= 0) close(ep);

    int usock = socket(AF_INET, SOCK_DGRAM, 0);
    int tsock = socket(AF_INET, SOCK_STREAM, 0);
    if (usock >= 0) {
        struct sockaddr_in addr = {0};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(0);
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        bind(usock, (struct sockaddr *)&addr, sizeof(addr));
        socklen_t alen = sizeof(addr);
        getsockname(usock, (struct sockaddr *)&addr, &alen);
        connect(usock, (struct sockaddr *)&addr, sizeof(addr));
        sendto(usock, "v7", 2, 0, (struct sockaddr *)&addr, sizeof(addr));
        struct sockaddr_in from = {0}; socklen_t flen = sizeof(from);
        recvfrom(usock, base, 2, MSG_DONTWAIT, (struct sockaddr *)&from, &flen);
        close(usock);
    }
    if (tsock >= 0) close(tsock);

    // 4. 3 worker pthreads + 1 short child process
    pthread_t th[3]; int ids[3] = {0, 1, 2};
    for (int i = 0; i < 3; ++i) pthread_create(&th[i], NULL, worker_fn, &ids[i]);
    for (int i = 0; i < 3; ++i) pthread_join(th[i], NULL);

    pid_t cp = fork();
    if (cp == 0) {
        char *argv_c[] = {"/system/bin/true", NULL};
        execve("/system/bin/true", argv_c, NULL);
        _exit(0);
    } else if (cp > 0) {
        int status = 0;
        waitpid(cp, &status, 0);
    }

    unlink(p2);
    rmdir(base);
    return 0;
}
