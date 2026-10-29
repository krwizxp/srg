#define _GNU_SOURCE
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

int main(int argc, char **argv) {
    if (argc < 3) return 125;
    struct timespec begin, end;
    if (clock_gettime(CLOCK_MONOTONIC, &begin)) return 125;
    pid_t child = fork();
    if (child < 0) return 125;
    if (!child) {
        execv(argv[2], argv + 2);
        _exit(127);
    }
    int status;
    struct rusage usage;
    while (wait4(child, &status, 0, &usage) < 0) {
        if (errno != EINTR) return 125;
    }
    if (clock_gettime(CLOCK_MONOTONIC, &end)) return 125;
    double wall = end.tv_sec - begin.tv_sec + (end.tv_nsec - begin.tv_nsec) / 1e9;
    int result = WIFEXITED(status) ? WEXITSTATUS(status) : -WTERMSIG(status);
    FILE *out = fopen(argv[1], "w");
    if (!out) return 125;
    fprintf(out, "{\"exit_code\":%d,\"wall_seconds\":%.9f,\"max_rss_kib\":%ld,"
                 "\"user_seconds\":%.6f,\"system_seconds\":%.6f,\"major_page_faults\":%ld,"
                 "\"minor_page_faults\":%ld,\"voluntary_context_switches\":%ld,"
                 "\"involuntary_context_switches\":%ld}\n",
            result, wall, usage.ru_maxrss,
            usage.ru_utime.tv_sec + usage.ru_utime.tv_usec / 1e6,
            usage.ru_stime.tv_sec + usage.ru_stime.tv_usec / 1e6,
            usage.ru_majflt, usage.ru_minflt, usage.ru_nvcsw, usage.ru_nivcsw);
    if (fclose(out)) return 125;
    return result < 0 ? 128 - result : result;
}
