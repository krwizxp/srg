#define _POSIX_C_SOURCE 200809L
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char **argv) {
    unsigned bits = argc == 2 ? (unsigned)strtoul(argv[1], NULL, 10) : 16;
    if (bits != 4 && bits != 8 && bits != 16) return 2;
    uint32_t n = UINT32_C(1) << bits;
    uint64_t visited = 0, accepted = 0, mainland = 0, island_a = 0, island_b = 0;
    uint64_t checksum = 0, square_checksum = 0, xor_checksum = 0;
    struct timespec before, after;
    clock_gettime(CLOCK_MONOTONIC, &before);
    for (uint32_t y = 0; y < n; ++y) {
        uint32_t cy = 2*y + 1;
        uint64_t prefix = (uint64_t)y * n;
        for (uint32_t x = 0; x < n; ++x) {
            uint32_t cx = 2*x + 1;
            uint64_t word = prefix + x;
            uint64_t main = (cx <= 3*n/2 && cy <= 3*n/2)
                && !(cx > n/2 && cx < n && cy > n/2 && cy < n);
            uint64_t a = cx >= 7*n/4 && cy >= 3*n/2;
            uint64_t b = cx >= 7*n/4 && cx <= 15*n/8
                && cy >= n/4 && cy <= 3*n/8;
            uint64_t inside = main | a | b;
            ++visited;
            accepted += inside;
            mainland += main;
            island_a += a;
            island_b += b;
            uint64_t selected = inside ? word : 0;
            checksum += selected;
            square_checksum += selected * selected;
            xor_checksum ^= selected;
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &after);
    double seconds = (after.tv_sec - before.tv_sec) + (after.tv_nsec - before.tv_nsec)*1e-9;
    printf("{\"axis_bits\":%u,\"word_bits\":%u,\"visited_candidates\":%" PRIu64
           ",\"accepted_candidates\":%" PRIu64 ",\"rejected_candidates\":%" PRIu64
           ",\"component_counts\":[%" PRIu64 ",%" PRIu64 ",%" PRIu64
           "],\"accepted_word_sum_mod_2_64\":\"%" PRIu64
           "\",\"accepted_word_square_sum_mod_2_64\":\"%" PRIu64
           "\",\"accepted_word_xor\":\"%" PRIu64
           "\",\"elapsed_seconds\":%.9f}\n",
           bits, bits*2, visited, accepted, visited-accepted, mainland, island_a, island_b,
           checksum, square_checksum, xor_checksum, seconds);
    return visited == (uint64_t)n*n && mainland+island_a+island_b == accepted ? 0 : 3;
}
