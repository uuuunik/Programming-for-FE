#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#ifndef SIZE
#define SIZE 4096
#endif

// ======== Baseline small functions (no inline) ========
#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
int getElement(const std::vector<std::vector<int>>& m, int r, int c) {
    return m[r][c];
}
#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
int add(int a, int b) { return a + b; }

// ========== Baseline version (unoptimized) ==========
long long sumMatrixBasic(const std::vector<std::vector<int>>& matrix) {
    long long sum = 0;
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            sum = add(sum, getElement(matrix, i, j));
        }
    }
    return sum;
}

// ======== Optimized versions ========
#if defined(__GNUC__) || defined(__clang__)
  #define ALWAYS_INLINE inline __attribute__((always_inline))
#else
  #define ALWAYS_INLINE inline
#endif

// Inline helpers for optimized versions
ALWAYS_INLINE int getElementFast(const int* row, int col) {
    return row[col];
}
ALWAYS_INLINE long long addFast(long long a, int b) {
    return a + b;
}

// Optimized Version 1: row pointers + loop unrolling
// This version improves performance in several ways:
// 1. Gets direct pointer to each row's data to avoid vector[] overhead
// 2. Uses loop unrolling (8 elements at a time) to:
//    - Reduce loop control overhead
//    - Allow compiler to use SIMD instructions
//    - Improve instruction-level parallelism
// 3. Maintains good spatial locality by accessing elements sequentially within each row
long long sumMatrixOptimizedRows(const std::vector<std::vector<int>>& matrix) {
    long long sum = 0;
    for (int i = 0; i < SIZE; ++i) {
        const int* p = matrix[i].data();
        int j = 0;
        for (; j + 7 < SIZE; j += 8) {
            sum += p[0] + p[1] + p[2] + p[3]
                 + p[4] + p[5] + p[6] + p[7];
            p += 8;
        }
        for (; j < SIZE; ++j) {
            sum += *p++;
        }
    }
    return sum;
}

// Optimized Version 2: flat 1D array + pointer traversal
// This version achieves even better performance by:
// 1. Using a flat 1D array instead of 2D vector to:
//    - Eliminate row pointer indirection
//    - Ensure perfect memory contiguity
// 2. More aggressive loop unrolling (16 elements) to:
//    - Further reduce loop overhead
//    - Enable wider SIMD operations
//    - Increase instruction parallelism
// 3. Using pointer arithmetic for fastest possible memory traversal
// 4. Pre-computing end pointer to simplify bounds checking
long long sumMatrixOptimizedFlat(const std::vector<int>& flat) {
    const int* p = flat.data();
    const int* const end = p + (SIZE * SIZE);
    long long sum = 0;

    for (; p + 15 < end; p += 16) {
        sum += p[0] + p[1] + p[2] + p[3]
             + p[4] + p[5] + p[6] + p[7]
             + p[8] + p[9] + p[10] + p[11]
             + p[12] + p[13] + p[14] + p[15];
    }
    while (p < end) sum += *p++;
    return sum;
}

// ======== Data generation helpers ========
void fillFlat(std::vector<int>& flat) {
    std::mt19937 gen(123456u);
    std::uniform_int_distribution<int> dist(-100, 100);
    for (auto& v : flat) v = dist(gen);
}

std::vector<std::vector<int>> build2DFromFlat(const std::vector<int>& flat) {
    std::vector<std::vector<int>> m(SIZE, std::vector<int>(SIZE));
    for (int i = 0; i < SIZE; ++i) {
        std::copy_n(flat.data() + i * SIZE, SIZE, m[i].data());
    }
    return m;
}

// Timing helper
template <class F>
auto timeit_ms(F&& f) {
    const auto t0 = std::chrono::high_resolution_clock::now();
    auto result   = f();
    const auto t1 = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    return std::pair<decltype(result), long long>(result, ms);
}

int main() {
    // Generate test data
    std::vector<int> flat(SIZE * SIZE);
    fillFlat(flat);
    auto matrix2d = build2DFromFlat(flat);

    // Baseline
    auto [sum_basic, ms_basic] = timeit_ms([&] { return sumMatrixBasic(matrix2d); });

    // Optimized row-based
    auto [sum_opt_rows, ms_opt_rows] = timeit_ms([&] { return sumMatrixOptimizedRows(matrix2d); });

    // Optimized flat-based
    auto [sum_opt_flat, ms_opt_flat] = timeit_ms([&] { return sumMatrixOptimizedFlat(flat); });

    // Verify correctness
    if (sum_basic != sum_opt_rows || sum_basic != sum_opt_flat) {
        std::cerr << "ERROR: sums mismatch!\n";
        return 1;
    }

    std::cout << "SIZE = " << SIZE << " (" << (SIZE * SIZE) << " elements)\n\n";
    std::cout << std::left << std::setw(28) << "Basic Sum"
              << ": " << sum_basic << " | time = " << ms_basic << " ms\n";
    std::cout << std::left << std::setw(28) << "Optimized (rows+unroll)"
              << ": " << sum_opt_rows << " | time = " << ms_opt_rows << " ms\n";
    std::cout << std::left << std::setw(28) << "Optimized (flat+pointer)"
              << ": " << sum_opt_flat << " | time = " << ms_opt_flat << " ms\n";
}
