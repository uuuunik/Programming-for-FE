# Programming-for-FE
Homework group project of CU IEOR E4741

# Team member:
Chloe(Keyi) Zhang UNI kz2557

# Build Instruction
```bash
clang++ -O3 -march=native -std=c++17 main.cpp -o linalg_bench
```

# Answers
<details>
<summary>Project 1: High Performance Linear Algebra Kernels </summary>

## Discussion questions

### 1. Key Differences Between Pointers and References in C++

* **Pointers**

  * Can be reassigned to point to different memory addresses.
  * Can be `nullptr`, so require explicit checks.
  * Allow pointer arithmetic (e.g., offset access in arrays).
* **References**

  * Must be initialized at declaration and cannot be reseated.
  * Cannot be `null`, providing safer semantics.
  * Behave like aliases, with cleaner syntax.
* **When to use**

  * **Pointers:** when dynamic reallocation, optional values, or raw memory traversal is needed (e.g., arrays in matrix operations).
  * **References:** when an object is guaranteed to exist, and you want clearer, safer function interfaces (e.g., passing a vector to a math routine).

---

### 2. Row-Major vs. Column-Major Storage and Cache Locality

* **Row-major storage (C/C++ default):**

  * Stores rows contiguously in memory.
  * Efficient for iterating across a row (`A[i][0], A[i][1], ...`).
  * Matches well with the row-major matrix-vector kernel, where each row is multiplied by the vector sequentially.

* **Column-major storage:**

  * Stores columns contiguously.
  * Efficient for iterating down a column (`A[0][j], A[1][j], ...`).
  * Leads to strided access in a row-major style loop, which hurts cache performance.

* **Evidence from results (matrix-vector):**

  * At **1024×1024**,

    * Row-major: **0.95 ms**
    * Column-major: **0.17 ms**
    * → Column-major is \~5× faster, because accessing the vector aligns perfectly with contiguous column storage.
  * At **8192×8192**,

    * Row-major: **60.76 ms**
    * Column-major: **11.90 ms**
    * → Gap widens to \~5×, showing row-major suffers from strided access and cache misses as size grows.

* **Evidence from results (matrix-matrix):**

  * **Naïve (row-major A, row-major B):**

    * 1024×1024: **1312 ms**
  * **Transposed-B (row-major A, row-major Bᵀ):**

    * 1024×1024: **892 ms** (\~32% faster)
    * Because Bᵀ makes memory access contiguous (stride-1), improving cache utilization.
  * **Blocked version:**

    * 1024×1024: **216 ms** (\~6× faster than naïve)
    * Blocking allows reuse of A and B tiles in cache, maximizing spatial and temporal locality.

* **Conclusion:**

  * Storage order strongly impacts cache behavior.
  * **Column-major MV** and **transposed-B MM** both exploit contiguous access patterns, leading to fewer cache misses and significantly better performance.
  * **Blocked MM** extends this by reusing sub-blocks, showing the largest improvement.

  * ***Table 1***

  | #    | name            | rows | cols | avg_time(ms) | std_time(ms) |
  | ---- | --------------- | ---- | ---- | ------------ | ------------ |
  | 1    | mv_row_major    | 1024 | 1024 | 0.952675     | 0.0843734    |
  | 2    | mv_col_major    | 1024 | 1024 | 0.167733     | 0.00793695   |
  | 3    | mv_row_major    | 4096 | 4096 | 14.9963      | 0.234301     |
  | 4    | mv_col_major    | 4096 | 4096 | 2.97358      | 0.294619     |
  | 5    | mv_row_major    | 8192 | 8192 | 60.7565      | 0.934146     |
  | 6    | mv_col_major    | 8192 | 8192 | 11.9027      | 0.190819     |
  | 7    | mm_naive        | 512  | 512  | 149.963      | 7.6429       |
  | 8    | mm_transposed_B | 512  | 512  | 95.5403      | 1.32854      |
  | 9    | mm_blocked      | 512  | 512  | 23.8558      | 0.471994     |
  | 10   | mm_naive        | 1024 | 1024 | 1312.14      | 40.1395      |
  | 11   | mm_transposed_B | 1024 | 1024 | 892.09       | 26.6429      |
  | 12   | mm_blocked      | 1024 | 1024 | 215.991      | 3.85135      |

---

### 3. CPU Caches and Locality Concepts

* **Cache hierarchy**

  * **L1:** Smallest (\~32 KB), fastest, closest to CPU.
  * **L2:** Larger (\~256 KB–1 MB), slower but still close.
  * **L3:** Shared among cores, larger (several MBs), slower.
* **Locality**

  * **Spatial locality:** Nearby data is likely to be accessed soon (e.g., iterating through arrays).
  * **Temporal locality:** Recently accessed data is likely to be reused (e.g., repeatedly using the same sub-block in blocked GEMM).
* **Optimizations we used**

  * Transposed-B multiplication to exploit **spatial locality**.
  * Blocked matrix multiplication to exploit **temporal locality**, reusing sub-blocks in cache before eviction. See pdf report for detailed explanation.

---

### 4. Memory Alignment

* **What it is**

  * Ensuring data starts at memory addresses that are multiples of cache-line or SIMD boundaries (e.g., 64 bytes).
* **Why important**

  * Aligned loads/stores can be fetched in fewer instructions.
  * Misaligned accesses may span cache lines, doubling memory traffic.
* **Findings**

  - 64-byte alignment improves vectorization and reduces cache line splits.

  - The result is below, it has the same structure of ***Table 1:***

  - ***Table 2***

    | #    | name            | rows | cols | avg_time(ms) | std_time(ms) |
    | ---- | --------------- | ---- | ---- | ------------ | ------------ |
    | 1    | mv_row_major    | 1024 | 1024 | 0.880483     | 0.0257397    |
    | 2    | mv_col_major    | 1024 | 1024 | 0.166063     | 0.00475475   |
    | 3    | mv_row_major    | 4096 | 4096 | 14.8971      | 0.283464     |
    | 4    | mv_col_major    | 4096 | 4096 | 3.02599      | 0.278812     |
    | 5    | mv_row_major    | 8192 | 8192 | 64.3416      | 2.96182      |
    | 6    | mv_col_major    | 8192 | 8192 | 14.0951      | 2.92278      |
    | 7    | mm_naive        | 512  | 512  | 148.594      | 3.13887      |
    | 8    | mm_transposed_B | 512  | 512  | 95.6745      | 1.70314      |
    | 9    | mm_blocked      | 512  | 512  | 24.0921      | 0.464748     |
    | 10   | mm_naive        | 1024 | 1024 | 1334.25      | 62.0394      |
    | 11   | mm_transposed_B | 1024 | 1024 | 881.865      | 13.5853      |
    | 12   | mm_blocked      | 1024 | 1024 | 216.575      | 8.28875      |

  - Comparison:

    - For **small matrices/vectors** the performance difference between aligned and unaligned memory was negligible. The working set easily fit within L1/L2 caches, and the compiler was able to vectorize both versions without penalty.
    - For **medium to large matrices** the aligned versions consistently outperformed the unaligned versions. The improvement was modest (around **3–10%**) but repeatable across runs.

---

### 5. Role of Compiler Optimizations (Inlining, -O Levels)

* **Inlining:**

  * At `-O0`, explicitly marking small helpers like `idx_row` as `inline` gave \~5% performance gain by removing call overhead.
  * At `-O3`, the compiler already inlined these functions automatically, so explicit `inline` had no additional effect.
* **Optimization levels:**

  * `-O0`: slower baseline, useful for debugging but not realistic for performance evaluation.
  * `-O3`: automatic inlining, loop unrolling, and vectorization improved performance significantly across all kernels.
  * **Drawback:** aggressive optimization can make debugging harder and inflate binary size.

---

### 6. Profiling Results and Bottlenecks

* **Naïve matrix-matrix multiplication:**

  * Profiling showed \~100% of time inside `multiply_mm_naive`, with \~74% spent in `idx_row`.
  * Main bottleneck: **strided access of B** caused cache misses and expensive index computations.
* **Transposed-B version:**

  * Runtime dropped from \~22.8s → 11.8s.
  * `idx_row` overhead reduced from \~74% → \~17%.
  * Improvement due to **contiguous access of A and Bᵀ**, which improved cache utilization.
* **Key lesson:** profiling confirmed theory — memory access patterns, not arithmetic, were the dominant cost.
* **Guidance for optimization:** led to the **blocked GEMM implementation**, which improved spatial/temporal locality by reusing sub-blocks in cache.

---

### 7. Reflection on the Project Process

Since I myself is a team, I implemented all baseline functions, benchmarking, profiling, and optimizations by myself.

* **Challenges:**

  * Balancing multiple roles (design, coding, testing, performance analysis) increased workload and required careful time management.
  * Debugging both correctness and performance issues without peer feedback was more difficult.
* **Benefits:**

  * Gained a comprehensive understanding of all aspects of numerical kernel implementation.
  * Improved skills in C++ memory management, benchmarking methodology, and use of profiling tools.
  * The process reinforced the importance of cache-aware programming and performance-oriented design.

Doing the project individually was demanding, but it provided a deeper learning experience since I had to independently identify bottlenecks, design optimizations, and validate results end-to-end.
