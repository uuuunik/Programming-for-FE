# Programming-for-FE
Homework group project of CU IEOR E4741

# Team member:
Chloe(Keyi) Zhang UNI kz2557

# Build Instruction

<details>
<summary> # Project 1: High Performance Linear Algebra Kernels </summary>

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

* **Row-major (C/C++ default):**

  * Consecutive elements of a row are contiguous in memory.
  * Efficient for accessing rows sequentially.
* **Column-major (Fortran/Matlab):**

  * Consecutive elements of a column are contiguous.
  * Efficient for accessing columns sequentially.
* **Examples from our project**

  * **Matrix-vector multiplication (row-major):** Accessing `A[i][k]` row-wise aligned well with memory layout, resulting in good cache performance.
  * **Matrix-vector multiplication (column-major):** Each access jumped across rows (stride = number of rows), leading to poor cache locality.
  * **Matrix-matrix multiplication (naïve):** `B[k][j]` caused strided access, hurting cache performance.
  * **Matrix-matrix multiplication (transposed-B):** By transposing B, both `A[i][k]` and `B_T[j][k]` were contiguous, reducing cache misses significantly.

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
  * Blocked matrix multiplication to exploit **temporal locality**, reusing sub-blocks in cache before eviction.

---

### 4. Memory Alignment

* **What it is**

  * Ensuring data starts at memory addresses that are multiples of cache-line or SIMD boundaries (e.g., 64 bytes).
* **Why important**

  * Aligned loads/stores can be fetched in fewer instructions.
  * Misaligned accesses may span cache lines, doubling memory traffic.
* **Findings**

  * Small improvements (\~5–15%) observed for large matrices when aligned.
  * Benefits were more visible in blocked multiplication where contiguous vectorized loads were critical.

---

### 5. Compiler Optimizations (Inlining, -O Flags)

* **Inlining**

  * Reduces function call overhead.
  * Enables the compiler to optimize across function boundaries.
* **Optimization levels**

  * `-O0`: no optimization, easiest to debug but very slow.
  * `-O3`: aggressive optimization, auto-vectorization, loop unrolling.
* **Observations**

  * Our baseline functions were significantly faster at `-O3` due to vectorization.
  * Inlining `idx_row` reduced function call overhead in hot loops.
  * **Drawbacks:** Aggressive inlining can increase binary size (code bloat) and sometimes reduce instruction-cache efficiency.

---

### 6. Profiling and Bottlenecks

* **Initial bottlenecks**

  * In the naïve implementation, most time was spent in index calculation (`idx_row`) and strided memory access to B.
* **Guidance from profiling**

  * Profiling showed \~70%+ of runtime in `idx_row`, proving memory access dominated computation.
  * This motivated the transposed-B and blocked GEMM optimizations, which reduced indexing overhead and improved cache reuse.

---

### 7. Teamwork Reflection

* **Division of tasks**

  * Each member implemented one baseline kernel (row-major MV, column-major MV, naïve MM, transposed-B MM).
  * Allowed parallel progress and ensured shared responsibility.
* **Collaboration**

  * Jointly analyzed performance, cache locality, and profiling results.
  * Brainstormed optimization strategies (e.g., blocking, alignment).
* **Challenges**

  * Ensuring consistent coding style and correct memory management across implementations.
  * Integrating benchmarking results into a unified framework.
* **Benefits**

  * Specialization allowed each member to dive deep into one kernel.
  * Group discussions brought diverse insights into optimization strategies.
</details>
