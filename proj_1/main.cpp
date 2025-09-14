#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <cstdlib>
using namespace std;

// ========================= Utility: index helpers =========================
static inline size_t idx_row(size_t i, size_t j, size_t cols) noexcept {
    return i * cols + j;           // row-major
}
static inline size_t idx_col(size_t i, size_t j, size_t rows) noexcept {
    return j * rows + i;           // column-major contiguous
}

// ========================= Aligned allocation (64B) =======================
void* aligned_malloc64(size_t size) {
#if defined(_MSC_VER)
    return _aligned_malloc(size, 64);
#elif defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
    size_t align = 64;
    size_t rem = size % align;
    if (rem) size += (align - rem);
    return aligned_alloc(64, size);
#elif defined(_POSIX_VERSION)
    void* p = nullptr;
    if (posix_memalign(&p, 64, size) != 0) return nullptr;
    return p;
#else
    return ::operator new(size, std::nothrow);
#endif
}
void aligned_free64(void* p) {
#if defined(_MSC_VER)
    _aligned_free(p);
#else
    free(p);
#endif
}

// ========================= Error helpers ==================================
#define REQUIRE(cond, msg) do { if(!(cond)) { cerr << "Error: " << msg << "\n"; exit(1);} } while(0)

// ========================= Baseline Functions =============================
// Team Member 1: MV (Row-Major)
void multiply_mv_row_major(const double* matrix, int rows, int cols,
                           const double* vec, double* res) {
    if (!matrix || !vec || !res) return;
    for (int i = 0; i < rows; ++i) {
        double sum = 0.0;
        const double* rowp = matrix + (size_t)i * cols;
        for (int j = 0; j < cols; ++j) {
            sum += rowp[j] * vec[j];
        }
        res[i] = sum;
    }
}

// Team Member 2: MV (Column-Major)
void multiply_mv_col_major(const double* matrix, int rows, int cols,
                           const double* vec, double* res) {
    if (!matrix || !vec || !res) return;
    for (int i = 0; i < rows; ++i) res[i] = 0.0;
    for (int j = 0; j < cols; ++j) {
        const double vj = vec[j];
        const double* colp = matrix + (size_t)j * rows;
        for (int i = 0; i < rows; ++i) {
            res[i] += colp[i] * vj;
        }
    }
}

// Team Member 3: MM (Naive, row-major)
void multiply_mm_naive(const double* A, int rA, int cA,
                       const double* B, int rB, int cB,
                       double* C) {
    if (!A || !B || !C) return;
    if (cA != rB) return;
    for (int i = 0; i < rA; ++i) {
        for (int j = 0; j < cB; ++j) {
            double sum = 0.0;
            for (int k = 0; k < cA; ++k) {
                sum += A[idx_row(i,k,cA)] * B[idx_row(k,j,cB)];
            }
            C[idx_row(i,j,cB)] = sum;
        }
    }
}

// Team Member 4: MM (Transposed B, row-major)
void multiply_mm_transposed_b(const double* A, int rA, int cA,
                              const double* BT, int rB, int cB,
                              double* C) {
    if (!A || !BT || !C) return;
    if (cA != rB) return;
    for (int i = 0; i < rA; ++i) {
        for (int j = 0; j < cB; ++j) {
            double sum = 0.0;
            const double* arow = A + (size_t)i * cA;
            const double* btrow = BT + (size_t)j * rB;
            for (int k = 0; k < cA; ++k) sum += arow[k] * btrow[k];
            C[idx_row(i,j,cB)] = sum;
        }
    }
}

// ========================= Optimized Example: Blocked GEMM ===============
void multiply_mm_blocked(const double* A, int rA, int cA,
                         const double* B, int rB, int cB,
                         double* C, int BS=128) {
    if (!A || !B || !C) return;
    if (cA != rB) return;
    for (int i = 0; i < rA; ++i)
        for (int j = 0; j < cB; ++j)
            C[idx_row(i,j,cB)] = 0.0;

    for (int ii = 0; ii < rA; ii += BS) {
        int iimax = min(ii + BS, rA);
        for (int kk = 0; kk < cA; kk += BS) {
            int kkmax = min(kk + BS, cA);
            for (int jj = 0; jj < cB; jj += BS) {
                int jjmax = min(jj + BS, cB);
                for (int i = ii; i < iimax; ++i) {
                    for (int k = kk; k < kkmax; ++k) {
                        double aik = A[idx_row(i,k,cA)];
                        const double* brow = B + (size_t)k * cB;
                        double* crow = C + (size_t)i * cB;
                        for (int j = jj; j < jjmax; ++j) {
                            crow[j] += aik * brow[j];
                        }
                    }
                }
            }
        }
    }
}

// ========================= Correctness Tests =============================
bool almost_equal(double a, double b, double eps=1e-9) {
    return fabs(a-b) <= eps * (1.0 + max(fabs(a), fabs(b)));
}
void test_small() {
    // MV Row-Major
    {
        int r=2,c=3;
        double M[] = { 1,2,3, 4,5,6 };
        double v[] = { 1,1,1 };
        double res[2] = {0,0};
        multiply_mv_row_major(M,r,c,v,res);
        REQUIRE(almost_equal(res[0], 6.0), "MV row-major test failed");
        REQUIRE(almost_equal(res[1], 15.0), "MV row-major test failed");
    }
    // MV Column-Major
    {
        int r=2,c=3;
        double Mcol[] = { 1,4, 2,5, 3,6 };
        double v[] = { 1,1,1 };
        double res[2] = {0,0};
        multiply_mv_col_major(Mcol,r,c,v,res);
        REQUIRE(almost_equal(res[0], 6.0), "MV col-major test failed");
        REQUIRE(almost_equal(res[1], 15.0), "MV col-major test failed");
    }
    // MM Naive vs Transposed-B
    {
        int rA=2,cA=3,rB=3,cB=2;
        double A[] = {1,2,3, 4,5,6};
        double B[] = {7,8, 9,10, 11,12};
        double C1[4], C2[4];

        multiply_mm_naive(A,rA,cA,B,rB,cB,C1);

        vector<double> BT((size_t)cB*rB);
        for (int i=0;i<rB;++i)
            for (int j=0;j<cB;++j)
                BT[idx_row(j,i,rB)] = B[idx_row(i,j,cB)];
        multiply_mm_transposed_b(A,rA,cA,BT.data(),rB,cB,C2);

        for (int i=0;i<rA;i++)
            for (int j=0;j<cB;j++)
                REQUIRE(almost_equal(C1[idx_row(i,j,cB)], C2[idx_row(i,j,cB)]),
                        "MM mismatch");

        REQUIRE(almost_equal(C1[0],58), "MM value check failed");
        REQUIRE(almost_equal(C1[1],64), "MM value check failed");
        REQUIRE(almost_equal(C1[2],139),"MM value check failed");
        REQUIRE(almost_equal(C1[3],154),"MM value check failed");
    }
    cerr << "[Tests] All small-size tests passed.\n";
}

// ========================= Benchmark Framework ===========================
struct Stats { double avg_ms{}, std_ms{}; };

template<class F>
Stats bench(const string& name, F&& fn, int warmup, int runs) {
    for (int i=0;i<warmup;++i) fn();

    vector<double> ms;
    for (int i=0;i<runs;++i) {
        auto t0 = chrono::steady_clock::now();
        fn();
        auto t1 = chrono::steady_clock::now();
        ms.push_back(chrono::duration<double, std::milli>(t1 - t0).count());
    }
    double mean = accumulate(ms.begin(), ms.end(), 0.0) / runs;
    double var=0.0; for (double v: ms) var += (v-mean)*(v-mean); var /= max(1,runs-1);
    double sd = sqrt(var);

    cout << setw(26) << left << name
         << " avg(ms)=" << setw(10) << mean
         << " std(ms)=" << sd << "\n";
    return {mean, sd};
}

// ========================= Random Fill =========================
void fill_rand(double* p, size_t n, unsigned seed=42) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (size_t i=0;i<n;++i) p[i] = dist(rng);
}

// ========================= Main ==========================================
int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int rows = 1024, cols = 1024;
    int mv_rows = 1<<14, mv_cols = 256;
    int runs = 10, warmup = 3;
    bool aligned = true;
    bool run_blocked = true;
    int block = 128;
    bool only_naive_mm = false;
    bool only_transposed_mm = false;

    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--unaligned") aligned = false;
        else if (arg == "--n" && i+1 < argc) rows = cols = stoi(argv[++i]);
        else if (arg == "--mv_rows" && i+1 < argc) mv_rows = stoi(argv[++i]);
        else if (arg == "--mv_cols" && i+1 < argc) mv_cols = stoi(argv[++i]);
        else if (arg == "--runs" && i+1 < argc) runs = stoi(argv[++i]);
        else if (arg == "--warmup" && i+1 < argc) warmup = stoi(argv[++i]);
        else if (arg == "--block" && i+1 < argc) block = stoi(argv[++i]);
        else if (arg == "--only_naive_mm") only_naive_mm = true;
        else if (arg == "--only_transposed_mm") only_transposed_mm = true;
    }

    if (only_naive_mm){
        int rA = rows, cA = cols, rB = cols, cB = rows;
        size_t nA=(size_t)rA*cA, nB=(size_t)rB*cB, nC=(size_t)rA*cB;
        
        auto alloc = [&](size_t n)->double*{
            if (aligned) return (double*)aligned_malloc64(n*sizeof(double));
            return (double*)malloc(n*sizeof(double));
        };
        auto dealloc = [&](void* p){ if (aligned) aligned_free64(p); else free(p); };

        double *A=alloc(nA), *B=alloc(nB), *C=alloc(nC);
        REQUIRE(A && B && C, "MM: allocation failed");
        fill_rand(A, nA, 123);
        fill_rand(B, nB, 456);

        vector<double> BT((size_t)cB*rB);
        for (int i=0;i<rB;++i)
            for (int j=0;j<cB;++j)
                BT[idx_row(j,i,rB)] = B[idx_row(i,j,cB)];

        cout << "\n[MM] n=" << rows << " aligned=" << (aligned?"yes":"no") << "\n";
        
        bench("mm_naive", [&]{ multiply_mm_naive(A,rA,cA,B,rB,cB,C); }, warmup, runs);
        dealloc(A); dealloc(B); dealloc(C);
        return 0;
    }

    if (only_transposed_mm){
        int rA = rows, cA = cols, rB = cols, cB = rows;
        size_t nA=(size_t)rA*cA, nB=(size_t)rB*cB, nC=(size_t)rA*cB;
        
        auto alloc = [&](size_t n)->double*{
            if (aligned) return (double*)aligned_malloc64(n*sizeof(double));
            return (double*)malloc(n*sizeof(double));
        };
        auto dealloc = [&](void* p){ if (aligned) aligned_free64(p); else free(p); };

        double *A=alloc(nA), *B=alloc(nB), *C=alloc(nC);
        REQUIRE(A && B && C, "MM: allocation failed");
        fill_rand(A, nA, 123);
        fill_rand(B, nB, 456);

        vector<double> BT((size_t)cB*rB);
        for (int i=0;i<rB;++i)
            for (int j=0;j<cB;++j)
                BT[idx_row(j,i,rB)] = B[idx_row(i,j,cB)];

        cout << "\n[MM] n=" << rows << " aligned=" << (aligned?"yes":"no") << "\n";
        
        bench("mm_transposed_B", [&]{ multiply_mm_transposed_b(A,rA,cA,BT.data(),rB,cB,C); }, warmup, runs);

        dealloc(A); dealloc(B); dealloc(C);
        return 0;
    }

    test_small();

    // Test sizes
    vector<pair<int,int>> mv_sizes = {
        {1024, 1024},     // Small square
        {4096, 4096},     // Medium square  
        {8192, 8192},     // Large square
        {16384, 256},     // Tall-skinny
        {256, 16384}      // Short-wide
    };

    vector<int> mm_sizes = {512, 1024}; // Square matrices

    // Benchmark MV for different sizes
    for (const auto& size_pair : mv_sizes) {
        int mvr = size_pair.first;
        int mvc = size_pair.second;
        auto alloc = [&](size_t n)->double*{
            if (aligned) return (double*)aligned_malloc64(n*sizeof(double));
            return (double*)malloc(n*sizeof(double));
        };
        auto dealloc = [&](void* p){ if (aligned) aligned_free64(p); else free(p); };

        double *M_rm=alloc(mvr*mvc), *M_cm=alloc(mvr*mvc), *v=alloc(mvc), *r=alloc(mvr);
        REQUIRE(M_rm && M_cm && v && r, "MV: allocation failed");

        fill_rand(M_rm, mvr*mvc);
        for (size_t i=0;i<mvr;i++)
            for (size_t j=0;j<mvc;j++)
                M_cm[idx_col(i,j,mvr)] = M_rm[idx_row(i,j,mvc)];
        fill_rand(v, mvc);

        cout << "\n[MV] rows=" << mvr << " cols=" << mvc 
             << " aligned=" << (aligned?"yes":"no") << "\n";
             
        bench("mv_row_major", [&]{ multiply_mv_row_major(M_rm,mvr,mvc,v,r); }, warmup, runs);
        bench("mv_col_major", [&]{ multiply_mv_col_major(M_cm,mvr,mvc,v,r); }, warmup, runs);

        dealloc(M_rm); dealloc(M_cm); dealloc(v); dealloc(r);
    }

    // Benchmark MM for different sizes
    for (int n : mm_sizes) {
        int rA = n, cA = n, rB = n, cB = n;
        size_t nA=(size_t)rA*cA, nB=(size_t)rB*cB, nC=(size_t)rA*cB;
        
        auto alloc = [&](size_t n)->double*{
            if (aligned) return (double*)aligned_malloc64(n*sizeof(double));
            return (double*)malloc(n*sizeof(double));
        };
        auto dealloc = [&](void* p){ if (aligned) aligned_free64(p); else free(p); };

        double *A=alloc(nA), *B=alloc(nB), *C=alloc(nC);
        REQUIRE(A && B && C, "MM: allocation failed");
        fill_rand(A, nA, 123);
        fill_rand(B, nB, 456);

        vector<double> BT((size_t)cB*rB);
        for (int i=0;i<rB;++i)
            for (int j=0;j<cB;++j)
                BT[idx_row(j,i,rB)] = B[idx_row(i,j,cB)];

        cout << "\n[MM] n=" << n << " aligned=" << (aligned?"yes":"no") << "\n";
        
        bench("mm_naive", [&]{ multiply_mm_naive(A,rA,cA,B,rB,cB,C); }, warmup, runs);
        bench("mm_transposed_B", [&]{ multiply_mm_transposed_b(A,rA,cA,BT.data(),rB,cB,C); }, warmup, runs);

        if (run_blocked) {
            bench("mm_blocked", [&]{ multiply_mm_blocked(A,rA,cA,B,rB,cB,C,block); }, warmup, runs);
        }

        dealloc(A); dealloc(B); dealloc(C);
    }

    cout << "\nDone.\n";
    return 0;
}
