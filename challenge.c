
// src/q15_axpy_challenge.c
// Single-solution RVV challenge: Q15 y = a + alpha * b  (saturating to Q15)
//

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// -------------------- Scalar reference (no intrinsics) --------------------
static inline int16_t sat_q15_scalar(int32_t v)
{
    if (v > 32767)
        return 32767;
    if (v < -32768)
        return -32768;
    return (int16_t)v;
}

void q15_axpy_ref(const int16_t *a, const int16_t *b,
                  int16_t *y, int n, int16_t alpha)
{
    for (int i = 0; i < n; ++i)
    {
        int32_t acc = (int32_t)a[i] + (int32_t)alpha * (int32_t)b[i];
        y[i] = sat_q15_scalar(acc);
    }
}

// The threshold acceptable; my latest manually compiled gcc build from https://github.com/riscv-collab/riscv-gnu-toolchain is 12000
#define RVV_SPEC_THRESHOLD 1

// -------------------- RVV include per ratified v1.0 spec ------------------
#if __riscv_v_intrinsic >= RVV_SPEC_THRESHOLD
#include <riscv_vector.h> // v1.0 test macro & header inclusion
#endif

// -------------------- RVV implementation (mentees edit only here) ---------
void q15_axpy_rvv(const int16_t *a, const int16_t *b,
                  int16_t *y, int n, int16_t alpha)
{
#if !defined(__riscv) || !defined(__riscv_vector) || (__riscv_v_intrinsic < RVV_SPEC_THRESHOLD)
    // Fallback (keeps correctness off-target)
    q15_axpy_ref(a, b, y, n, alpha);
#else
    // TODO: Enter your solution here

    /*
    Goal, compute forumula:

    y[i] = sat_q15(a[i] + alpha * b[i])

    - a, b: int16_t input arrays
    - y: int16_t output array

    - alpha: int16_t scalar

    - intermediate calculations done using int32_t RISC-V Vectors (RVV)
    - result saturated/clamped to int16_t vectors

    */

    // baseline software engineering practice: give readable meaningful names to variables
    int elementCount = n;

    // hardware vector length (VL)
    size_t vectorLength = 0;

    // convet alpha to 32bit because interim vector will hold int32_t elements
    int32_t scalar = (int32_t)alpha;

    // process in batches, depending on the hardware vector length (VL) of fitting int16_t elements.
    for (size_t elementsProcessed = 0; elementsProcessed < elementCount; elementsProcessed += vectorLength)
    {

        // 1. get VL
        vectorLength = __riscv_vsetvl_e16m1(elementCount - elementsProcessed);

        // 2. load `vectorLength` int16_t elements into a vector
        vint16m1_t tempVectorA = __riscv_vle16_v_i16m1(&a[elementsProcessed], vectorLength);

        // 3. widen the int16_t elements vector to int32_t
        vint32m2_t vectorA = __riscv_vwcvt_x_x_v_i32m2(tempVectorA, vectorLength);

        // 4. and 5. do the same for array `b`
        vint16m1_t tempVectorB = __riscv_vle16_v_i16m1(&b[elementsProcessed], vectorLength);
        vint32m2_t vectorB = __riscv_vwcvt_x_x_v_i32m2(tempVectorB, vectorLength);

        // 6. scalar multiply (scalar * vectorB)
        vint32m2_t vectorMultiplied = __riscv_vmul_vx_i32m2(vectorB, scalar, vectorLength);

        // 7. sum (vectorA + vectorMultiply)
        vint32m2_t vectorSum = __riscv_vadd_vv_i32m2(vectorA, vectorMultiplied, vectorLength);

        // 8. *crucial* narrow the int32_t to int16_t by first saturating or clamping [-32768, 32767]
        // argument 2 set right shift amount: we do 0.
        // arfument 3  becomes irrelevant.
        vint16m1_t vectorY = __riscv_vnclip_wx_i16m1(vectorSum, 0, 0, vectorLength);

        // 9. store `vectorY` into array `y`
        __riscv_vse16_v_i16m1(&y[elementsProcessed], vectorY, vectorLength);
    }

#endif
}

// -------------------- Verification & tiny benchmark -----------------------
static int verify_equal(const int16_t *ref, const int16_t *test, int n, int32_t *max_diff)
{
    int ok = 1;
    int32_t md = 0;
    for (int i = 0; i < n; ++i)
    {
        int32_t d = (int32_t)ref[i] - (int32_t)test[i];
        if (d < 0)
            d = -d;
        if (d > md)
            md = d;
        if (d != 0)
            ok = 0;
    }
    *max_diff = md;
    return ok;
}

#if defined(__riscv)
static inline uint64_t rdcycle(void)
{
    uint64_t c;
    asm volatile("rdcycle %0" : "=r"(c));
    return c;
}
#endif

int main(void)
{
    int ok = 1;
    const int N = 4096;
    int16_t *a = (int16_t *)aligned_alloc(64, N * sizeof(int16_t));
    int16_t *b = (int16_t *)aligned_alloc(64, N * sizeof(int16_t));
    int16_t *y0 = (int16_t *)aligned_alloc(64, N * sizeof(int16_t));
    int16_t *y1 = (int16_t *)aligned_alloc(64, N * sizeof(int16_t));

    // Deterministic integer data (no libm)
    srand(1234);
    for (int i = 0; i < N; ++i)
    {
        a[i] = (int16_t)((rand() % 65536) - 32768);
        b[i] = (int16_t)((rand() % 65536) - 32768);
    }

    const int16_t alpha = 3; // example scalar gain

    uint32_t c0 = rdcycle();
    q15_axpy_ref(a, b, y0, N, alpha);
    uint32_t c1 = rdcycle();
    printf("Cycles ref: %u\n", c1 - c0);

    int32_t md = 0;

#if defined(__riscv)
    c0 = rdcycle();
    q15_axpy_rvv(a, b, y1, N, alpha);
    c1 = rdcycle();
    ok = verify_equal(y0, y1, N, &md);
    printf("Verify RVV: %s (max diff = %d)\n", ok ? "OK" : "FAIL", md);
    printf("Cycles RVV: %llu\n", (unsigned long long)(c1 - c0));
#endif

    free(a);
    free(b);
    free(y0);
    free(y1);
    return ok ? 0 : 1;
}
