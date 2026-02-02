# My Solution to the RISC-V AudioMark™ Mentorship Challenge

by Abdullah Fatota
[https://github.com/new-AF/challenge-rvv-audiomark-mentorship](https://github.com/new-AF/challenge-rvv-audiomark-mentorship)

## The challenge

Implement a _vectorized_ _C_ function _(`q15_axpy_rvv`)_ that uses the RISC-V Vector instructions (RVV) to compute the scalar formula:

```
y[i] = sat_q15(a[i] + alpha * b[i])
```

### Function prototype

```c
void q15_axpy_rvv(
    const int16_t *a,
    const int16_t *b,
    int16_t *y,
    int n,
    int16_t alpha
)
```

- `a` and `b` are input arrays, of `int16_t` elements.
- `y` is the output array, of `int16_t` elements.
- `n` is the length of `a`, `b` and `y`.
- `alpha` is a singular (scalar) value.

### Scalar vs vectorized definitions

_Scalar_ means we apply a sequence operations on the **single** element e.g.:

- `const multiplyResult = alpha * b[i]`
- `const sumResult = multiplyResult + a[i]`
- `const clampedResult = sat_q15_scalar(result)`
- `y[i] = clampedResult`

A _vectorized_ solution however applies the **same** operation on **multiple** elements **simultaneously**.

It does this by _packing_ multiple elements inside a _hardware vector register_ and applying the _same_ operation on the _entire_ vector e.g.:

- `const multiplyVector = alpha * vectorB[i : i+VL]` where `VL` is the number of elements _packable_ in a single hardware vector register.
- `const sumVector = multiplyVector + vectorA[i : i+VL]`
- `const clampedAndNarrowedVector = __riscv_vnclip_wx_i16m1(sumVector)`
- `y[i : i+VL] = clampedAndNarrowedVector`

### Formula notes

```
y[i] = sat_q15(a[i] + alpha * b[i])
```

- Interim element calculations will be of type `int32_t` (doesn't matter if _scalar_ or _vectorized_).
- This is because `a[i] + alpha * b[i]` will in most cases exceed the `int16_t` range `[-32768, 32767]` and overflow to the wrong value.
- `sat_q15` clamps the `int32_t` result back to `int16_t`.

> _The vectorized solution should **not** implement `sat_q15` because the proper RVV instructions will do that internally. The reference scalar version is provided here purely for educational purposes:_

```c
int16_t sat_q15(int32_t arrayElement)
{
    if (arrayElement > 32767)
        return 32767;
    if (arrayElement < -32768)
        return -32768;
    return (int16_t)arrayElement;
}
```

## My vectorized solution

```c
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

    // baseline software engineering practice: give readable meaningful to variables
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

        // 8. *crucial* narrow the int32_t to int16_t by clamping [-32768, 32767]
        // argument 2 set right shift amount: we do 0.
        // arfument 3 sets the rounding mode: 0 means trucnate.
        vint16m1_t vectorY = __riscv_vnclip_wx_i16m1(vectorSum, 0, 0, vectorLength);

        // 9. store `vectorY` into array `y`
        __riscv_vse16_v_i16m1(&y[elementsProcessed], vectorY, vectorLength);
    }

#endif
}
```

### Correctness output

![Correctness output screenshot of running the vectorized ELF binary on the QEMU Simulator](./correctness-output.PNG)

### High-level overview

- We will process the `int16_t` arrays (`a`, `b` and `y`) in **batches**.

The batch size is dynamic and depends on the underlying RISC-V CPU hardware, we initially request the full array `elementCount` to fit inside a vector register. We call `__riscv_vsetvl_e16m1` with our _request_ but get the _actual_ length of the vector that will accommodate our elements, which is often less than the full `elementCount`

`__riscv_vsetvl_e16m1` works as following pseudo-code:

```c
size_t __riscv_vsetvl_e16m1(requestedVectorLength) {
    return min(requestedVectorLength, __maximumLengthAvailableVector) // e.g. 64 or 10
}
```

`__riscv_vsetvl_e16m1` ensures we only need a _single_ `for` loop

During each batch we:

- Load that `int16_t` array chunk (e.g. `a[elementsProcessed : vectorLength]`) into a vector.

- Widen the elements to `int32_t` but keep the same `vectorLength`. TH

### x

The provided scalar solution is straightforward:

## The reference scalar solution

- Loop over the input arrays and apply the formula.
- Use an interim `int32_t` variable storage.
- Clamp the `int32_t` to `int16_t` using the provided `sat_q15_scalar(arrayElement)` function.
- Store the `int16_t` value into the output array.

```c
void q15_axpy_ref(
    const int16_t *a,
    const int16_t *b,
    int16_t *y,
    int n,
    int16_t alpha
)
{
    for (int i = 0; i < n; ++i)
    {
        int32_t acc = (int32_t)a[i] + (int32_t)alpha * (int32_t)b[i];
        y[i] = sat_q15_scalar(acc);
    }
}
```

### Cost

Here's the disassembled output of the scalar solution:

```
000102ec <q15_axpy_ref>:
q15_axpy_ref():
   102ec:	02d05b63          	blez	a3,10322 <q15_axpy_ref+0x36>
   102f0:	0686                	slli	a3,a3,0x1
   102f2:	96aa                	add	a3,a3,a0
   102f4:	68a1                	lui	a7,0x8
   102f6:	7e61                	lui	t3,0xffff8
   102f8:	00059783          	lh	a5,0(a1)
   102fc:	00051303          	lh	t1,0(a0)
   10300:	fff88813          	addi	a6,a7,-1 # 7fff <exit-0x80b5>
   10304:	02e787b3          	mul	a5,a5,a4
   10308:	979a                	add	a5,a5,t1
   1030a:	0117d563          	bge	a5,a7,10314 <q15_axpy_ref+0x28>
   1030e:	01c7cb63          	blt	a5,t3,10324 <q15_axpy_ref+0x38>
   10312:	883e                	mv	a6,a5
   10314:	01061023          	sh	a6,0(a2)
   10318:	0509                	addi	a0,a0,2
   1031a:	0589                	addi	a1,a1,2
   1031c:	0609                	addi	a2,a2,2
   1031e:	fcd51de3          	bne	a0,a3,102f8 <q15_axpy_ref+0xc>
   10322:	8082                	ret
   10324:	7861                	lui	a6,0xffff8
   10326:	b7fd                	j	10314 <q15_axpy_ref+0x28>
```

## Building the toolchain

I opted to build the [latest official GCC toolchain provided by RISC-V International themselves](https://github.com/riscv-collab/riscv-gnu-toolchain)
