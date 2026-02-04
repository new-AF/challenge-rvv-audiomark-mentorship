# Design Choices of My Solution to the RISC-V AudioMark™ Mentorship Challenge

by Abdullah Fatota [[1]](https://github.com/new-AF/challenge-rvv-audiomark-mentorship)

## The challenge

Implement a _vectorized_ _C_ function _(`q15_axpy_rvv`)_ that uses the RISC-V Vector instructions (RVV) to compute the scalar formula [[2]](https://docs.google.com/document/d/1BLO9GU57161sGLYuBxm7MzcDJSVZIj5OYhqFli7t-Y0/edit?tab=t.0):

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

_Scalar_ means we apply a sequence operations to a **single** array element e.g.:

- `const multiplyResult = alpha * b[i]`
- `const sumResult = multiplyResult + a[i]`
- `const clampedResult = sat_q15_scalar(result)`
- `y[i] = clampedResult`

A _vectorized_ solution however applies the **same** operation on **multiple** elements **simultaneously**.

It does this by _packing_ multiple elements inside a _hardware vector register_ and applying the _same_ operation on the _entire_ vector e.g.:

- `const multiplyVector = alpha * vectorB[i : i+VL]` where `VL` is `vectorLength` or the number of elements that can be _packed_ in a single hardware vector register.
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
    Goal, compute the formula:

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

    // convert alpha to 32bit because interim vector will hold int32_t elements
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
        // argument 2 sets right shift amount: we do 0.
        // argument 3  becomes irrelevant.
        vint16m1_t vectorY = __riscv_vnclip_wx_i16m1(vectorSum, 0, 0, vectorLength);

        // 9. store `vectorY` into array `y`
        __riscv_vse16_v_i16m1(&y[elementsProcessed], vectorY, vectorLength);
    }

#endif
}
```

### Correctness output

![Correctness output screenshot of running the vectorized ELF binary on the QEMU risc-v 32bit emulator](./correctness-output.PNG)

I ran my solution on QEMU as following:

```bash
qemu-riscv32 ./challenge.elf
```

### Walkthrough

> The crux of the solution is: We process the arrays (`a`, `b` and `y`) in **batches** of `vectorLength` size.

`vectorLength` is **dynamic** and depends on the underlying RISC-V CPU. At the start of each batch we call `__riscv_vsetvl_e16m1` to get our `vectorLength`, we always be greedy and request the full remaining array length `(elementCount - elementsProcessed)` to fit inside a vector register.

We then get the _actual_ length of the available vector that will accommodate our elements, which is often less than the full `elementCount`

`__riscv_vsetvl_e16m1` works as the following pseudo-code:

```c
size_t __riscv_vsetvl_e16m1(requestedVectorLength) {
    return min(requestedVectorLength, VLMAX)
}
```

If our input arrays are `4096` elements wide, and `vectorLength` remains consistent at `10` for whatever reason, then we do `409` batches where we process `10` elements, and _one_ final run where we process the remaining `6` elements.

`__riscv_vsetvl_e16m1` ensures we only need a _single_ `for` loop, making our implementation fully **vector-length agnostic**.

During each batch we do the same for arrays `a` and `b`:

- Load `vectorLength` chunk of the array into a vector, by calling `__riscv_vle16_v_i16m1`.

- Widen `tempVectorA` elements to `int32_t` in preparation for subsequent calculations. We increase the element width but keep the same `vectorLength`. This is done by calling `__riscv_vwcvt_x_x_v_i32m2` which increases our `LMUL` from `1` to `2`.

We perform the formula-part calculations:

- Scalar-vector calculations `alpha * vectorB` by calling `__riscv_vmul_vx_i32m2`

- Vector-vector calculations `scaledVectorB + vectorA` by calling `__riscv_vadd_vv_i32m2`

Next we narrow down the elements from `int32_t` to `int16_t`

- We do this by calling `__riscv_vnclip_wx_i16m1` to first saturate or clip the `int32_t` value into `int16_t` range. The crucial part is the second argument it tells the instruction to `0` right shifts, the 3rd argument then becomes irrelevant.

Finally we store the nascent `int16_t` vector into the corresponding chunk of the output array `y`

- We call `__riscv_vse16_v_i16m1` to specify the address of the `y` chunk and the length of `vectorY`
  y

## Building and running my solution

I opted to build the latest official GCC toolchain provided by RISC-V International themselves [[5]](https://github.com/riscv-collab/riscv-gnu-toolchain).

It takes around 6.5GB disk space once `make` is done cloning the GitHub repo. Building itself takes around 6 hours on my weak laptop.

Here are the commands I used to build `riscv32-unknown-elf-gcc` on my _Linux Mint 22.3_ installation and successfully compile `./challenge.c` with both the reference and my vectorized solution:

```bash
cd ~
mkdir build-here-riscv

# Install the userland QEMU emulator for RISC-V 32
sudo apt install qemu-user

# Install the dependencies for GCC
sudo apt-get install autoconf automake autotools-dev curl python3 python3-pip python3-tomli libmpc-dev libmpfr-dev libgmp-dev gawk build-essential bison flex texinfo gperf libtool patchutils bc zlib1g-dev libexpat-dev ninja-build git cmake libglib2.0-dev libslirp-dev libncurses-dev

git clone https://github.com/riscv-collab/riscv-gnu-toolchain
cd riscv-gnu-toolchain
./configure --prefix=/home/YOUR_USERNAME/build-here-riscv --with-arch=rv32gcv --with-abi=ilp32d

# This will take around 6.5GB disk space and 6 hours on a laptop
make

# So you can run `riscv32-unknown-elf-gcc` from anywhere from the terminal
echo 'export PATH=/home/af/build-here-riscv/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# Confirm the Vector instructions were installed
# You should see for example
#define __riscv_v 1000000
echo | riscv32-unknown-elf-gcc -march=rv32gcv -mabi=ilp32d -dM -E - | grep __riscv_v

# clone my solution
cd ~
git clone https://github.com/new-AF/challenge-rvv-audiomark-mentorship
cd challenge-rvv-audiomark-mentorship

# compile it
riscv32-unknown-elf-gcc -O2 -march=rv32gcv -mabi=ilp32d -o challenge.elf challenge.c

# run it
qemu-riscv32 ./challenge.elf
```

After running `qemu-riscv32 ./challenge.elf` you should see an output similar to below:

![Correctness output screenshot of running the vectorized ELF binary on the QEMU risc-v 32bit emulator](./correctness-output.PNG)

## References

[[1] https://github.com/new-AF/challenge-rvv-audiomark-mentorship](https://github.com/new-AF/challenge-rvv-audiomark-mentorship)

[[2] https://docs.google.com/document/d/1BLO9GU57161sGLYuBxm7MzcDJSVZIj5OYhqFli7t-Y0/edit?tab=t.0](https://docs.google.com/document/d/1BLO9GU57161sGLYuBxm7MzcDJSVZIj5OYhqFli7t-Y0/edit?tab=t.0)

[[3] https://research.samsung.com/blog/Bringing-RVV-to-Life-Overcoming-Hardware-Gaps-in-RISC-V-Development](https://research.samsung.com/blog/Bringing-RVV-to-Life-Overcoming-Hardware-Gaps-in-RISC-V-Development)

[[4] https://blog.talli.ai/intel-cpu-security-flaw-settlement/](https://blog.talli.ai/intel-cpu-security-flaw-settlement/)

[[5] https://github.com/riscv-collab/riscv-gnu-toolchain](https://github.com/riscv-collab/riscv-gnu-toolchain)
