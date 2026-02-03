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

## Speedup

> The expected speedup is between **1.6x** for a 32-bit vector register, up to **25.6x** for a 512-bit vector register.
>
> The speedup formula is **`0.05 x Vector Register Width`**
>
> Because our elements are `int16_t`, the expected speedup is **`0.8 x VL`** or **`0.8 x vectorLength`**
>
> _(`vectorLength` is the number of `int16_t` elements that can be packed in a vector register)_

### Backstory

![Correctness output screenshot of running the vectorized ELF binary on the QEMU risc-v 32bit emulator](./correctness-output.PNG)

I ran the compiled _ELF_ on `qemu-riscv32` and even though the cycle counter shows the RVV solution being slower than the scalar option, this is because `qemu` is not a cycle-accurate emulator and does not emulate the RVV hardware pipeline. It instead transforms the RVV instructions into individual scalar instructions while being functionally correct with the end results and RVV spec.

I tried to compile `gem5` twice hoping it might have a better RVV performance but my hardware-bugged and weak Intel Gen11 i3 laptop became unresponsive.

So we have to disassemble the binary outputs of both the scalar and vectorized versions to calculate the instruction count _per_ element.

### Scalar solution disassembly analysis

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

```c
void q15_axpy_ref(const int16_t *a, const int16_t *b,
                  int16_t *y, int n, int16_t alpha)
{
    for (int i = 0; i < n; ++i)
    {
        int32_t acc = (int32_t)a[i] + (int32_t)alpha * (int32_t)b[i];
        y[i] = sat_q15_scalar(acc);
    }
}
```

Let's ignore the initializations and focus on the `for` loop because that's where the majority of execution time is spent, from address `102f8` .

> The CPU runs **12 instructions** per **single** element when there's no positive or negative saturation, when the interim `int32_t` value is within the `int16_t` range. These are:

```
1. 102f8: lh a5,0(a1)        // load b[0]
2. 102fc: lh t1,0(a0)        // load a[0]
3. 10300: addi a6,a7,-1      // a6 = 0x7fff
4. 10304: mul a5,a5,a4       // b[0] * alpha
5. 10308: add a5,a5,t1       // result + a[i]
6. 1030a: bge a5,a7,10314    // not taken as result within int16_t range
7. 1030e: blt a5,t3,10324    // same not taken
8. 10312: mv a6,a5           // a6 = result
9. 10314: sh a6,0(a2)        // y[0] = a6
10. 10318: addi a0,a0,2      // a++
11. 1031a: addi a1,a1,2      // b++
12. 1031c: addi a2,a2,2      // y++
```

In case of positive saturation it's 10 instructions per single element, in case of negative saturation it's 13 instructions.

### Vectorized solution disassembly analysis

```
00010328 <q15_axpy_rvv>:
q15_axpy_rvv():
   10328:	00a05073          	csrwi	vxrm,0
   1032c:	c6a1                	beqz	a3,10374 <q15_axpy_rvv+0x4c>
   1032e:	4881                	li	a7,0
   10330:	00189813          	slli	a6,a7,0x1
   10334:	411687b3          	sub	a5,a3,a7
   10338:	0c87f7d7          	vsetvli	a5,a5,e16,m1,ta,ma
   1033c:	01058333          	add	t1,a1,a6
   10340:	02035107          	vle16.v	v2,(t1)
   10344:	01050333          	add	t1,a0,a6
   10348:	02035087          	vle16.v	v1,(t1)
   1034c:	9832                	add	a6,a6,a2
   1034e:	98be                	add	a7,a7,a5
   10350:	c6206257          	vwcvt.x.x.v	v4,v2
   10354:	c6106157          	vwcvt.x.x.v	v2,v1
   10358:	0d107057          	vsetvli	zero,zero,e32,m2,ta,ma
   1035c:	96476257          	vmul.vx	v4,v4,a4
   10360:	02220157          	vadd.vv	v2,v2,v4
   10364:	0c807057          	vsetvli	zero,zero,e16,m1,ta,ma
   10368:	be203157          	vnclip.wi	v2,v2,0
   1036c:	02085127          	vse16.v	v2,(a6)
   10370:	fcd8e0e3          	bltu	a7,a3,10330 <q15_axpy_rvv+0x8>
   10374:	8082                	ret
```

> The CPU runs **15 instructions** per the **entire** `int16_t` vector of size `vectorLength`.

```
1. 10330: slli a6,a7,0x1              // byteOffset = elementsProcessed * 2
2. 10334: sub  a5,a3,a7               // remaining = n - elementsProcessed
3. 10338: vsetvli a5,a5,e16,m1        // vectorLength = min(remaining, VLMAX)

4. 1033c: add  t1,a1,a6               // get &b[elementsProcessed]
5. 10340: vle16.v v2,(t1)             // tempVectorA = b[elementsProcessed : elementsProcessed+vectorLength] into  tempVectorA

6. 10344: add  t1,a0,a6               // get &a[elementsProcessed]
7. 10348: vle16.v v1,(t1)             // load a[...] into vectorA

8. 10350: vwcvt.x.x.v v4,v2           // vectorA = widen tempVectorA from int16 to int32
9. 10354: vwcvt.x.x.v v2,v1           // vectorB = widen tempVectorA

10. 10358: vsetvli zero,zero,e32,m2    // set LMUL = 2; switch to int32 vectors

11. 1035c: vmul.vx v4,v4,a4            // vectorMultiplied = vectorB * alpha
12. 10360: vadd.vv v2,v2,v4            // vectorSum = vectorMultiplied + vectorA

13. 10364: vsetvli zero,zero,e16,m1    // set LMUL = 1; switch back to int16 vectors

14. 10368: vnclip.wi v2,v2,0           // vectorY = saturate and narrow vectorSum elements from int32 to int16

15. 1036c: vse16.v v2,(a6)             //  y[elementsProcessed : elementsProcessed+vectorLength] = vectorY
```

So 1 element costs `15 / vectorLength` instructions in the vectorized solution, as opposed to `12` in the scalar solution.

The Speedup = `Scalar instructions per element / Vector instructions per element`

The Speedup = `12 / (15 / vectorLength)`

The Speedup = `12 * vectorLength / 15`

> The Speedup = `0.8 * vectorLength`
>
> Because `vectorLength` = `vectorWidthInBits / 16`
>
> The Speedup in terms of Vector Register Width = `0.05 * vectorWidthInBits`

#### Concrete calculations:

| `vectorLength` | Vector Register Width (bits) | Speedup |
| -------------- | ---------------------------- | ------- |
| 1              | 16                           | 0.8     |
| 2              | 32                           | 1.6     |
| 3              | 48                           | 2.4     |
| 4              | 64                           | 3.2     |
| 5              | 80                           | 4       |
| 6              | 96                           | 4.8     |
| 7              | 112                          | 5.6     |
| 8              | 128                          | 6.4     |
| 9              | 144                          | 7.2     |
| 10             | 160                          | 8       |
| 11             | 176                          | 8.8     |
| 12             | 192                          | 9.6     |
| 13             | 208                          | 10.4    |
| 14             | 224                          | 11.2    |
| 15             | 240                          | 12      |
| 16             | 256                          | 12.8    |
| 17             | 272                          | 13.6    |
| 18             | 288                          | 14.4    |
| 19             | 304                          | 15.2    |
| 20             | 320                          | 16      |
| 21             | 336                          | 16.8    |
| 22             | 352                          | 17.6    |
| 23             | 368                          | 18.4    |
| 24             | 384                          | 19.2    |
| 25             | 400                          | 20      |
| 26             | 416                          | 20.8    |
| 27             | 432                          | 21.6    |
| 28             | 448                          | 22.4    |
| 29             | 464                          | 23.2    |
| 30             | 480                          | 24      |
| 31             | 496                          | 24.8    |
| 32             | 512                          | 25.6    |

![Plot of the speedup Vector Register Width (bits) on x-axis, Speedup on Y-axis](./speedup-1.png)

## Building and running my solution

I opted to build the [latest official GCC toolchain provided by RISC-V International themselves](https://github.com/riscv-collab/riscv-gnu-toolchain).

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
