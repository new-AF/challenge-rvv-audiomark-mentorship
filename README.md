# My Solution to the RISC-V Audiomark mentorship challenge

## Challenge

Implement a C function `q15_axpy_rvv` using RISC-V Vector (RVV) intrinsics.

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

### Formula

`q15_axpy_rvv` computes the formula:

```
y[i] = sat_q15(a[i] + alpha * b[i])
```

- Interim calculations will need to be `int32_t`. This is because `a[i] + alpha * b[i]` exceeds the `int16_t` range `[-32768, 32767]` and will overflow to the wrong values.
- `sat_q15` then clamps the output back to `int16_t`.

    > `sat_q15` needs **not be implemented** as the proper RVV intrinsics will do that internally , it's provided here purely for educational purposes:

    ```c
    int16_t sat_q15_core(int32_t arrayElement)
    {
        if (arrayElement > 32767)
            return 32767;
        if (arrayElement < -32768)
            return -32768;
        return (int16_t)arrayElement;
    }
    ```

## The provided scalar solution

Pretty straightforward:

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

## Building the toolchain

I opted to build the [latest official GCC toolchain provided by RISC-V International themselves](https://github.com/riscv-collab/riscv-gnu-toolchain)
