import torch
import triton
import triton.language as tl


@triton.jit
def row_sum_kernel(X_ptr, output_ptr, M, D, BLOCK_SIZE_D: tl.constexpr):
    row_idx = tl.program_id(axis=0)  # row_idx (0, M-1)
    row_start_ptr = X_ptr + D * row_idx
    accumulator = 0.0
    for d_offset in range(0, tl.cdiv(D, BLOCK_SIZE_D)):
        d_offsets = d_offset * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
        d_mask = d_offsets < D
        x_tile = tl.load(row_start_ptr + d_offsets, mask=d_mask, other=0.0)
        accumulator += tl.sum(x_tile)
    tl.store(output_ptr + row_idx, accumulator)


if __name__ == "__main__":
    M, D, BLOCK_SIZE_D = 4, 5, 2
    X = torch.randn((M, D), device="cuda")
    output = torch.empty(M, device=X.device, dtype=X.dtype)
    grid = (M,)
    row_sum_kernel[grid](X, output, M, D, BLOCK_SIZE_D)
    print(X)
    print(output)
    torch_output = X.sum(dim=-1)
    print(torch_output)
    assert torch.allclose(output, torch_output)
