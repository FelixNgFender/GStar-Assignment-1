"""
FlashAttention-2 Forward Pass — Educational Walkthrough
========================================================

This is a readable, small-scale implementation showing the core ideas:
- Tiling: Q split into row blocks, K/V into column blocks
- Online softmax: incremental max/sum tracking (no full S matrix in HBM)
- SRAM-resident computation: only Oi, mi, li live on-chip per Q-block
"""

import torch
import math


def flash_attention_2_forward(Q, K, V, block_size_q=2, block_size_kv=2):
    """
    Educational FlashAttention-2 forward pass.

    Args:
        Q: (N, d) query matrix
        K: (N, d) key matrix
        V: (N, d) value matrix
        block_size_q: Br (rows of Q per block)
        block_size_kv: Bc (rows of K/V per block)

    Returns:
        O: (N, d) output
        L: (N,) log-sum-exp (for backward pass)
    """
    N, d = Q.shape
    assert K.shape == (N, d) and V.shape == (N, d)

    # Number of blocks
    Tr = (N + block_size_q - 1) // block_size_q  # Q blocks (row tiles)
    Tc = (N + block_size_kv - 1) // block_size_kv  # K/V blocks (column tiles)

    O = torch.zeros_like(Q)
    L = torch.zeros(N)

    print("=== FlashAttention-2 Forward ===")
    print(f"N={N}, d={d}, Br={block_size_q}, Bc={block_size_kv}")
    print(f"Tr={Tr} Q-blocks, Tc={Tc} K/V-blocks\n")

    for i in range(Tr):
        # --- 1. Load Qi (Br x d) into SRAM ---
        q_start = i * block_size_q
        q_end = min(q_start + block_size_q, N)
        Qi = Q[q_start:q_end]  # (Br, d)
        Br_actual = Qi.shape[0]

        # --- 2. Initialize on-chip accumulators ---
        Oi = torch.zeros(Br_actual, d)  # output accumulator
        mi = torch.full((Br_actual,), -float("inf"))  # running row max
        li = torch.zeros(Br_actual)  # running row sum of exp

        print(f"--- Q-block {i} (rows {q_start}:{q_end}) ---")
        print(f"  Initial: mi={mi.tolist()}, li={li.tolist()}")

        for j in range(Tc):
            # --- 3. Load Kj, Vj (Bc x d) into SRAM ---
            k_start = j * block_size_kv
            k_end = min(k_start + block_size_kv, N)
            Kj = K[k_start:k_end]  # (Bc, d)
            Vj = V[k_start:k_end]  # (Bc, d)
            Bc_actual = Kj.shape[0]

            # --- 4. Compute Sij = Qi @ Kj^T (Br x Bc) ---
            Sij = Qi @ Kj.T / math.sqrt(d)  # (Br, Bc)

            # --- 5. Online softmax update ---
            #    m_new = max(m_old, rowmax(Sij))
            #    l_new = l_old * exp(m_old - m_new) + rowsum(exp(Sij - m_new))
            #    Oi = exp(m_old - m_new) * Oi + exp(Sij - m_new) @ Vj

            mij = Sij.max(dim=1).values  # rowmax(Sij) -> (Br,)
            m_new = torch.maximum(mi, mij)  # (Br,)

            # exp(m_old - m_new): scaling factor for old accumulator
            alpha = torch.exp(mi - m_new)  # (Br,)

            # Pij_tilde = exp(Sij - m_new)  (broadcast m_new over columns)
            Pij_tilde = torch.exp(Sij - m_new.unsqueeze(1))  # (Br, Bc)

            l_new = alpha * li + Pij_tilde.sum(dim=1)  # (Br,)

            # Update Oi: scale old + add new contribution
            Oi = alpha.unsqueeze(1) * Oi + Pij_tilde @ Vj  # (Br, d)

            # Commit new running stats
            mi = m_new
            li = l_new

            print(
                f"  KV-block {j} (rows {k_start}:{k_end}): "
                f"mij={mij.tolist()}, mi->{mi.tolist()}, li->{li.tolist()}"
            )

        # --- 6. Final normalization: Oi = Oi / li ---
        Oi = Oi / li.unsqueeze(1)  # (Br, d)

        # --- 7. Log-sum-exp for backward: Li = mi + log(li) ---
        Li = mi + torch.log(li)

        # --- 8. Write back to HBM ---
        O[q_start:q_end] = Oi
        L[q_start:q_end] = Li

        print(f"  Final Oi:\n{Oi}")
        print(f"  Final Li: {Li.tolist()}\n")

    return O, L


# ============================================================
# Tiny concrete example you can trace by hand
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(42)

    # N=4 tokens, d=2 (tiny so you can verify manually)
    Q = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 1.0]])
    K = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0], [0.0, 0.5]])
    V = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])

    print("Input Q:\n", Q)
    print("Input K:\n", K)
    print("Input V:\n", V)
    print()

    # Br=2, Bc=2 -> Tr=2, Tc=2
    O_fa2, L_fa2 = flash_attention_2_forward(Q, K, V, block_size_q=2, block_size_kv=2)

    # Reference: naive full attention
    S_full = Q @ K.T / math.sqrt(2)
    P_full = torch.softmax(S_full, dim=-1)
    O_ref = P_full @ V
    L_ref = torch.logsumexp(S_full, dim=-1)

    print("=== Verification ===")
    print("FlashAttention-2 O:\n", O_fa2)
    print("Reference O:\n", O_ref)
    print("Match:", torch.allclose(O_fa2, O_ref, atol=1e-5))
    print()
    print("FlashAttention-2 L:", L_fa2.tolist())
    print("Reference L:", L_ref.tolist())
    print("Match:", torch.allclose(L_fa2, L_ref, atol=1e-5))

