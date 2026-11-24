#pragma once

#include "Tensor.h"

// The attention entry points of Block-Sparse-Attention's flash_api.h, for ROCm builds, where that library is
// not built. Dense attention runs through torch's scaled_dot_product_attention; block-sparse attention is not
// supported.

std::vector<Tensor> mha_fwd(Tensor &q, // batch_size x seqlen_q x num_heads x head_size
                            Tensor &k, // batch_size x seqlen_k x num_heads_k x head_size
                            Tensor &v, // batch_size x seqlen_k x num_heads_k x head_size
                            const float p_dropout,
                            const float softmax_scale,
                            bool is_causal,
                            int window_size_left,
                            int window_size_right,
                            const bool return_softmax);

std::vector<Tensor> mha_varlen_fwd(Tensor &q,            // total_q x num_heads x head_size
                                   Tensor &k,            // total_k x num_heads_k x head_size
                                   Tensor &v,            // total_k x num_heads_k x head_size
                                   Tensor &cu_seqlens_q, // b+1
                                   Tensor &cu_seqlens_k, // b+1
                                   int max_seqlen_q,
                                   const int max_seqlen_k,
                                   const float p_dropout,
                                   const float softmax_scale,
                                   const bool zero_tensors,
                                   bool is_causal,
                                   int window_size_left,
                                   int window_size_right,
                                   const bool return_softmax);

std::vector<Tensor> mha_fwd_block(const Tensor &q,
                                  const Tensor &k,
                                  const Tensor &v,
                                  const Tensor &cu_seqlens_q,
                                  const Tensor &cu_seqlens_k,
                                  const int m_block_dim,
                                  const int n_block_dim,
                                  Tensor &head_mask_type,
                                  std::optional<Tensor> streaming_info_,
                                  std::optional<Tensor> row_blockmask_,
                                  const int max_seqlen_q_,
                                  const int max_seqlen_k_,
                                  const float p_dropout,
                                  const float softmax_scale,
                                  const bool is_causal,
                                  const bool exact_streaming,
                                  const bool return_softmax,
                                  int window_size_left,
                                  int window_size_right);
