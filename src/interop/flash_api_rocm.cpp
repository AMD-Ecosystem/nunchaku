#include "torch.h"
#include "flash_api_rocm.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace {

// A view of a (possibly strided) device tensor as an at::Tensor, without a copy.
at::Tensor view_as_torch(const Tensor &input) {
    std::vector<int64_t> sizes, strides;
    for (size_t i = 0; i < input.ndims(); i++) {
        sizes.push_back(input.size(i));
        strides.push_back(input.stride(i));
    }
    const at::ScalarType type = input.scalar_type() == Tensor::BF16 ? at::ScalarType::BFloat16 : at::ScalarType::Half;
    if (input.scalar_type() != Tensor::BF16 && input.scalar_type() != Tensor::FP16) {
        throw std::runtime_error("attention supports only fp16 and bf16 inputs");
    }
    auto opts = c10::TensorOptions(type).device(c10::Device(c10::DeviceType::CUDA, input.device().idx));
    return torch::from_blob(const_cast<void *>(input.data_ptr()), sizes, strides, opts);
}

void check_supported(float p_dropout, int window_size_left, int window_size_right, bool return_softmax) {
    if (p_dropout != 0.0f || window_size_left != -1 || window_size_right != -1 || return_softmax) {
        throw std::runtime_error("attention on ROCm supports no dropout, sliding window or returned softmax");
    }
}

// q: [B, Sq, H, D], k and v: [B, Sk, H, D]; returns [B, Sq, H, D]
at::Tensor attention(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, float scale, bool is_causal) {
    at::Tensor out = at::scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), {}, 0.0, is_causal, scale);
    return out.transpose(1, 2);
}

} // namespace

std::vector<Tensor> mha_fwd(Tensor &q,
                            Tensor &k,
                            Tensor &v,
                            const float p_dropout,
                            const float softmax_scale,
                            bool is_causal,
                            int window_size_left,
                            int window_size_right,
                            const bool return_softmax) {
    check_supported(p_dropout, window_size_left, window_size_right, return_softmax);
    c10::cuda::CUDAStreamGuard guard(c10::cuda::getStreamFromExternal(getCurrentCUDAStream(), q.device().idx));

    at::Tensor out =
        attention(view_as_torch(q), view_as_torch(k), view_as_torch(v), softmax_scale, is_causal).contiguous();
    return {from_torch(out)};
}

std::vector<Tensor> mha_varlen_fwd(Tensor &q,
                                   Tensor &k,
                                   Tensor &v,
                                   Tensor &cu_seqlens_q,
                                   Tensor &cu_seqlens_k,
                                   int max_seqlen_q,
                                   const int max_seqlen_k,
                                   const float p_dropout,
                                   const float softmax_scale,
                                   const bool zero_tensors,
                                   bool is_causal,
                                   int window_size_left,
                                   int window_size_right,
                                   const bool return_softmax) {
    check_supported(p_dropout, window_size_left, window_size_right, return_softmax);
    c10::cuda::CUDAStreamGuard guard(c10::cuda::getStreamFromExternal(getCurrentCUDAStream(), q.device().idx));

    const at::Tensor tq = view_as_torch(q);
    const at::Tensor tk = view_as_torch(k);
    const at::Tensor tv = view_as_torch(v);

    auto offsets = [](Tensor &cu_seqlens) {
        at::Tensor t = torch::from_blob(cu_seqlens.data_ptr(),
                                        {(int64_t)cu_seqlens.numel()},
                                        c10::TensorOptions(at::ScalarType::Int)
                                            .device(c10::Device(c10::DeviceType::CUDA, cu_seqlens.device().idx)));
        return t.cpu();
    };
    const at::Tensor offsets_q = offsets(cu_seqlens_q);
    const at::Tensor offsets_k = offsets(cu_seqlens_k);
    const int batch_size       = (int)offsets_q.numel() - 1;

    at::Tensor out = zero_tensors ? at::zeros_like(tq, at::MemoryFormat::Contiguous)
                                  : at::empty_like(tq, at::MemoryFormat::Contiguous);
    for (int i = 0; i < batch_size; i++) {
        const int64_t q0 = offsets_q[i].item<int>(), q1 = offsets_q[i + 1].item<int>();
        const int64_t k0 = offsets_k[i].item<int>(), k1 = offsets_k[i + 1].item<int>();
        out.slice(0, q0, q1).copy_(attention(tq.slice(0, q0, q1).unsqueeze(0),
                                             tk.slice(0, k0, k1).unsqueeze(0),
                                             tv.slice(0, k0, k1).unsqueeze(0),
                                             softmax_scale,
                                             is_causal)
                                       .squeeze(0));
    }
    return {from_torch(out)};
}

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
                                  int window_size_right) {
    throw std::runtime_error("Block-sparse attention is not supported on ROCm");
}
