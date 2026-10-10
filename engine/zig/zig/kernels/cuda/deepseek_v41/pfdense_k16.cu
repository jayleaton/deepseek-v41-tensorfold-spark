// Device code of src/tensorfold/families/deepseek_v41/cuda/pfdense_k16.cu (git blob b474eefeeb26 at dsv41-quant-e2; the whole file),
// generated and checked by zig/kernels/cuda/deepseek_v41/sync.py. Do not edit: change the Python source.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jay Leaton. The DeepSeek-V4.1-Flash family of TensorFold (Apache-2.0): see THIRD_PARTY_NOTICES.md.
// pfdense (TF_DSV41_PF_DENSE=fused) at K2 = 16 (8 bits): strips / stored (the indexer's wk); no lanes kernels at 8 bits (dense3 leaves wk in strips), every configuration of
// pfdense_args.h. One file a width so the extension's builds run in parallel.
#include "pfdense.cuh"

#define DSV41_PFD_K2 16
#define DSV41_PFD_LN false
DSV41_PFD_CFGS(DSV41_PFD_INST)
template __global__ void dsv41_pfd::pfd_dequant_kernel<16, false>(const uint32_t*, long long, long long, half*, int);


DSV41_PFD_WIDTH_STRIPS(16)
