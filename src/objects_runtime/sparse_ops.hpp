// Host-side index tables for the SLat sparse stages (see sparse_ops.cpp).
// The token set is fixed for the whole sampling loop (coords come from the
// SS stage occupancy), so every gather/scatter table is built once.
#pragma once

#include <cstdint>
#include <vector>

namespace sam3d {

struct SlatTables {
    int64_t nf = 0;   // fine tokens (one per active cell at resolution 64)
    int64_t nc = 0;   // coarse tokens (after the 2x mean-pool downsample)

    // fine token -> owning coarse token row (torch scatter_reduce idx; this
    // is also the SparseUpsample gather table: fine_i = coarse[idx[i]])
    std::vector<int32_t> fine_to_coarse;
    // coarse token -> its children rows in the fine table, (8*nc) with the
    // child index fastest (c*8+j); missing children point at the zero
    // sentinel. Coarse grids are partial (a 2x2x2 block need not be full),
    // so this is NOT a reshape of fine_to_coarse.
    std::vector<int32_t> coarse_children;
    std::vector<float> coarse_inv_count;   // (nc) 1/|children| for the mean
    // SubMConv3d neighbor rows, offset-major flattened (n*27 + o) with
    // o = kd*9 + kh*3 + kw matching the (Cout, 3,3,3, Cin) weight flatten;
    // missing neighbors point at the zero sentinel row (index N in a
    // (N+1)-row feature table).
    std::vector<int32_t> conv_fine;    // (27*nf)
    std::vector<int32_t> conv_coarse;  // (27*nc)
    // AbsolutePositionEmbedder output for the coarse grid, (1024, nc) with
    // channel-fastest layout (a plain graph input).
    std::vector<float> ape;

    // coords: (nf, 4) [batch, x, y, z]; builds every table. Returns false on
    // out-of-range coordinates.
    bool build(const int32_t* coords, int64_t n_fine, int channels = 1024);
};

// Swin-window attention tables for the GS decoder (slat_decoder_gs). The
// sparse structure is fixed for the whole decode, so the window grouping is
// built once per shift (blocks alternate shift_window 0 / 4).
struct GsWindowTables {
    int64_t nf = 0;
    // bucket-order gather: out[:, i] = x[:, bkt_idx[i]] reorders tokens so
    // that same-length windows sit contiguous (window-major, token within
    // window in original table order - attention is order invariant)
    std::vector<int32_t> bkt_idx;
    std::vector<int32_t> bwd_idx;  // bucket order -> original order
    struct Bucket {
        int32_t len;      // tokens per window in this bucket
        int32_t off;      // first row in bucket-order token space
        int32_t n_win;    // windows in this bucket (batch dim)
    };
    std::vector<Bucket> buckets;
};

struct GsTables {
    GsWindowTables shifts[2];  // [0]: shift (0,0,0), [1]: shift (4,4,4)
    // AbsolutePositionEmbedder output for the fine coords, (768, nf)
    // token-major rows (channel fastest) - a plain graph input.
    std::vector<float> ape;

    // coords: (nf, 4) [batch, x, y, z]; window_size fixed at 8 (matches
    // slat_decoder_gs.yaml).
    bool build(const int32_t* coords, int64_t n_fine);
};

// One exact SparseSubdivide followed by the SubMConv3d lookup table used by
// the mesh decoder. Child ordering matches torch.nonzero(ones([2,2,2])):
// z changes fastest, then y, then x, for every parent row in input order.
struct MeshSubdivideTables {
    int64_t parent_count = 0;
    int64_t child_count = 0;
    std::vector<int32_t> child_coords;  // [child_count, 4]
    std::vector<int32_t> neighbors;     // [child_count, 27], child-major

    bool build(const int32_t* parent_coords, int64_t count);
};

}  // namespace sam3d
