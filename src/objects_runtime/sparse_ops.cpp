// Host-side tables for the SLat sparse stages - see sparse_ops.hpp.
// These mirror torch's SparseDownsample (//2 coords, unique-sorted codes,
// scatter mean), SparseUpsample (gather by the downsample inverse index) and
// spconv's SubMConv3d (submanifold: output set == input set, missing
// neighbours contribute zero).
#include "sparse_ops.hpp"

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <map>
#include <cstdio>
#include <tuple>
#include <unordered_map>

namespace sam3d {

namespace {

using Key = std::tuple<int32_t, int32_t, int32_t>;

struct MeshParentKey {
    int32_t batch;
    int32_t x;
    int32_t y;
    int32_t z;

    bool operator==(const MeshParentKey&) const = default;
};

struct MeshParentKeyHash {
    size_t operator()(const MeshParentKey& key) const noexcept {
        size_t hash = 0xcbf29ce484222325ULL;
        const auto mix = [&](int32_t value) {
            hash ^= static_cast<uint32_t>(value);
            hash *= 0x100000001b3ULL;
        };
        mix(key.batch);
        mix(key.x);
        mix(key.y);
        mix(key.z);
        return hash;
    }
};

}  // namespace

bool SlatTables::build(const int32_t* coords, int64_t n_fine, int channels) {
    nf = n_fine;
    if (nf <= 0) return false;

    // ---- downsample: coarse key = (b, x/2, y/2, z/2) --------------------
    // torch orders coarse rows by code.unique() with a mixed-radix code over
    // the coarse grid maxima -> lexicographic (b, x, y, z) order. Row ids are
    // the SORTED position (torch's return_inverse), not insertion order.
    std::map<std::tuple<int32_t, Key>, int32_t> coarse_index;  // sorted = code order
    fine_to_coarse.assign((size_t)nf, 0);
    {
        // pass 1: collect the unique coarse keys (any temp id)
        std::map<std::tuple<int32_t, Key>, int32_t> tmp;
        for (int64_t i = 0; i < nf; i++) {
            Key k{coords[i * 4 + 1] / 2, coords[i * 4 + 2] / 2,
                  coords[i * 4 + 3] / 2};
            tmp.emplace(std::make_tuple(coords[i * 4 + 0], k), 0);
        }
        // pass 2: assign sorted row ids
        int32_t id = 0;
        for (auto& [k, v] : tmp) v = id++;
        coarse_index = std::move(tmp);
        nc = (int64_t)id;
        std::vector<int32_t> counts((size_t)nc, 0);
        for (int64_t i = 0; i < nf; i++) {
            Key k{coords[i * 4 + 1] / 2, coords[i * 4 + 2] / 2,
                  coords[i * 4 + 3] / 2};
            const int32_t c = coarse_index.at(std::make_tuple(coords[i * 4 + 0], k));
            fine_to_coarse[(size_t)i] = c;
            counts[(size_t)c]++;
        }

    coarse_inv_count.resize((size_t)nc);
    // children table (c*8+j): gather rows for the mean-pool; a coarse cell's
    // 2x2x2 block is often partial, missing slots -> sentinel (nf = zero row)
    std::vector<int32_t> cnts((size_t)nc, 0);
    coarse_children.assign((size_t)(8 * nc), (int32_t)nf);
    for (int64_t i = 0; i < nf; i++) {
        const int32_t c = fine_to_coarse[(size_t)i];
        const int slot = cnts[(size_t)c]++;
        if (slot < 8) coarse_children[(size_t)(c * 8 + slot)] = (int32_t)i;
    }
    for (int64_t c = 0; c < nc; c++) {
        const int32_t k = cnts[(size_t)c];
        // NOTE: torch scatter_reduce(reduce="mean") averages over the written
        // values PLUS the initial zero row, i.e. sum/(count+1) - this quirk
        // is part of the reference behaviour and must be reproduced.
        coarse_inv_count[(size_t)c] = 1.0f / (float)(k + 1);
        // pack children compactly (keep the used slots first)
        int w = 0;
        for (int j = 0; j < 8; j++) {
            const int32_t v = coarse_children[(size_t)(c * 8 + j)];
            if (v != (int32_t)nf) coarse_children[(size_t)(c * 8 + w++)] = v;
        }
        for (int j = w; j < 8; j++) coarse_children[(size_t)(c * 8 + j)] = (int32_t)nf;
    }
    }

    if (getenv("SAM3D_DBG_TABLE")) {
        fprintf(stderr, "[tbl] first 12 coarse keys:\n");
        int printed = 0;
        for (const auto& [bk, row] : coarse_index) {
            if (printed++ >= 40) break;
            fprintf(stderr, "[tbl] row %d: b=%d xyz=(%d,%d,%d)\n", row,
                    std::get<0>(bk), std::get<0>(std::get<1>(bk)),
                    std::get<1>(std::get<1>(bk)), std::get<2>(std::get<1>(bk)));
        }
    }
    // ---- submanifold conv neighbour tables ------------------------------
    // hash the point sets for O(1) membership
    std::map<std::tuple<int32_t, Key>, int32_t> fine_index;
    for (int64_t i = 0; i < nf; i++)
        fine_index.emplace(std::make_tuple(
            coords[i * 4 + 0],
            Key{coords[i * 4 + 1], coords[i * 4 + 2], coords[i * 4 + 3]}),
            (int32_t)i);

    auto build_conv = [&](const std::map<std::tuple<int32_t, Key>, int32_t>& idx,
                          int64_t n, std::vector<int32_t>& table) {
        table.assign((size_t)(27 * n), (int32_t)n);  // sentinel = n (zero row)
        for (const auto& [bk, row] : idx) {
            const auto& [b, k] = bk;
            for (int o = 0; o < 27; o++) {
                const int32_t dx = o / 9 - 1, dy = (o / 3) % 3 - 1, dz = o % 3 - 1;
                auto it = idx.find({b, Key{std::get<0>(k) + dx,
                                           std::get<1>(k) + dy,
                                           std::get<2>(k) + dz}});
                if (it != idx.end())
                    table[(size_t)(row * 27 + o)] = it->second;
            }
        }
    };
    build_conv(fine_index, nf, conv_fine);
    build_conv(coarse_index, nc, conv_coarse);

    // ---- AbsolutePositionEmbedder for the coarse grid -------------------
    // per-axis: outer(coord, freqs) -> [sin | cos] (2*freq_dim per axis),
    // axes concatenated, zero-padded to `channels`.
    const int freq_dim = channels / 3 / 2;
    ape.assign((size_t)channels * nc, 0.0f);
    // coarse coords in row order
    std::vector<int32_t> cc((size_t)nc * 3);
    for (const auto& [bk, row] : coarse_index) {
        const Key& k = std::get<1>(bk);
        cc[(size_t)row * 3 + 0] = std::get<0>(k);
        cc[(size_t)row * 3 + 1] = std::get<1>(k);
        cc[(size_t)row * 3 + 2] = std::get<2>(k);
    }
    std::vector<float> freqs((size_t)freq_dim);
    for (int i = 0; i < freq_dim; i++)
        freqs[(size_t)i] = 1.0f / powf(10000.0f, (float)i / (float)freq_dim);
    for (int64_t n = 0; n < nc; n++) {
        for (int ax = 0; ax < 3; ax++) {
            const float v = (float)cc[(size_t)n * 3 + ax];
            float* sin_seg = &ape[(size_t)n * channels + ax * 2 * freq_dim];
            float* cos_seg = sin_seg + freq_dim;
            for (int i = 0; i < freq_dim; i++) {
                const float ph = v * freqs[(size_t)i];
                sin_seg[i] = sinf(ph);
                cos_seg[i] = cosf(ph);
            }
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// GS decoder swin-window tables (see sparse_ops.hpp). Mirrors torch
// calc_window_partition: window id over shifted coords with mixed-radix
// OFFSET, tokens grouped by window id (order inside a window is irrelevant -
// attention is permutation invariant), then compacted into same-length
// buckets for batched attention.
// ---------------------------------------------------------------------------
bool GsTables::build(const int32_t* coords, int64_t n_fine) {
    if (n_fine <= 0) return false;
    const int W = 8;  // window_size from slat_decoder_gs.yaml

    for (int si = 0; si < 2; si++) {
        const int32_t sh = 4 * si;  // swin shift: (4,4,4) for odd blocks
        GsWindowTables& t = shifts[si];
        t.nf = n_fine;

        // window key per token: (b, x/8, y/8, z/8) of the shifted coords.
        // torch computes the NUM_WINDOWS grid from the shifted maxima; the
        // mixed-radix id ordering equals lexicographic (b, wx, wy, wz).
        std::map<std::tuple<int32_t, int32_t, int32_t, int32_t>, int32_t> win_id;
        const int64_t nf = n_fine;
        std::vector<int32_t> wid((size_t)nf);
        for (int64_t i = 0; i < nf; i++) {
            std::tuple<int32_t, int32_t, int32_t, int32_t> k{
                coords[i * 4 + 0], (coords[i * 4 + 1] + sh) / W,
                (coords[i * 4 + 2] + sh) / W, (coords[i * 4 + 3] + sh) / W};
            auto it = win_id.find(k);
            if (it == win_id.end()) it = win_id.emplace(k, (int32_t)win_id.size()).first;
            wid[(size_t)i] = it->second;
        }

        // rows sorted by window id (stable: keep table order inside a window)
        std::vector<int32_t> order((size_t)nf);
        for (int64_t i = 0; i < nf; i++) order[(size_t)i] = (int32_t)i;
        std::stable_sort(order.begin(), order.end(),
                         [&](int32_t a, int32_t b) { return wid[(size_t)a] < wid[(size_t)b]; });

        // window segments in sorted order
        std::vector<std::pair<int32_t, int32_t>> segs;  // (row_start, len)
        for (int64_t i = 0; i < nf;) {
            int64_t j = i;
            while (j < nf && wid[(size_t)order[(size_t)j]] == wid[(size_t)order[(size_t)i]]) j++;
            segs.emplace_back((int32_t)i, (int32_t)(j - i));
            i = j;
        }

        // bucket by segment length (map iterates len ascending; torch makes
        // no layout promise across windows, only the partition matters)
        std::map<int32_t, std::vector<int32_t>> by_len;  // len -> seg indices
        for (int32_t s = 0; s < (int32_t)segs.size(); s++)
            by_len[segs[(size_t)s].second].push_back(s);

        t.bkt_idx.assign((size_t)nf, 0);
        t.bwd_idx.assign((size_t)nf, 0);
        t.buckets.clear();
        int32_t row = 0;
        for (auto& [len, seg_list] : by_len) {
            const int32_t off = row;
            for (int32_t s : seg_list) {
                const auto& [start, slen] = segs[(size_t)s];
                for (int32_t j = 0; j < slen; j++) {
                    const int32_t orig = order[(size_t)(start + j)];
                    t.bkt_idx[(size_t)(row)] = orig;        // orig -> bucket pos
                    t.bwd_idx[(size_t)orig] = (int32_t)row; // bucket pos -> orig
                    row++;
                }
            }
            t.buckets.push_back({len, off, (int32_t)seg_list.size()});
        }
        if (row != (int32_t)nf) return false;
    }

    // ---- AbsolutePositionEmbedder for the fine grid (768 ch) ------------
    {
        const int channels = 768;
        const int freq_dim = channels / 3 / 2;
        const int64_t nf = n_fine;
        ape.assign((size_t)channels * nf, 0.0f);
        std::vector<float> freqs((size_t)freq_dim);
        for (int i = 0; i < freq_dim; i++)
            freqs[(size_t)i] = 1.0f / powf(10000.0f, (float)i / (float)freq_dim);
        for (int64_t n = 0; n < nf; n++) {
            for (int ax = 0; ax < 3; ax++) {
                const float v = (float)coords[n * 4 + 1 + ax];
                float* sin_seg = &ape[(size_t)n * channels + ax * 2 * freq_dim];
                float* cos_seg = sin_seg + freq_dim;
                for (int i = 0; i < freq_dim; i++) {
                    const float ph = v * freqs[(size_t)i];
                    sin_seg[i] = sinf(ph);
                    cos_seg[i] = cosf(ph);
                }
            }
        }
    }
    return true;
}

bool MeshSubdivideTables::build(const int32_t* parent_coords, int64_t count) {
    if (!parent_coords || count <= 0 || count > INT32_MAX / 8) return false;
    parent_count = count;
    child_count = count * 8;
    child_coords.resize((size_t)child_count * 4);

    // Every parent contributes all eight children. Build an index over the
    // much smaller parent set, then derive child neighbours from the parent
    // row and the three low coordinate bits. This needs 27 lookups per parent
    // instead of 27 lookups for every child.
    const int32_t first_batch = parent_coords[0];
    int32_t minimum[3] = {parent_coords[1], parent_coords[2], parent_coords[3]};
    int32_t maximum[3] = {minimum[0], minimum[1], minimum[2]};
    bool single_batch = true;
    for (int64_t parent = 1; parent < count; ++parent) {
        single_batch &= parent_coords[parent * 4] == first_batch;
        for (int axis = 0; axis < 3; ++axis) {
            minimum[axis] = std::min(minimum[axis], parent_coords[parent * 4 + axis + 1]);
            maximum[axis] = std::max(maximum[axis], parent_coords[parent * 4 + axis + 1]);
        }
    }
    const uint64_t extent_x = static_cast<uint64_t>(
        static_cast<int64_t>(maximum[0]) - minimum[0] + 1);
    const uint64_t extent_y = static_cast<uint64_t>(
        static_cast<int64_t>(maximum[1]) - minimum[1] + 1);
    const uint64_t extent_z = static_cast<uint64_t>(
        static_cast<int64_t>(maximum[2]) - minimum[2] + 1);
    constexpr uint64_t kMaximumDenseCells = 64ULL * 1024 * 1024;
    uint64_t dense_cells = kMaximumDenseCells + 1;
    if (single_batch && extent_x <= kMaximumDenseCells &&
        extent_y <= kMaximumDenseCells / extent_x) {
        const uint64_t plane = extent_x * extent_y;
        if (extent_z <= kMaximumDenseCells / plane) dense_cells = plane * extent_z;
    }
    const bool use_dense = single_batch && dense_cells <= kMaximumDenseCells &&
                           dense_cells <= static_cast<uint64_t>(count) * 32;
    std::vector<int32_t> dense_index;
    if (use_dense) dense_index.assign(static_cast<size_t>(dense_cells), -1);
    std::unordered_map<MeshParentKey, int32_t, MeshParentKeyHash> parent_index;
    if (!use_dense) {
        parent_index.max_load_factor(0.7f);
        parent_index.reserve(static_cast<size_t>(count));
    }
    const auto dense_offset = [&](int32_t x, int32_t y, int32_t z) {
        return (static_cast<size_t>(static_cast<int64_t>(x) - minimum[0]) * extent_y +
                static_cast<size_t>(static_cast<int64_t>(y) - minimum[1])) * extent_z +
               static_cast<size_t>(static_cast<int64_t>(z) - minimum[2]);
    };
    for (int64_t parent = 0; parent < count; ++parent) {
        const int32_t batch = parent_coords[parent * 4];
        const int32_t parent_x = parent_coords[parent * 4 + 1];
        const int32_t parent_y = parent_coords[parent * 4 + 2];
        const int32_t parent_z = parent_coords[parent * 4 + 3];
        if (use_dense) {
            int32_t& slot = dense_index[dense_offset(parent_x, parent_y, parent_z)];
            if (slot >= 0) return false;
            slot = static_cast<int32_t>(parent);
        } else if (!parent_index.emplace(
                       MeshParentKey{batch, parent_x, parent_y, parent_z},
                       static_cast<int32_t>(parent)).second) {
            return false;
        }
        for (int child = 0; child < 8; ++child) {
            const int64_t row = parent * 8 + child;
            const int32_t x = parent_x * 2 + child / 4;
            const int32_t y = parent_y * 2 + (child / 2) % 2;
            const int32_t z = parent_z * 2 + child % 2;
            child_coords[(size_t)row * 4] = batch;
            child_coords[(size_t)row * 4 + 1] = x;
            child_coords[(size_t)row * 4 + 2] = y;
            child_coords[(size_t)row * 4 + 3] = z;
        }
    }

    std::array<std::array<uint8_t, 27>, 8> adjacent_offset{};
    std::array<std::array<uint8_t, 27>, 8> adjacent_child{};
    for (int child = 0; child < 8; ++child) {
        const int bits[3] = {child / 4, (child / 2) % 2, child % 2};
        for (int offset = 0; offset < 27; ++offset) {
            const int delta[3] = {
                offset / 9 - 1, (offset / 3) % 3 - 1, offset % 3 - 1};
            int parent_delta[3];
            int target_bits[3];
            for (int axis = 0; axis < 3; ++axis) {
                const int coordinate = bits[axis] + delta[axis];
                parent_delta[axis] = coordinate < 0 ? -1 : coordinate > 1 ? 1 : 0;
                target_bits[axis] = coordinate - 2 * parent_delta[axis];
            }
            adjacent_offset[child][offset] = static_cast<uint8_t>(
                (parent_delta[0] + 1) * 9 + (parent_delta[1] + 1) * 3 + parent_delta[2] + 1);
            adjacent_child[child][offset] = static_cast<uint8_t>(
                target_bits[0] * 4 + target_bits[1] * 2 + target_bits[2]);
        }
    }

    neighbors.assign((size_t)child_count * 27, -1);
    for (int64_t parent = 0; parent < count; ++parent) {
        const int32_t batch = parent_coords[parent * 4];
        const int32_t parent_x = parent_coords[parent * 4 + 1];
        const int32_t parent_y = parent_coords[parent * 4 + 2];
        const int32_t parent_z = parent_coords[parent * 4 + 3];
        std::array<int32_t, 27> adjacent_parents;
        for (int offset = 0; offset < 27; ++offset) {
            const int32_t dx = offset / 9 - 1;
            const int32_t dy = (offset / 3) % 3 - 1;
            const int32_t dz = offset % 3 - 1;
            const int32_t x = parent_x + dx;
            const int32_t y = parent_y + dy;
            const int32_t z = parent_z + dz;
            if (use_dense) {
                adjacent_parents[static_cast<size_t>(offset)] =
                    x < minimum[0] || x > maximum[0] || y < minimum[1] || y > maximum[1] ||
                    z < minimum[2] || z > maximum[2] ? -1 : dense_index[dense_offset(x, y, z)];
            } else {
                const auto found = parent_index.find(MeshParentKey{batch, x, y, z});
                adjacent_parents[static_cast<size_t>(offset)] =
                    found == parent_index.end() ? -1 : found->second;
            }
        }
        for (int child = 0; child < 8; ++child) {
            const int64_t row = parent * 8 + child;
            for (int offset = 0; offset < 27; ++offset) {
                const int32_t target_parent =
                    adjacent_parents[adjacent_offset[child][offset]];
                if (target_parent >= 0) {
                    neighbors[static_cast<size_t>(row) * 27 + offset] =
                        target_parent * 8 + adjacent_child[child][offset];
                }
            }
        }
    }
    return true;
}

}  // namespace sam3d
