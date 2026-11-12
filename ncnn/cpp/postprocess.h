#pragma once
#include "common.h"
#include <cstring>

constexpr int ANCHORS = 8400, CLASSES = 80, RAW_WIDTH = 84;

// Ultralytics Detect.get_topk_index (groups=1): top 300 anchor maxima,
// then top 300 of the flattened 300x80 scores, retaining repeated anchors.
// No NMS and no confidence threshold here. Match torch 2.11 CPU's choice
// of nth_element+sort versus partial_sort, including its score-only comparator.
// Tied rows are not portable across Torch devices/standard-library versions.
inline std::vector<float> postprocess(const float* raw) {
    std::vector<int> anchors(ANCHORS);
    std::iota(anchors.begin(), anchors.end(), 0);
    std::vector<float> maxima(ANCHORS);
    for (int a = 0; a < ANCHORS; ++a) {
        const float* p = raw + a * RAW_WIDTH;
        for (int c = 0; c < RAW_WIDTH; ++c)
            if (!std::isfinite(p[c])) throw std::runtime_error("non-finite raw output");
        maxima[a] = *std::max_element(p + 4, p + RAW_WIDTH);
    }
    auto anchor_order = [&](int a, int b) { return maxima[a] > maxima[b]; };
    std::nth_element(anchors.begin(), anchors.begin() + 299, anchors.end(), anchor_order);
    std::sort(anchors.begin(), anchors.begin() + 299, anchor_order);
    std::vector<int> indices(300 * CLASSES);
    std::iota(indices.begin(), indices.end(), 0);
    auto score = [&](int i) { return raw[anchors[i / CLASSES] * RAW_WIDTH + 4 + i % CLASSES]; };
    std::partial_sort(indices.begin(), indices.begin() + 300, indices.end(),
        [&](int a, int b) { return score(a) > score(b); });
    std::vector<float> result(OUTPUT_SIZE);
    for (int row = 0; row < 300; ++row) {
        int i = indices[row];
        std::copy_n(raw + anchors[i / CLASSES] * RAW_WIDTH, 4, result.data() + row * 6);
        result[row * 6 + 4] = score(i);
        result[row * 6 + 5] = i % CLASSES;
    }
    return result;
}

inline int check_postprocess(const fs::path& directory) {
    auto raw = read_binary(directory / "bus.raw.bin", ANCHORS * RAW_WIDTH);
    auto expected = read_binary(directory / "bus.postprocess.bin", OUTPUT_SIZE);
    auto actual = postprocess(raw.data());
    bool same = std::memcmp(actual.data(), expected.data(), OUTPUT_SIZE * sizeof(float)) == 0;
    std::cout << (same ? "PASS" : "FAIL") << " postprocess bus bit-match (all 300 rows)\n";
    if (!same) {
        int mismatches = 0;
        for (size_t i = 0; i < OUTPUT_SIZE; ++i) if (actual[i] != expected[i]) {
            if (mismatches++ < 12) std::cout << "difference " << i << " actual=" << actual[i] << " expected=" << expected[i] << '\n';
        }
        std::cout << "different elements=" << mismatches << '\n';
    }
    return same ? 0 : 1;
}
