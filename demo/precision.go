package main

import "strings"

func (c config) precisionName() string {
	if c.precision == "bf16" {
		return "bf16"
	}
	return "f32"
}

func (c config) precisionLabel() string {
	if c.precisionName() == "bf16" {
		return "BF16 encoder · F32 decoder/MHR"
	}
	return "F32"
}

func (c config) precisionArgs(args []string) []string {
	if c.precisionName() == "bf16" {
		return append(args, "--bf16")
	}
	return args
}

// Select one tested arithmetic contract before the child initializes GGML.
// Do not inherit conflicting process-wide experimental precision switches.
// Vulkan BF16 requires the build-copy patched backend; its native capability
// check rejects an incompatible module instead of silently falling back.
func (c config) nativeEnvironment(inherited []string) []string {
	owned := map[string]bool{
		"GGML_VK_F32_NARROW_MATMUL": true, "GGML_VK_F32_NARROW_TRACE": true,
		"GGML_VK_F32_NARROW_TILE":       true,
		"SAM3D_BF16_PACKED_FFN":         true,
		"GGML_VK_FUSE_BF16_SILU_GATE":   true,
		"GGML_VK_FUSE_BF16_AFFINE":      true,
		"GGML_VK_FUSE_BF16_NORM_AFFINE": true,
		"GGML_VK_F32_MATVEC_ROWS":       true, "GGML_VK_F32_MATVEC_TRACE": true,
		"SAM3D_SIMD_SKINNING":      true,
		"SAM3D_SIMD_IMAGE":         true,
		"SAM3D_IMAGE_GATHER":       true,
		"SAM3D_BATCHED_TRANSFERS":  true,
		"GGML_VK_BF16_MATMUL_TILE": true, "GGML_VK_BF16_MATMUL_TRACE": true,
		"GGML_VK_DISABLE_F16": true, "GGML_VK_DISABLE_COOPMAT": true,
		"GGML_VK_DISABLE_COOPMAT2": true, "GGML_VK_FUSE_BF16_ROUND": true,
		"GGML_VK_FUSE_BF16_BINARY":   true,
		"GGML_VK_BF16_BINARY_LINEAR": true,
		"SAM3D_BF16_COOPMAT2":        true, "SAM3D_BF16_FLASH_ATTENTION": true,
		"SAM3D_BF16_PRECISE_PREFIX":   true,
		"SAM3D_BF16_PREFIX_PAD":       true,
		"SAM3D_BF16_PREFIX_MV":        true,
		"SAM3D_BF16_PREFIX_TRANSPOSE": true,
	}
	out := make([]string, 0, len(inherited)+7)
	for _, entry := range inherited {
		key, _, _ := strings.Cut(entry, "=")
		if !owned[key] {
			out = append(out, entry)
		}
	}
	out = append(out, "GGML_VK_DISABLE_F16=1")
	if c.precisionName() == "bf16" && c.backend == "Vulkan" {
		return append(out, "SAM3D_BF16_COOPMAT2=1", "SAM3D_BF16_FLASH_ATTENTION=1",
			"GGML_VK_FUSE_BF16_SILU_GATE=1",
			"GGML_VK_FUSE_BF16_AFFINE=1", "GGML_VK_FUSE_BF16_NORM_AFFINE=1", "SAM3D_IMAGE_GATHER=1",
			"SAM3D_BF16_PRECISE_PREFIX=1", "GGML_VK_FUSE_BF16_ROUND=1", "GGML_VK_FUSE_BF16_BINARY=1",
			"GGML_VK_BF16_BINARY_LINEAR=1", "GGML_VK_BF16_MATMUL_TILE=small", "SAM3D_BATCHED_TRANSFERS=1", "SAM3D_SIMD_SKINNING=1", "GGML_VK_F32_NARROW_MATMUL=1", "GGML_VK_F32_NARROW_TILE=tiny32")
	}
	return append(out, "GGML_VK_DISABLE_COOPMAT=1", "GGML_VK_DISABLE_COOPMAT2=1")
}
