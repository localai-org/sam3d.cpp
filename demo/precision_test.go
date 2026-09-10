package main

import (
	"encoding/json"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
)

func TestNativePrecisionContracts(t *testing.T) {
	for _, backend := range []string{"CPU", "Vulkan"} {
		for _, precision := range []string{"", "f32", "bf16"} {
			c := config{backend: backend, precision: precision}
			t.Run(backend+"/"+precision, func(t *testing.T) {
				// Conflicting inherited values and duplicates must not leak in.
				input := []string{"PATH=/bin", "VK_DRIVER_FILES=test-icd.json",
					"GGML_VK_F32_NARROW_MATMUL=0", "GGML_VK_F32_NARROW_MATMUL=1", "GGML_VK_F32_NARROW_TRACE=1",
					"GGML_VK_F32_MATVEC_ROWS=8", "GGML_VK_F32_MATVEC_TRACE=1",
					"GGML_VK_F32_NARROW_TILE=tiny64",
					"SAM3D_BF16_PACKED_FFN=1",
					"GGML_VK_FUSE_BF16_SILU_GATE=1",
					"GGML_VK_FUSE_BF16_AFFINE=1",
					"GGML_VK_FUSE_BF16_NORM_AFFINE=1",
					"SAM3D_SIMD_SKINNING=1",
					"SAM3D_SIMD_IMAGE=1",
					"SAM3D_IMAGE_GATHER=1",
					"SAM3D_BATCHED_TRANSFERS=0", "SAM3D_BATCHED_TRANSFERS=1",
					"GGML_VK_BF16_MATMUL_TILE=large", "GGML_VK_BF16_MATMUL_TRACE=1",
					"GGML_VK_DISABLE_F16=0", "GGML_VK_DISABLE_F16=0", "GGML_VK_DISABLE_COOPMAT=1",
					"GGML_VK_DISABLE_COOPMAT2=1", "GGML_VK_FUSE_BF16_ROUND=0",
					"GGML_VK_FUSE_BF16_BINARY=1",
					"GGML_VK_BF16_BINARY_LINEAR=1",
					"SAM3D_BF16_PREFIX_PAD=1",
					"SAM3D_BF16_PREFIX_MV=1",
					"SAM3D_BF16_PREFIX_TRANSPOSE=1",
					"SAM3D_BF16_COOPMAT2=0", "SAM3D_BF16_FLASH_ATTENTION=0", "SAM3D_BF16_PRECISE_PREFIX=0"}
				before := append([]string(nil), input...)
				env := map[string]string{}
				for _, v := range c.nativeEnvironment(input) {
					k, value, _ := strings.Cut(v, "=")
					if _, ok := env[k]; ok {
						t.Fatal("duplicate", k)
					}
					env[k] = value
				}
				if !reflect.DeepEqual(input, before) {
					t.Fatal("mutated caller environment")
				}
				if env["PATH"] != "/bin" || env["VK_DRIVER_FILES"] != "test-icd.json" || env["GGML_VK_DISABLE_F16"] != "1" {
					t.Fatal(env)
				}
				cm2 := backend == "Vulkan" && precision == "bf16"
				for _, key := range []string{"SAM3D_SIMD_IMAGE", "GGML_VK_F32_MATVEC_ROWS", "GGML_VK_F32_MATVEC_TRACE", "GGML_VK_F32_NARROW_TRACE", "SAM3D_BF16_PACKED_FFN"} {
					if _, ok := env[key]; ok {
						t.Fatal("unaccepted matvec experiment leaked into demo", key)
					}
				}
				if _, ok := env["GGML_VK_BF16_MATMUL_TRACE"]; ok {
					t.Fatal("matrix tile tracing leaked into demo")
				}
				wantTile := ""
				wantNarrowTile := ""
				if cm2 {
					wantTile = "small"
					wantNarrowTile = "tiny32"
				}
				if env["GGML_VK_F32_NARROW_TILE"] != wantNarrowTile {
					t.Fatal("unexpected narrow tile contract", env)
				}
				if env["GGML_VK_BF16_MATMUL_TILE"] != wantTile {
					t.Fatal("unexpected matrix tile contract", env)
				}
				if _, ok := env["SAM3D_BF16_PREFIX_PAD"]; ok {
					t.Fatal("unaccepted prefix padding leaked into demo")
				}
				if _, ok := env["SAM3D_BF16_PREFIX_MV"]; ok {
					t.Fatal("unaccepted prefix vector batching leaked into demo")
				}
				if _, ok := env["SAM3D_BF16_PREFIX_TRANSPOSE"]; ok {
					t.Fatal("unaccepted prefix transpose leaked into demo")
				}
				for _, key := range []string{"SAM3D_IMAGE_GATHER", "GGML_VK_FUSE_BF16_NORM_AFFINE", "GGML_VK_FUSE_BF16_AFFINE", "GGML_VK_FUSE_BF16_SILU_GATE", "GGML_VK_F32_NARROW_MATMUL", "SAM3D_SIMD_SKINNING", "SAM3D_BATCHED_TRANSFERS", "SAM3D_BF16_COOPMAT2", "SAM3D_BF16_FLASH_ATTENTION", "SAM3D_BF16_PRECISE_PREFIX", "GGML_VK_FUSE_BF16_ROUND", "GGML_VK_FUSE_BF16_BINARY", "GGML_VK_BF16_BINARY_LINEAR"} {
					want := ""
					if cm2 {
						want = "1"
					}
					if env[key] != want {
						t.Fatalf("%s: %q != %q", key, env[key], want)
					}
				}
				for _, key := range []string{"GGML_VK_DISABLE_COOPMAT", "GGML_VK_DISABLE_COOPMAT2"} {
					want := "1"
					if cm2 {
						want = ""
					}
					if env[key] != want {
						t.Fatalf("%s: %q != %q", key, env[key], want)
					}
				}
				for _, base := range [][]string{{"--worker", "module"}, {"module", "CPU"}} {
					want := append([]string(nil), base...)
					if precision == "bf16" {
						want = append(want, "--bf16")
					}
					if got := c.precisionArgs(base); !reflect.DeepEqual(got, want) {
						t.Fatal(got, want)
					}
				}
				a := &app{cfg: c}
				r := httptest.NewRecorder()
				a.routes().ServeHTTP(r, httptest.NewRequest("GET", "/api/config", nil))
				var data map[string]any
				if err := json.Unmarshal(r.Body.Bytes(), &data); err != nil {
					t.Fatal(err)
				}
				if data["precision"] != c.precisionName() || !strings.Contains(data["model"].(string), c.precisionLabel()) {
					t.Fatal(data)
				}
			})
		}
	}
}

func TestRejectInvalidPrecision(t *testing.T) {
	for _, value := range []string{"fp16", "BF16", "auto", "bf16 --other"} {
		if _, err := loadApp(config{precision: value}); err == nil {
			t.Fatal("accepted", value)
		}
	}
}
