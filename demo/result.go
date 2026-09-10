package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
)

// Strictly consume the public C API's complete schema, not a convenient subset.
var shapes = map[string][]uint64{
	"vertices": {1, 18439, 3}, "joints": {1, 127, 3}, "joint_rotations": {1, 127, 3, 3},
	"keypoints": {1, 70, 3}, "keypoints_pixels": {1, 70, 2}, "vertices_pixels": {1, 18439, 2},
	"camera_translation": {1, 3}, "camera_parameters": {1, 3}, "pose_raw": {1, 266},
	"global_rotation": {1, 3}, "body_pose": {1, 133}, "shape": {1, 45}, "scale": {1, 28},
	"hand": {1, 108}, "face": {1, 72}, "mhr_model_parameters": {1, 204},
	"hand_boxes": {1, 2, 4}, "hand_logits": {1, 2, 2}, "faces": {36874, 3},
}

type bodyResult struct {
	Schema      string               `json:"schema"`
	Coordinates string               `json:"coordinates"`
	Tensors     map[string][]float32 `json:"tensors"`
	Faces       []uint32             `json:"faces"`
}

func parseResult(r io.Reader) (*bodyResult, error) {
	var magic [8]byte
	if _, e := io.ReadFull(r, magic[:]); e != nil || string(magic[:]) != "S3DOUT01" {
		return nil, fmt.Errorf("invalid native result magic")
	}
	read := func(v any) error { return binary.Read(r, binary.LittleEndian, v) }
	var count uint32
	if read(&count) != nil || count != uint32(len(shapes)) {
		return nil, fmt.Errorf("invalid tensor count")
	}
	b := &bodyResult{Schema: "sam3d.body.pose_branch.v1", Coordinates: "body metres; x right, y down, z forward; camera translation not applied", Tensors: map[string][]float32{}}
	seen := map[string]bool{}
	for i := uint32(0); i < count; i++ {
		var n, typ, rank uint32
		var elems uint64
		if read(&n) != nil || n < 1 || n > 64 {
			return nil, fmt.Errorf("invalid tensor name length")
		}
		name := make([]byte, n)
		if _, e := io.ReadFull(r, name); e != nil {
			return nil, e
		}
		key := string(name)
		expected, ok := shapes[key]
		if !ok || seen[key] {
			return nil, fmt.Errorf("unexpected or duplicate tensor %q", key)
		}
		seen[key] = true
		if read(&typ) != nil || read(&rank) != nil || read(&elems) != nil || rank != uint32(len(expected)) {
			return nil, fmt.Errorf("invalid descriptor %s", key)
		}
		total := uint64(1)
		for _, want := range expected {
			var dim uint64
			if read(&dim) != nil || dim != want {
				return nil, fmt.Errorf("invalid shape %s", key)
			}
			total *= dim
		}
		if elems != total || typ != map[bool]uint32{false: 1, true: 2}[key == "faces"] {
			return nil, fmt.Errorf("invalid dtype/length %s", key)
		}
		if key == "faces" {
			b.Faces = make([]uint32, total)
			if e := read(&b.Faces); e != nil {
				return nil, e
			}
			for _, v := range b.Faces {
				if v >= 18439 {
					return nil, fmt.Errorf("face index outside mesh")
				}
			}
		} else {
			v := make([]float32, total)
			if e := read(&v); e != nil {
				return nil, e
			}
			for _, x := range v {
				if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
					return nil, fmt.Errorf("nonfinite %s", key)
				}
			}
			b.Tensors[key] = v
		}
	}
	var tail [1]byte
	if n, e := r.Read(tail[:]); n != 0 || e != io.EOF {
		return nil, fmt.Errorf("trailing native data")
	}
	return b, nil
}

// A single rigid coordinate conversion, never pose-dependent alignment.
func viewerVertices(v []float32) []float32 {
	out := append([]float32(nil), v...)
	for i := 0; i < len(out); i += 3 {
		out[i+1] = -out[i+1]
		out[i+2] = -out[i+2]
	}
	return out
}
func writeOBJ(path string, b *bodyResult) error {
	f, e := os.Create(path)
	if e != nil {
		return e
	}
	defer f.Close()
	var out bytes.Buffer
	out.WriteString("# SAM 3D Body pose branch; metres, Y up; static posed mesh, not an animation rig\n")
	v := viewerVertices(b.Tensors["vertices"])
	for i := 0; i < len(v); i += 3 {
		fmt.Fprintf(&out, "v %.9g %.9g %.9g\n", v[i], v[i+1], v[i+2])
	}
	for i := 0; i < len(b.Faces); i += 3 {
		fmt.Fprintf(&out, "f %d %d %d\n", b.Faces[i]+1, b.Faces[i+1]+1, b.Faces[i+2]+1)
	}
	_, e = f.Write(out.Bytes())
	return e
}

// Static glTF 2.0 mesh, not a guessed skin/animation. Preserve metric scale and origin.
func writeGLB(path string, b *bodyResult) error {
	v := viewerVertices(b.Tensors["vertices"])
	var bin bytes.Buffer
	binary.Write(&bin, binary.LittleEndian, v)
	offset := bin.Len()
	binary.Write(&bin, binary.LittleEndian, b.Faces)
	lo, hi := []float32{v[0], v[1], v[2]}, []float32{v[0], v[1], v[2]}
	for i, x := range v {
		j := i % 3
		if x < lo[j] {
			lo[j] = x
		}
		if x > hi[j] {
			hi[j] = x
		}
	}
	doc := map[string]any{"asset": map[string]any{"version": "2.0", "generator": "sam3d.cpp body demo"}, "scene": 0,
		"scenes": []any{map[string]any{"nodes": []int{0}}}, "nodes": []any{map[string]any{"mesh": 0, "name": "Estimated body (static, metres)"}},
		"meshes":      []any{map[string]any{"primitives": []any{map[string]any{"attributes": map[string]int{"POSITION": 0}, "indices": 1, "material": 0}}}},
		"materials":   []any{map[string]any{"doubleSided": true, "pbrMetallicRoughness": map[string]any{"baseColorFactor": []float32{.3, .7, .85, 1}, "metallicFactor": 0, "roughnessFactor": .75}}},
		"buffers":     []any{map[string]any{"byteLength": bin.Len()}},
		"bufferViews": []any{map[string]any{"buffer": 0, "byteOffset": 0, "byteLength": offset, "target": 34962}, map[string]any{"buffer": 0, "byteOffset": offset, "byteLength": len(b.Faces) * 4, "target": 34963}},
		"accessors":   []any{map[string]any{"bufferView": 0, "componentType": 5126, "count": len(v) / 3, "type": "VEC3", "min": lo, "max": hi}, map[string]any{"bufferView": 1, "componentType": 5125, "count": len(b.Faces), "type": "SCALAR"}},
		"extras":      map[string]any{"scope": "Body pose branch only; no refined hands", "coordinates": "metres; Y up; original body origin; diag(1,-1,-1) applied"},
	}
	js, e := json.Marshal(doc)
	if e != nil {
		return e
	}
	for len(js)%4 != 0 {
		js = append(js, ' ')
	}
	var out bytes.Buffer
	binary.Write(&out, binary.LittleEndian, []uint32{0x46546c67, 2, uint32(12 + 8 + len(js) + 8 + bin.Len()), uint32(len(js)), 0x4e4f534a})
	out.Write(js)
	binary.Write(&out, binary.LittleEndian, []uint32{uint32(bin.Len()), 0x004e4942})
	out.Write(bin.Bytes())
	return os.WriteFile(path, out.Bytes(), 0600)
}
