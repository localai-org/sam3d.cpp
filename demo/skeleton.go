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

const skeletonJoints = 127
const skeletonFrameBytes = 16 + (skeletonJoints*8+3)*4

type skeletonDefinition struct {
	Names   []string `json:"names"`
	Parents []int    `json:"parents"`
}

func skeletonTopology() (skeletonDefinition, error) {
	var d skeletonDefinition
	b, e := assets.ReadFile("web/skeleton.json")
	if e != nil {
		return d, e
	}
	if e = json.Unmarshal(b, &d); e != nil {
		return d, e
	}
	if len(d.Names) != skeletonJoints || len(d.Parents) != skeletonJoints {
		return d, fmt.Errorf("invalid skeleton topology")
	}
	for j, p := range d.Parents {
		if (j == 0 && p != -1) || (j > 0 && (p < 0 || p >= j)) {
			return d, fmt.Errorf("invalid skeleton parent")
		}
	}
	return d, nil
}

// Native MHR global states: centimetres, XYZW, uniform scale. Keep these raw
// states so exports never infer bone orientation from joint positions.
type skeletonFrame struct {
	Time    float64
	Globals []float32
	Camera  []float32
}

func skeletonFromResult(b *bodyResult, stamp float64) (skeletonFrame, error) {
	f := skeletonFrame{stamp, b.Tensors["joint_transforms"], b.Tensors["camera_translation"]}
	return f, validateSkeleton(f)
}
func validateSkeleton(f skeletonFrame) error {
	if !finite64(f.Time) || f.Time < 0 || len(f.Globals) != 127*8 || len(f.Camera) != 3 {
		return fmt.Errorf("skeleton transforms unavailable; reprocess with the current native runner")
	}
	for _, v := range f.Globals {
		if !finite64(float64(v)) {
			return fmt.Errorf("nonfinite skeleton")
		}
	}
	for _, v := range f.Camera {
		if !finite64(float64(v)) || math.Abs(float64(v)) > 100 {
			return fmt.Errorf("invalid skeleton camera")
		}
	}
	for j := 0; j < 127; j++ {
		s := f.Globals[j*8:]
		n := float64(s[3]*s[3] + s[4]*s[4] + s[5]*s[5] + s[6]*s[6])
		if math.Abs(n-1) > 0.01 || s[7] < 1e-5 || s[7] > 1e5 {
			return fmt.Errorf("invalid joint rotation/scale")
		}
		for k := 0; k < 3; k++ {
			if math.Abs(float64(s[k])) > 10000 {
				return fmt.Errorf("extreme joint position")
			}
		}
	}
	return nil
}
func finite64(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }
func encodeSkeleton(f skeletonFrame) ([]byte, error) {
	if e := validateSkeleton(f); e != nil {
		return nil, e
	}
	var b bytes.Buffer
	b.WriteString("S3DSKL01")
	binary.Write(&b, binary.LittleEndian, f.Time)
	binary.Write(&b, binary.LittleEndian, f.Globals)
	binary.Write(&b, binary.LittleEndian, f.Camera)
	return b.Bytes(), nil
}
func readSkeleton(path string) (skeletonFrame, error) {
	var f skeletonFrame
	b, e := os.ReadFile(path)
	if e != nil {
		return f, e
	}
	if len(b) != skeletonFrameBytes || string(b[:8]) != "S3DSKL01" {
		return f, fmt.Errorf("invalid skeleton frame")
	}
	r := bytes.NewReader(b[8:])
	f.Globals = make([]float32, 127*8)
	f.Camera = make([]float32, 3)
	binary.Read(r, binary.LittleEndian, &f.Time)
	binary.Read(r, binary.LittleEndian, f.Globals)
	binary.Read(r, binary.LittleEndian, f.Camera)
	return f, validateSkeleton(f)
}

type quat [4]float64
type vec3 [3]float64

func qmul(a, b quat) quat {
	return quat{a[3]*b[0] + a[0]*b[3] + a[1]*b[2] - a[2]*b[1], a[3]*b[1] - a[0]*b[2] + a[1]*b[3] + a[2]*b[0], a[3]*b[2] + a[0]*b[1] - a[1]*b[0] + a[2]*b[3], a[3]*b[3] - a[0]*b[0] - a[1]*b[1] - a[2]*b[2]}
}
func qinv(q quat) quat { return quat{-q[0], -q[1], -q[2], q[3]} }
func normalized(q quat) quat {
	n := math.Sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3])
	for i := range q {
		q[i] /= n
	}
	return q
}
func rotated(q quat, v vec3) vec3 {
	p := qmul(qmul(q, quat{v[0], v[1], v[2], 0}), qinv(q))
	return vec3{p[0], p[1], p[2]}
}

type jointTRS struct {
	Translation vec3 `json:"translation"`
	Rotation    quat `json:"rotation"`
	Scale       vec3 `json:"scale"`
}

func localSkeleton(f skeletonFrame, d skeletonDefinition) []jointTRS {
	out := make([]jointTRS, 127)
	for j, p := range d.Parents {
		s := f.Globals[j*8:]
		q := normalized(quat{float64(s[3]), float64(s[4]), float64(s[5]), float64(s[6])})
		tr := vec3{float64(s[0]) / 100, float64(s[1]) / 100, float64(s[2]) / 100}
		scale := float64(s[7])
		if p >= 0 {
			a := f.Globals[p*8:]
			pq := normalized(quat{float64(a[3]), float64(a[4]), float64(a[5]), float64(a[6])})
			for k := range tr {
				tr[k] -= float64(a[k]) / 100
			}
			tr = rotated(qinv(pq), tr)
			for k := range tr {
				tr[k] /= float64(a[7])
			}
			q = normalized(qmul(qinv(pq), q))
			scale /= float64(a[7])
		}
		out[j] = jointTRS{tr, q, vec3{scale, scale, scale}}
	}
	return out
}

// A named node hierarchy: no character mesh, skinning weights or rig mapping.
// Animated translations/scales preserve MHR's estimated proportions exactly.
func skeletonGLB(frames []skeletonFrame, movement string) ([]byte, error) {
	if len(frames) == 0 || len(frames) > maxTrackFrames {
		return nil, fmt.Errorf("export requires 1–1800 poses")
	}
	if movement != "in-place" && movement != "camera" {
		return nil, fmt.Errorf("movement must be in-place or camera")
	}
	d, e := skeletonTopology()
	if e != nil {
		return nil, e
	}
	locals := make([][]jointTRS, len(frames))
	times := make([]float32, len(frames))
	roots := make([]vec3, len(frames))
	for i, f := range frames {
		if e = validateSkeleton(f); e != nil {
			return nil, e
		}
		times[i] = float32(f.Time - frames[0].Time)
		if i > 0 && times[i] <= times[i-1] {
			return nil, fmt.Errorf("timestamps must increase at GLB precision")
		}
		locals[i] = localSkeleton(f, d)
		if movement == "camera" {
			roots[i] = vec3{float64(f.Camera[0]), -float64(f.Camera[1]), -float64(f.Camera[2])}
		} else if len(frames) > 1 {
			// Hold pelvis position at the first estimate; retain orientation and pose.
			for k := 0; k < 3; k++ {
				roots[i][k] = float64(frames[0].Globals[8+k]-f.Globals[8+k]) / 100
			}
		}
		if i > 0 {
			for j := 0; j < 127; j++ {
				dot := 0.
				for k := 0; k < 4; k++ {
					dot += locals[i-1][j].Rotation[k] * locals[i][j].Rotation[k]
				}
				if dot < 0 {
					for k := 0; k < 4; k++ {
						locals[i][j].Rotation[k] *= -1
					}
				}
			}
		}
	}
	nodes := make([]map[string]any, 128)
	nodes[0] = map[string]any{"name": "SAM3D", "children": []int{1}, "translation": roots[0]}
	for j := 0; j < 127; j++ {
		tr := locals[0][j]
		nodes[j+1] = map[string]any{"name": d.Names[j], "translation": tr.Translation, "rotation": tr.Rotation, "scale": tr.Scale}
		var children []int
		for child, p := range d.Parents {
			if p == j {
				children = append(children, child+1)
			}
		}
		if len(children) > 0 {
			nodes[j+1]["children"] = children
		}
	}
	doc := map[string]any{"asset": map[string]string{"version": "2.0", "generator": "sam3d.cpp skeleton exporter"}, "scene": 0, "scenes": []any{map[string]any{"nodes": []int{0}}}, "nodes": nodes, "extras": map[string]any{"skeleton": "MHR LOD1", "jointCount": 127, "units": "metres", "up": "Y", "movement": movement, "sourceStartSeconds": frames[0].Time, "scope": "body pose branch; no refined hands; named nodes, no character mesh"}}
	var bin bytes.Buffer
	views := []any{}
	accessors := []any{}
	add := func(values []float32, width int) int {
		offset := bin.Len()
		binary.Write(&bin, binary.LittleEndian, values)
		view := len(views)
		views = append(views, map[string]any{"buffer": 0, "byteOffset": offset, "byteLength": bin.Len() - offset})
		a := map[string]any{"bufferView": view, "componentType": 5126, "count": len(values) / width, "type": map[int]string{1: "SCALAR", 3: "VEC3", 4: "VEC4"}[width]}
		if width == 1 {
			a["min"] = []float32{values[0]}
			a["max"] = []float32{values[len(values)-1]}
		}
		accessors = append(accessors, a)
		return len(accessors) - 1
	}
	if len(frames) > 1 {
		timeIndex := add(times, 1)
		channels := []any{}
		samplers := []any{}
		channel := func(node int, path string, values []float32, width int) {
			output := add(values, width)
			index := len(samplers)
			samplers = append(samplers, map[string]any{"input": timeIndex, "output": output, "interpolation": "LINEAR"})
			channels = append(channels, map[string]any{"sampler": index, "target": map[string]any{"node": node, "path": path}})
		}
		var rv []float32
		for _, v := range roots {
			for _, x := range v {
				rv = append(rv, float32(x))
			}
		}
		channel(0, "translation", rv, 3)
		for j := 0; j < 127; j++ {
			var t, r, s []float32
			for i := range frames {
				v := locals[i][j]
				for _, x := range v.Translation {
					t = append(t, float32(x))
				}
				for _, x := range v.Rotation {
					r = append(r, float32(x))
				}
				for _, x := range v.Scale {
					s = append(s, float32(x))
				}
			}
			channel(j+1, "translation", t, 3)
			channel(j+1, "rotation", r, 4)
			channel(j+1, "scale", s, 3)
		}
		doc["animations"] = []any{map[string]any{"name": "Take", "channels": channels, "samplers": samplers}}
		doc["buffers"] = []any{map[string]any{"byteLength": bin.Len()}}
		doc["bufferViews"] = views
		doc["accessors"] = accessors
	}
	js, e := json.Marshal(doc)
	if e != nil {
		return nil, e
	}
	for len(js)%4 != 0 {
		js = append(js, ' ')
	}
	size := 12 + 8 + len(js)
	if bin.Len() > 0 {
		size += 8 + bin.Len()
	}
	var out bytes.Buffer
	binary.Write(&out, binary.LittleEndian, []uint32{0x46546c67, 2, uint32(size), uint32(len(js)), 0x4e4f534a})
	out.Write(js)
	if bin.Len() > 0 {
		binary.Write(&out, binary.LittleEndian, []uint32{uint32(bin.Len()), 0x004e4942})
		io.Copy(&out, &bin)
	}
	return out.Bytes(), nil
}
