package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"math"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type glbNode struct {
	Name        string
	Translation [3]float64
	Rotation    [4]float64
	Scale       [3]float64
	Children    []int
}
type glbTestDoc struct {
	Nodes      []glbNode
	Animations []struct {
		Channels []struct {
			Sampler int
			Target  struct {
				Node int
				Path string
			}
		}
		Samplers []struct {
			Input, Output int
			Interpolation string
		}
	}
	Accessors []struct {
		BufferView, Count int
		Type              string
	}
	BufferViews []struct{ ByteOffset, ByteLength int }
}

func readGLBTest(t *testing.T, b []byte) (glbTestDoc, []byte) {
	t.Helper()
	var d glbTestDoc
	if len(b) < 20 || binary.LittleEndian.Uint32(b) != 0x46546c67 || int(binary.LittleEndian.Uint32(b[8:])) != len(b) {
		t.Fatal("bad GLB header")
	}
	n := int(binary.LittleEndian.Uint32(b[12:]))
	if e := json.Unmarshal(b[20:20+n], &d); e != nil {
		t.Fatal(e)
	}
	if len(b) == 20+n {
		return d, nil
	}
	return d, b[28+n:]
}
func accessorTest(d glbTestDoc, b []byte, index int) []float64 {
	v := d.BufferViews[d.Accessors[index].BufferView]
	out := make([]float64, v.ByteLength/4)
	for i := range out {
		out[i] = float64(math.Float32frombits(binary.LittleEndian.Uint32(b[v.ByteOffset+i*4:])))
	}
	return out
}

// Independent matrix FK consumes serialized animation channels and nodes.
type matrixTest [16]float64

func matrixTRS(t [3]float64, q [4]float64, s [3]float64) matrixTest {
	x, y, z, w := q[0], q[1], q[2], q[3]
	return matrixTest{(1 - 2*y*y - 2*z*z) * s[0], (2*x*y - 2*z*w) * s[1], (2*x*z + 2*y*w) * s[2], t[0], (2*x*y + 2*z*w) * s[0], (1 - 2*x*x - 2*z*z) * s[1], (2*y*z - 2*x*w) * s[2], t[1], (2*x*z - 2*y*w) * s[0], (2*y*z + 2*x*w) * s[1], (1 - 2*x*x - 2*y*y) * s[2], t[2], 0, 0, 0, 1}
}
func matrixMul(a, b matrixTest) (c matrixTest) {
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			for k := 0; k < 4; k++ {
				c[i*4+j] += a[i*4+k] * b[k*4+j]
			}
		}
	}
	return
}
func motionTest() []skeletonFrame {
	frames := make([]skeletonFrame, 3)
	for i, stamp := range []float64{.4, .47, .71} {
		f := skeletonFrame{Time: stamp, Globals: make([]float32, 127*8), Camera: []float32{.1, .2, 3}}
		for j := 0; j < 127; j++ {
			s := f.Globals[j*8:]
			s[0] = float32(j*2 + i*15)
			s[1] = float32(j*3 + i*4)
			s[2] = float32(-j + i*2)
			angle := float64(j)*.01 + float64(i)*.3
			s[4] = float32(math.Sin(angle / 2))
			s[6] = float32(math.Cos(angle / 2))
			s[7] = 1 + float32(j)*.001 + float32(i)*.03
			if i == 1 {
				for k := 3; k < 7; k++ {
					s[k] *= -1
				}
			}
		}
		frames[i] = f
	}
	return frames
}
func TestSkeletonGLBRoundTrip(t *testing.T) {
	frames := motionTest()
	topology, _ := skeletonTopology()
	for _, mode := range []string{"in-place", "camera"} {
		b, e := skeletonGLB(frames, mode)
		if e != nil {
			t.Fatal(e)
		}
		d, bin := readGLBTest(t, b)
		if len(d.Nodes) != 128 || len(d.Animations) != 1 || len(d.Animations[0].Channels) != 382 {
			t.Fatal("missing skeleton/channels")
		}
		for frame, f := range frames {
			nodes := append([]glbNode(nil), d.Nodes...)
			nodes[0].Rotation = [4]float64{0, 0, 0, 1}
			nodes[0].Scale = [3]float64{1, 1, 1}
			for _, c := range d.Animations[0].Channels {
				s := d.Animations[0].Samplers[c.Sampler]
				times := accessorTest(d, bin, s.Input)
				if math.Abs(times[frame]-(f.Time-frames[0].Time)) > 1e-7 {
					t.Fatal("timing changed")
				}
				v := accessorTest(d, bin, s.Output)
				node := &nodes[c.Target.Node]
				switch c.Target.Path {
				case "translation":
					copy(node.Translation[:], v[frame*3:])
				case "scale":
					copy(node.Scale[:], v[frame*3:])
				case "rotation":
					copy(node.Rotation[:], v[frame*4:])
					if frame > 0 {
						dot := 0.
						for k := 0; k < 4; k++ {
							dot += v[(frame-1)*4+k] * v[frame*4+k]
						}
						if dot < 0 {
							t.Fatal("quaternion discontinuity")
						}
					}
				}
			}
			matrices := make([]matrixTest, 128)
			matrices[0] = matrixTRS(nodes[0].Translation, nodes[0].Rotation, nodes[0].Scale)
			for j, p := range topology.Parents {
				n := nodes[j+1]
				if n.Name != topology.Names[j] {
					t.Fatal("joint name changed")
				}
				matrices[j+1] = matrixMul(matrices[p+1], matrixTRS(n.Translation, n.Rotation, n.Scale))
				s := f.Globals[j*8:]
				q := [4]float64{float64(s[3]), float64(s[4]), float64(s[5]), float64(s[6])}
				norm := math.Sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3])
				for k := range q {
					q[k] /= norm
				}
				position := [3]float64{float64(s[0]) / 100, float64(s[1]) / 100, float64(s[2]) / 100}
				for k := 0; k < 3; k++ {
					position[k] += nodes[0].Translation[k]
				}
				scale := float64(s[7])
				want := matrixTRS(position, q, [3]float64{scale, scale, scale})
				for k := range want {
					if math.Abs(want[k]-matrices[j+1][k]) > 3e-5 {
						t.Fatalf("%s frame %d joint %d matrix %d: %g != %g", mode, frame, j, k, matrices[j+1][k], want[k])
					}
				}
			}
			if mode == "in-place" {
				for k := 0; k < 3; k++ {
					if math.Abs(matrices[2][k*4+3]-float64(frames[0].Globals[8+k])/100) > 1e-5 {
						t.Fatal("pelvis drift")
					}
				}
			}
		}
	}
	b, e := skeletonGLB(frames[:1], "in-place")
	if e != nil {
		t.Fatal(e)
	}
	d, bin := readGLBTest(t, b)
	if len(d.Animations) != 0 || len(bin) != 0 || len(d.Nodes) != 128 {
		t.Fatal("static pose animated")
	}
	for _, bad := range []func([]skeletonFrame){func(f []skeletonFrame) { f[1].Time = f[0].Time }, func(f []skeletonFrame) { f[0].Globals[7] = 0 }, func(f []skeletonFrame) { f[0].Globals[3] = float32(math.NaN()) }} {
		f := motionTest()
		bad(f)
		if _, e := skeletonGLB(f, "in-place"); e == nil {
			t.Fatal("accepted invalid pose/timeline")
		}
	}
	if _, e := skeletonGLB(frames, "world"); e == nil {
		t.Fatal("accepted unknown movement")
	}
}
func requestTest(a *app, method, path, body string) *httptest.ResponseRecorder {
	w := httptest.NewRecorder()
	a.routes().ServeHTTP(w, httptest.NewRequest(method, path, strings.NewReader(body)))
	return w
}
func TestRecordingExportLifecycle(t *testing.T) {
	a, ctx := trackingWorker(t)
	live := createTestTrack(t, a, "live")
	successfulFrame(t, a, live.ID, "0", ctx)
	w := requestTest(a, "POST", "/api/tracks/"+live.ID+"/record", `{"time":0.15,"name":"My take"}`)
	if w.Code != 201 {
		t.Fatal(w.Code, w.Body)
	}
	var take track
	json.Unmarshal(w.Body.Bytes(), &take)
	if w = requestTest(a, "POST", "/api/tracks/"+take.ID+"/finish", ""); w.Code != 409 {
		t.Fatal("saved take accepted session finish")
	}
	if w = requestTest(a, "POST", "/api/tracks/"+live.ID+"/record", `{"time":0.2}`); w.Code != 409 {
		t.Fatal("double recording accepted")
	}
	successfulFrame(t, a, live.ID, "0.2", ctx)
	successfulFrame(t, a, live.ID, "0.4", ctx)
	w = requestTest(a, "POST", "/api/tracks/"+live.ID+"/record/stop", "")
	if w.Code != 200 {
		t.Fatal(w.Code, w.Body)
	}
	json.Unmarshal(w.Body.Bytes(), &take)
	if take.Count != 2 || take.State != "complete" {
		t.Fatal(take)
	}
	successfulFrame(t, a, live.ID, "0.6", ctx)
	f, e := readSkeleton(filepath.Join(a.trackDir(take.ID), "000000.pose"))
	if e != nil || math.Abs(f.Time-.05) > 1e-10 {
		t.Fatal("capture timestamp lost", e)
	}
	for _, name := range []string{"000002.pose", "000000.bin"} {
		if _, e = os.Stat(filepath.Join(a.trackDir(take.ID), name)); !os.IsNotExist(e) {
			t.Fatal("unexpected recorded file", name)
		}
	}
	w = requestTest(a, "GET", "/api/tracks/"+take.ID+"/skeleton.glb?movement=camera", "")
	if w.Code != 200 || w.Header().Get("Content-Type") != "model/gltf-binary" {
		t.Fatal(w.Code, w.Body)
	}
	d, _ := readGLBTest(t, w.Body.Bytes())
	if len(d.Animations) != 1 {
		t.Fatal("missing animation")
	}
	w = requestTest(a, "GET", "/api/tracks/"+take.ID+"/skeleton.glb?start=0.1&end=0.3", "")
	if w.Code != 200 {
		t.Fatal(w.Code, w.Body)
	}
	d, _ = readGLBTest(t, w.Body.Bytes())
	if len(d.Animations) != 0 {
		t.Fatal("trim did not select single pose")
	}
	for _, query := range []string{"start=NaN", "start=1&end=0", "movement=world", "start=30"} {
		if w = requestTest(a, "GET", "/api/tracks/"+take.ID+"/skeleton.glb?"+query, ""); w.Code == 200 {
			t.Fatal("accepted bad export", query)
		}
	}
	if w = requestTest(a, "PATCH", "/api/tracks/"+take.ID, `{"name":"Renamed"}`); w.Code != 200 {
		t.Fatal(w.Code)
	}
	loaded, e := loadApp(a.cfg)
	if e != nil || loaded.tracks[take.ID].Name != "Renamed" {
		t.Fatal("take not recoverable", e)
	}
	w = requestTest(a, "POST", "/api/tracks/"+live.ID+"/record", `{"time":0.65}`)
	if w.Code != 201 {
		t.Fatal(w.Code, w.Body)
	}
	var second track
	json.Unmarshal(w.Body.Bytes(), &second)
	successfulFrame(t, a, live.ID, "0.8", ctx)
	requestTest(a, "POST", "/api/tracks/"+live.ID+"/finish", "")
	loaded, e = loadApp(a.cfg)
	if e != nil || loaded.tracks[second.ID].State != "complete" || loaded.tracks[second.ID].Count != 1 {
		t.Fatal("session finish lost take", e)
	}
	if w = requestTest(a, "DELETE", "/api/tracks/"+take.ID, ""); w.Code != 200 {
		t.Fatal(w.Code, w.Body)
	}
	if _, e = os.Stat(a.trackDir(take.ID)); !os.IsNotExist(e) {
		t.Fatal("files survived deletion")
	}
}
func TestPhotoAndVideoSkeletonExport(t *testing.T) {
	a, ctx := trackingWorker(t)
	video := createTestTrack(t, a, "offline")
	successfulFrame(t, a, video.ID, "1.5", ctx)
	successfulFrame(t, a, video.ID, "1.7", ctx)
	requestTest(a, "POST", "/api/tracks/"+video.ID+"/finish", "")
	w := requestTest(a, "GET", "/api/tracks/"+video.ID+"/skeleton.glb", "")
	if w.Code != 200 {
		t.Fatal(w.Code, w.Body)
	}
	id := "012345678901234567890123"
	a.mu.Lock()
	a.jobs[id] = &job{ID: id, State: "complete"}
	a.mu.Unlock()
	os.Mkdir(a.dir(id), 0700)
	b, _ := parseResult(bytes.NewReader(syntheticResult()))
	jsonFile(filepath.Join(a.dir(id), "result.json"), b)
	w = requestTest(a, "GET", "/api/jobs/"+id+"/skeleton.glb", "")
	if w.Code != 200 {
		t.Fatal(w.Code, w.Body)
	}
	d, _ := readGLBTest(t, w.Body.Bytes())
	if len(d.Animations) != 0 {
		t.Fatal("photo animated")
	}
	a.mu.Lock()
	a.tracks[video.ID].Skeleton = false
	a.mu.Unlock()
	w = requestTest(a, "GET", "/api/tracks/"+video.ID+"/skeleton.glb", "")
	if w.Code != 409 {
		t.Fatal("legacy sequence silently exported without rotations")
	}
}

// Optional native capture check: also writes a reviewable static GLB beside it.
func TestSkeletonNativeCapture(t *testing.T) {
	path := os.Getenv("SAM3D_TEST_RESULT")
	if path == "" {
		t.Skip("set SAM3D_TEST_RESULT to a current S3DOUT01 native capture")
	}
	file, e := os.Open(path)
	if e != nil {
		t.Fatal(e)
	}
	defer file.Close()
	b, e := parseResult(file)
	if e != nil {
		t.Fatal(e)
	}
	f, e := skeletonFromResult(b, 0)
	if e != nil {
		t.Fatal(e)
	}
	data, e := skeletonGLB([]skeletonFrame{f}, "in-place")
	if e != nil {
		t.Fatal(e)
	}
	d, _ := readGLBTest(t, data)
	topology, _ := skeletonTopology()
	matrices := make([]matrixTest, 128)
	matrices[0] = matrixTRS([3]float64{}, [4]float64{0, 0, 0, 1}, [3]float64{1, 1, 1})
	for j, p := range topology.Parents {
		n := d.Nodes[j+1]
		matrices[j+1] = matrixMul(matrices[p+1], matrixTRS(n.Translation, n.Rotation, n.Scale))
		s := f.Globals[j*8:]
		for k := 0; k < 3; k++ {
			want := float64(b.Tensors["joints"][j*3+k])
			if k > 0 {
				want = -want
			}
			if math.Abs(matrices[j+1][k*4+3]-want) > 1e-5 {
				t.Fatalf("native joint %d axis %d position mismatch", j, k)
			}
			for c := 0; c < 3; c++ {
				want := float64(b.Tensors["joint_rotations"][j*9+k*3+c])
				if math.Abs(matrices[j+1][k*4+c]/float64(s[7])-want) > 1e-5 {
					t.Fatalf("native joint %d rotation mismatch", j)
				}
			}
		}
	}
	if e := os.WriteFile(filepath.Join(filepath.Dir(path), "skeleton.glb"), data, 0600); e != nil {
		t.Fatal(e)
	}
	t.Logf("native 127-joint FK matches public positions/rotations; exported %d bytes", len(data))
}
