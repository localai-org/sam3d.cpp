package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"image"
	"image/color"
	"image/png"
	"io"
	"math"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sort"
	"testing"
	"time"
)

func syntheticResult() []byte {
	var b bytes.Buffer
	b.WriteString("S3DOUT01")
	binary.Write(&b, binary.LittleEndian, uint32(len(shapes)))
	names := make([]string, 0, len(shapes))
	for name := range shapes {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		shape := shapes[name]
		binary.Write(&b, binary.LittleEndian, uint32(len(name)))
		b.WriteString(name)
		typ := uint32(1)
		if name == "faces" {
			typ = 2
		}
		binary.Write(&b, binary.LittleEndian, typ)
		binary.Write(&b, binary.LittleEndian, uint32(len(shape)))
		n := uint64(1)
		for _, v := range shape {
			n *= v
		}
		binary.Write(&b, binary.LittleEndian, n)
		binary.Write(&b, binary.LittleEndian, shape)
		v := make([]uint32, n)
		if typ == 1 {
			for i := range v {
				v[i] = math.Float32bits(float32(i%17) / 10)
			}
		}
		if name == "joint_transforms" {
			for j := 0; j < 127; j++ {
				for k := 0; k < 8; k++ {
					v[j*8+k] = 0
				}
				v[j*8] = math.Float32bits(float32(j))
				v[j*8+6] = math.Float32bits(1)
				v[j*8+7] = math.Float32bits(1)
			}
		}
		binary.Write(&b, binary.LittleEndian, v)
	}
	return b.Bytes()
}
func TestResultStrict(t *testing.T) {
	v := syntheticResult()
	if _, e := parseResult(bytes.NewReader(v)); e != nil {
		t.Fatal(e)
	}
	for _, bad := range [][]byte{v[:7], v[:len(v)-1], append(append([]byte{}, v...), 0)} {
		if _, e := parseResult(bytes.NewReader(bad)); e == nil {
			t.Fatal("accepted malformed result")
		}
	}
	nan := append([]byte{}, v...)
	n := int(binary.LittleEndian.Uint32(nan[12:16]))
	typ := binary.LittleEndian.Uint32(nan[16+n:])
	rank := int(binary.LittleEndian.Uint32(nan[20+n:]))
	if typ == 1 {
		binary.LittleEndian.PutUint32(nan[32+n+8*rank:], math.Float32bits(float32(math.NaN())))
		if _, e := parseResult(bytes.NewReader(nan)); e == nil {
			t.Fatal("accepted nonfinite result")
		}
	}
}
func TestExports(t *testing.T) {
	b, e := parseResult(bytes.NewReader(syntheticResult()))
	if e != nil {
		t.Fatal(e)
	}
	dir := t.TempDir()
	if e = writeGLB(filepath.Join(dir, "body.glb"), b); e != nil {
		t.Fatal(e)
	}
	v, _ := os.ReadFile(filepath.Join(dir, "body.glb"))
	if binary.LittleEndian.Uint32(v) != 0x46546c67 || int(binary.LittleEndian.Uint32(v[8:])) != len(v) {
		t.Fatal("GLB header")
	}
	n := int(binary.LittleEndian.Uint32(v[12:]))
	var doc map[string]any
	if e = json.Unmarshal(v[20:20+n], &doc); e != nil {
		t.Fatal(e)
	}
	bin := v[28+n:]
	for i, x := range viewerVertices(b.Tensors["vertices"]) {
		if math.Float32frombits(binary.LittleEndian.Uint32(bin[i*4:])) != x {
			t.Fatal("GLB coordinate mismatch")
		}
	}
	if e = writeOBJ(filepath.Join(dir, "body.obj"), b); e != nil {
		t.Fatal(e)
	}
	obj, _ := os.ReadFile(filepath.Join(dir, "body.obj"))
	if bytes.Count(obj, []byte("\nv ")) != 18439 || bytes.Count(obj, []byte("\nf ")) != 36874 {
		t.Fatal("OBJ topology counts")
	}
}
func TestImageContract(t *testing.T) {
	im := image.NewRGBA(image.Rect(0, 0, 8, 8))
	im.SetRGBA(0, 0, color.RGBA{255, 0, 127, 255})
	s := settings{Box: [4]float32{0, 0, 8, 8}, Camera: [4]float32{100, 100, 4, 4}}
	p := packImage(im, s)
	if len(p) != 52+8*8*3 || !bytes.Equal(p[52:55], []byte{255, 0, 127}) {
		t.Fatal("RGB packing")
	}
	if validateSettings(s, 8, 8) != nil {
		t.Fatal("valid settings")
	}
	s.Box[2] = 7
	if validateSettings(s, 8, 8) == nil {
		t.Fatal("tiny crop accepted")
	}
	s.Box[2] = 8
	s.Camera[0] = float32(math.NaN())
	if validateSettings(s, 8, 8) == nil {
		t.Fatal("NaN accepted")
	}
	if _, e := decodeImage([]byte("not an image")); e == nil {
		t.Fatal("bad image accepted")
	}
}
func testApp(t *testing.T) *app {
	t.Helper()
	a, e := loadApp(config{data: t.TempDir(), maxJobs: 20, storage: 2 << 30, timeout: time.Minute})
	if e != nil {
		t.Fatal(e)
	}
	return a
}
func upload(t *testing.T, a *app, origin string, valid bool) *httptest.ResponseRecorder {
	t.Helper()
	var b bytes.Buffer
	w := multipart.NewWriter(&b)
	f, _ := w.CreateFormFile("image", "photo.png")
	if valid {
		png.Encode(f, image.NewRGBA(image.Rect(0, 0, 8, 8)))
	} else {
		f.Write([]byte("bad"))
	}
	w.WriteField("settings", `{"box":[0,0,8,8],"camera":[100,100,4,4]}`)
	w.Close()
	r := httptest.NewRequest("POST", "http://localhost/api/jobs", &b)
	r.Header.Set("Content-Type", w.FormDataContentType())
	if origin != "" {
		r.Header.Set("Origin", origin)
	}
	out := httptest.NewRecorder()
	a.routes().ServeHTTP(out, r)
	return out
}
func TestUploadQueueAndOrigins(t *testing.T) {
	a := testApp(t)
	if got := upload(t, a, "https://evil.invalid", true).Code; got != 403 {
		t.Fatal(got)
	}
	if got := upload(t, a, "", false).Code; got != 400 {
		t.Fatal(got)
	}
	for i := 0; i < 2; i++ {
		if got := upload(t, a, "http://localhost", true).Code; got != 202 {
			t.Fatal(got)
		}
	}
	if got := upload(t, a, "", true).Code; got != 429 {
		t.Fatal(got)
	}
	if len(a.jobs) != 2 {
		t.Fatal("invalid uploads persisted")
	}
}
func objectUpload(t *testing.T, a *app, selected bool) *httptest.ResponseRecorder {
	t.Helper()
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	imagePart, _ := writer.CreateFormFile("image", "scene.png")
	scene := image.NewRGBA(image.Rect(0, 0, 8, 8))
	scene.SetRGBA(3, 4, color.RGBA{R: 12, G: 34, B: 56, A: 255})
	png.Encode(imagePart, scene)
	maskPart, _ := writer.CreateFormFile("mask", "mask.png")
	mask := image.NewRGBA(image.Rect(0, 0, 8, 8))
	if selected {
		for y := 2; y <= 4; y++ {
			for x := 1; x <= 3; x++ {
				mask.SetRGBA(x, y, color.RGBA{R: 255, G: 255, B: 255, A: 255})
			}
		}
	}
	png.Encode(maskPart, mask)
	writer.WriteField("preparation_note", "FFmpeg resized this object photo.")
	writer.Close()
	req := httptest.NewRequest("POST", "http://localhost/api/object-jobs", &body)
	req.Header.Set("Content-Type", writer.FormDataContentType())
	out := httptest.NewRecorder()
	a.routes().ServeHTTP(out, req)
	return out
}
func TestObjectUploadPacksExactMask(t *testing.T) {
	a := testApp(t)
	if got := objectUpload(t, a, false).Code; got != 400 {
		t.Fatalf("empty mask status = %d", got)
	}
	out := objectUpload(t, a, true)
	if out.Code != 202 {
		t.Fatalf("object upload: %d %s", out.Code, out.Body.String())
	}
	var j job
	if err := json.Unmarshal(out.Body.Bytes(), &j); err != nil {
		t.Fatal(err)
	}
	packed, err := os.ReadFile(filepath.Join(a.dir(j.ID), "object.input"))
	if err != nil {
		t.Fatal(err)
	}
	if j.Kind != "object" || !bytes.Equal(packed[:8], []byte("S3DOBJ01")) ||
		binary.LittleEndian.Uint32(packed[8:12]) != 8 || binary.LittleEndian.Uint32(packed[12:16]) != 8 {
		t.Fatal("invalid object job/container metadata")
	}
	if j.Preparation != "FFmpeg resized this object photo." {
		t.Fatal("object photo preparation notice lost")
	}
	inputHash := sha256.Sum256(packed)
	if j.InputSHA != hex.EncodeToString(inputHash[:]) || len(j.SourceSHA) != 64 {
		t.Fatal("invalid object source/input hashes")
	}
	if packed[16+4*(4*8+3)+3] != 255 || !bytes.Equal(packed[16+4*(4*8+3):16+4*(4*8+3)+3], []byte{12, 34, 56}) || packed[16+3] != 0 {
		t.Fatal("object RGB/mask packing mismatch")
	}
}
func TestPersistenceAndCancel(t *testing.T) {
	a := testApp(t)
	r := upload(t, a, "", true)
	var j job
	if e := json.Unmarshal(r.Body.Bytes(), &j); e != nil {
		t.Fatal(e)
	}
	req := httptest.NewRequest("DELETE", "http://localhost/api/jobs/"+j.ID, nil)
	out := httptest.NewRecorder()
	a.routes().ServeHTTP(out, req)
	if out.Code != 200 || a.jobs[j.ID].State != "cancelled" {
		t.Fatal("cancel failed")
	}
	a.jobs[j.ID].State = "running"
	a.save(a.jobs[j.ID])
	restored, e := loadApp(a.cfg)
	if e != nil {
		t.Fatal(e)
	}
	if restored.jobs[j.ID].State != "failed" {
		t.Fatal("interrupted work silently resumed")
	}
	f, e := os.Open(filepath.Join(a.dir(j.ID), "image.input"))
	if e != nil {
		t.Fatal(e)
	}
	defer f.Close()
	p, _ := io.ReadAll(f)
	if len(p) != 244 {
		t.Fatal(len(p))
	}
}
func TestRoutingAndStaticAssets(t *testing.T) {
	a := testApp(t)
	for _, path := range []string{"/", "/app.js", "/frame-pipeline.js", "/frame-encoder-worker.js", "/live-presentation.js", "/vendor/three.core.min.js", "/localai.png", "/skeleton.json"} {
		w := httptest.NewRecorder()
		a.routes().ServeHTTP(w, httptest.NewRequest("GET", path, nil))
		if w.Code != http.StatusOK {
			t.Fatal(path, w.Code)
		}
	}
	for _, path := range []string{"/files/invalid/result.json", "/files/000000000000000000000000/inference.log", "/reference/manifest.json"} {
		w := httptest.NewRecorder()
		a.routes().ServeHTTP(w, httptest.NewRequest("GET", path, nil))
		if w.Code != 404 {
			t.Fatal(path, w.Code)
		}
	}
}
func TestUploadDecoderSerialization(t *testing.T) {
	a := testApp(t)
	a.uploads <- struct{}{}
	if got := upload(t, a, "", true).Code; got != 429 {
		t.Fatal(got)
	}
	<-a.uploads
}

func FuzzResultParser(f *testing.F) {
	f.Add([]byte("S3DOUT01"))
	f.Add(syntheticResult())
	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) > 2<<20 {
			t.Skip()
		}
		parseResult(bytes.NewReader(data))
	})
}
