package main

import (
	"bytes"
	"context"
	"encoding/json"
	"image"
	"image/color"
	"image/gif"
	"image/png"
	"mime/multipart"
	"net/http/httptest"
	"os"
	"os/exec"
	"strings"
	"testing"
)

func prepareRequest(t *testing.T, a *app, data []byte, force bool) *httptest.ResponseRecorder {
	t.Helper()
	var body bytes.Buffer
	m := multipart.NewWriter(&body)
	p, _ := m.CreateFormFile("image", "untrusted;name.gif")
	p.Write(data)
	m.Close()
	path := "/api/prepare"
	if force {
		path += "?force=1"
	}
	r := httptest.NewRequest("POST", path, &body)
	r.Header.Set("Content-Type", m.FormDataContentType())
	w := httptest.NewRecorder()
	a.routes().ServeHTTP(w, r)
	return w
}
func TestPreparePNGUnchanged(t *testing.T) {
	a := testApp(t)
	a.cfg.ffmpeg = "/missing/ffmpeg"
	var b bytes.Buffer
	png.Encode(&b, image.NewRGBA(image.Rect(0, 0, 16, 16)))
	r := prepareRequest(t, a, b.Bytes(), false)
	if r.Code != 200 || !bytes.Equal(r.Body.Bytes(), b.Bytes()) || r.Header().Get("X-Sam3d-Converted") != "false" {
		t.Fatalf("%d %s", r.Code, r.Body.String())
	}
}
func TestPrepareMissingFFmpeg(t *testing.T) {
	a := testApp(t)
	a.cfg.ffmpeg = "/missing/ffmpeg"
	r := prepareRequest(t, a, []byte("bad image"), false)
	if r.Code != 422 || !strings.Contains(r.Body.String(), "FFmpeg is unavailable") {
		t.Fatal(r.Code, r.Body.String())
	}
}

func TestConversionNoticePersists(t *testing.T) {
	a := testApp(t)
	var body bytes.Buffer
	m := multipart.NewWriter(&body)
	p, _ := m.CreateFormFile("image", "converted.png")
	png.Encode(p, image.NewRGBA(image.Rect(0, 0, 16, 16)))
	m.WriteField("settings", `{"box":[0,0,16,16],"camera":[100,100,8,8]}`)
	m.WriteField("preparation_note", "FFmpeg resized this photo.")
	m.Close()
	r := httptest.NewRequest("POST", "/api/jobs", &body)
	r.Header.Set("Content-Type", m.FormDataContentType())
	w := httptest.NewRecorder()
	a.routes().ServeHTTP(w, r)
	if w.Code != 202 {
		t.Fatal(w.Code, w.Body.String())
	}
	var j job
	if e := json.Unmarshal(w.Body.Bytes(), &j); e != nil {
		t.Fatal(e)
	}
	restored, e := loadApp(a.cfg)
	if e != nil {
		t.Fatal(e)
	}
	if restored.jobs[j.ID].Preparation != "FFmpeg resized this photo." {
		t.Fatal("conversion notice lost")
	}
}
func TestConversionCommandIsClosed(t *testing.T) {
	args := conversionArgs(4000)
	s := strings.Join(args, " ")
	for _, want := range []string{"-protocol_whitelist fd,pipe", "-fd 0 -i fd:", "-frames:v 1", "-max_alloc 268435456", "-threads 2"} {
		if !strings.Contains(s, want) {
			t.Fatal(want)
		}
	}
	if strings.Contains(s, "file,") || strings.Contains(s, "https") {
		t.Fatal("unbounded input protocol")
	}
}
func TestCappedFFmpegDiagnostics(t *testing.T) {
	b := cappedBuffer{limit: 3}
	n, e := b.Write([]byte("123456"))
	if e != nil || n != 6 || b.String() != "123" || !b.exceeded {
		t.Fatal("unbounded output")
	}
	b.Write([]byte("more"))
	if b.Len() != 3 {
		t.Fatal("cap exceeded")
	}
}
func TestFFmpegPhotoConversion(t *testing.T) {
	if _, e := exec.LookPath("ffmpeg"); e != nil {
		t.Skip("optional FFmpeg not installed")
	}
	a := testApp(t)
	im := image.NewPaletted(image.Rect(0, 0, 16, 12), color.Palette{color.RGBA{255, 0, 0, 255}, color.RGBA{0, 0, 0, 0}})
	im.SetColorIndex(0, 0, 1)
	var b bytes.Buffer
	if e := gif.Encode(&b, im, nil); e != nil {
		t.Fatal(e)
	}
	r := prepareRequest(t, a, b.Bytes(), false)
	if r.Code != 200 {
		t.Fatal(r.Code, r.Body.String())
	}
	if r.Header().Get("X-Sam3d-Converted") != "true" || !strings.Contains(r.Header().Get("X-Sam3d-Notice"), "FFmpeg") {
		t.Fatal("missing notice")
	}
	converted, e := decodeImage(r.Body.Bytes())
	if e != nil {
		t.Fatal(e)
	}
	red, g, blue, _ := converted.At(0, 0).RGBA()
	if red|g|blue != 0 {
		t.Fatal("transparency not composited on black")
	}
	bad := prepareRequest(t, a, []byte("#EXTM3U\nhttp://127.0.0.1/private"), false)
	if bad.Code != 422 || !strings.Contains(bad.Body.String(), "FFmpeg conversion failed") {
		t.Fatal(bad.Code, bad.Body.String())
	}
}
func TestFFmpegOversizedDimensions(t *testing.T) {
	if _, e := exec.LookPath("ffmpeg"); e != nil {
		t.Skip("optional FFmpeg not installed")
	}
	a := testApp(t)
	// Very compressible PNG, but more decoded pixels than the 16 MP bound.
	var b bytes.Buffer
	if e := png.Encode(&b, image.NewGray(image.Rect(0, 0, 5000, 3500))); e != nil {
		t.Fatal(e)
	}
	r := prepareRequest(t, a, b.Bytes(), false)
	if r.Code != 200 {
		t.Fatal(r.Code, r.Body.String())
	}
	im, _, e := image.DecodeConfig(r.Body)
	if e != nil {
		t.Fatal(e)
	}
	if im.Width != 4000 || im.Height != 2800 {
		t.Fatal(im)
	}
	if !strings.Contains(r.Header().Get("X-Sam3d-Notice"), "resized") {
		t.Fatal("missing resize notice")
	}
}
func TestPrepareCancellation(t *testing.T) {
	if _, e := exec.LookPath("ffmpeg"); e != nil {
		t.Skip("optional FFmpeg not installed")
	}
	a := testApp(t)
	f, e := os.CreateTemp(t.TempDir(), "photo-")
	if e != nil {
		t.Fatal(e)
	}
	defer f.Close()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, e = a.convertPhoto(ctx, f, 4000)
	if e == nil {
		t.Fatal("cancelled conversion succeeded")
	}
}
