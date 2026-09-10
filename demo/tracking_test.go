package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"image"
	"image/png"
	"math"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func createTestTrack(t *testing.T, a *app, mode string) track {
	t.Helper()
	w := httptest.NewRecorder()
	a.routes().ServeHTTP(w, httptest.NewRequest("POST", "/api/tracks", strings.NewReader(`{"name":"test","mode":"`+mode+`","hz":10,"width":8,"height":8}`)))
	if w.Code != 201 {
		t.Fatalf("create: %d %s", w.Code, w.Body)
	}
	var result track
	if e := json.Unmarshal(w.Body.Bytes(), &result); e != nil {
		t.Fatal(e)
	}
	return result
}
func postTestFrame(a *app, id, stamp string, ctx context.Context) *httptest.ResponseRecorder {
	var b bytes.Buffer
	png.Encode(&b, image.NewRGBA(image.Rect(0, 0, 8, 8)))
	u := "/api/tracks/" + id + "/frame?time=" + stamp + "&settings=" + url.QueryEscape(`{"box":[0,0,8,8],"camera":[100,100,4,4]}`)
	r := httptest.NewRequest("POST", u, &b).WithContext(ctx)
	w := httptest.NewRecorder()
	a.routes().ServeHTTP(w, r)
	return w
}
func TestTrackingValidation(t *testing.T) {
	a := testApp(t)
	for _, v := range []string{`{}`, `{"mode":"live","hz":31,"width":8,"height":8}`, `{"mode":"live","hz":10,"width":10000,"height":8}`, `{"mode":"live","hz":10,"width":8,"height":8,"extra":true}`, `{"mode":"live","hz":1,"width":8,"height":8} {}`} {
		w := httptest.NewRecorder()
		a.routes().ServeHTTP(w, httptest.NewRequest("POST", "/api/tracks", strings.NewReader(v)))
		if w.Code != 400 {
			t.Fatal(w.Code, v)
		}
	}
	tk := createTestTrack(t, a, "offline")
	for _, stamp := range []string{"NaN", "Inf", "-1", "999999"} {
		if w := postTestFrame(a, tk.ID, stamp, context.Background()); w.Code != 400 {
			t.Fatal(w.Code, stamp)
		}
	}
	if w := postTestFrame(a, tk.ID, "0", context.Background()); w.Code != 429 {
		t.Fatal("must not queue without available worker", w.Code)
	}
	for _, path := range []string{"/tracks/" + tk.ID + "/000000.bin", "/tracks/" + tk.ID + "/image.input", "/tracks/unknown/track.json"} {
		w := httptest.NewRecorder()
		a.routes().ServeHTTP(w, httptest.NewRequest("GET", path, nil))
		if w.Code != 404 {
			t.Fatal(path, w.Code)
		}
	}
}
func trackingWorker(t *testing.T) (*app, context.Context) {
	t.Helper()
	t.Setenv("SAM3D_FAKE_WORKER", "1")
	exe, e := os.Executable()
	if e != nil {
		t.Fatal(e)
	}
	a := testApp(t)
	a.cfg.runner = exe
	a.cfg.backend = "CPU"
	a.cfg.persistent = true
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { a.worker(ctx); close(done) }()
	t.Cleanup(func() { cancel(); <-done })
	return a, ctx
}
func successfulFrame(t *testing.T, a *app, id, stamp string, ctx context.Context) *httptest.ResponseRecorder {
	t.Helper()
	for i := 0; i < 100; i++ {
		w := postTestFrame(a, id, stamp, ctx)
		if w.Code != 429 {
			if w.Code != 200 {
				t.Fatalf("frame %d: %s", w.Code, w.Body)
			}
			return w
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("worker unavailable")
	return nil
}
func TestTrackingNativeWorkerPersistenceAndFormat(t *testing.T) {
	a, ctx := trackingWorker(t)
	tk := createTestTrack(t, a, "offline")
	w := successfulFrame(t, a, tk.ID, "0", ctx)
	if w.Body.Len() != trackFrameBytes+36874*3*4 || string(w.Body.Bytes()[:8]) != "S3DTRK01" {
		t.Fatal("frame protocol")
	}
	if !strings.Contains(w.Header().Get("Server-Timing"), "server;dur=") || !strings.Contains(w.Header().Get("Server-Timing"), "worker;dur=") {
		t.Fatal("missing stage timing")
	}
	if !strings.Contains(w.Header().Get("Server-Timing"), "infer;dur=12.500") {
		t.Fatal("native timing not attributed to this frame")
	}
	w = successfulFrame(t, a, tk.ID, "0.1", ctx)
	if w.Body.Len() != trackFrameBytes {
		t.Fatal("repeated topology")
	}
	if got := math.Float64frombits(binary.LittleEndian.Uint64(w.Body.Bytes()[8:16])); got != .1 {
		t.Fatal(got)
	}
	if w = postTestFrame(a, tk.ID, "0.05", ctx); w.Code != 400 {
		t.Fatal("nonmonotonic timestamp", w.Code)
	}
	w = httptest.NewRecorder()
	a.routes().ServeHTTP(w, httptest.NewRequest("POST", "/api/tracks/"+tk.ID+"/finish", nil))
	if w.Code != 200 {
		t.Fatal(w.Code)
	}
	if w = postTestFrame(a, tk.ID, "0.2", ctx); w.Code != 409 {
		t.Fatal("finished session accepted", w.Code)
	}
	b, e := os.ReadFile(filepath.Join(a.trackDir(tk.ID), "000001.bin"))
	if e != nil || len(b) != trackFrameBytes {
		t.Fatal("saved frame", e)
	}
	expected, e := parseResult(bytes.NewReader(syntheticResult()))
	if e != nil || !bytes.Equal(b, compactFrame(expected, .1, false)) {
		t.Fatal("saved final vertices/joints/camera differ from complete native output", e)
	}
	loaded, e := loadApp(a.cfg)
	if e != nil {
		t.Fatal(e)
	}
	saved := loaded.tracks[tk.ID]
	if saved.Count != 2 || len(saved.Samples) != 2 || saved.State != "complete" {
		t.Fatal(saved)
	}
	if dirs, _ := filepath.Glob(filepath.Join(a.cfg.data, ".video-frame-*")); len(dirs) != 0 {
		t.Fatal("scratch not cleaned")
	}
}
func TestLiveCapAndNoHistory(t *testing.T) {
	a, ctx := trackingWorker(t)
	tk := createTestTrack(t, a, "live")
	successfulFrame(t, a, tk.ID, "0", ctx)
	a.mu.Lock()
	a.tracks[tk.ID].last = time.Now()
	a.mu.Unlock()
	if w := postTestFrame(a, tk.ID, "0.2", ctx); w.Code != 429 {
		t.Fatal("live cap", w.Code)
	}
	if _, e := os.Stat(a.trackDir(tk.ID)); !os.IsNotExist(e) {
		t.Fatal("live frames persisted")
	}
	a.mu.Lock()
	if len(a.tracks[tk.ID].Samples) != 0 {
		t.Fatal("unbounded live metadata")
	}
	a.mu.Unlock()
	w := httptest.NewRecorder()
	a.routes().ServeHTTP(w, httptest.NewRequest("POST", "/api/tracks/"+tk.ID+"/finish", nil))
	a.mu.Lock()
	_, exists := a.tracks[tk.ID]
	a.mu.Unlock()
	if exists {
		t.Fatal("live session retained")
	}
}
func TestTrackingCancellationReapsWorker(t *testing.T) {
	t.Setenv("SAM3D_FAKE_IMAGE_HANG", "1")
	a, _ := trackingWorker(t)
	tk := createTestTrack(t, a, "offline")
	ctx, cancel := context.WithTimeout(context.Background(), 300*time.Millisecond)
	defer cancel()
	var w *httptest.ResponseRecorder
	for ctx.Err() == nil {
		w = postTestFrame(a, tk.ID, "0", ctx)
		if w.Code != 429 {
			break
		}
		time.Sleep(time.Millisecond)
	}
	if w == nil || (w.Code != 408 && w.Code != 500) {
		t.Fatal("cancel not surfaced", w)
	}
	// The HTTP request does not release ownership until the worker has reaped.
	a.mu.Lock()
	busy, count := a.tracks[tk.ID].busy, a.tracks[tk.ID].Count
	a.mu.Unlock()
	if busy || count != 0 {
		t.Fatal("cancel published or busy")
	}
	if dirs, _ := filepath.Glob(filepath.Join(a.cfg.data, ".video-frame-*")); len(dirs) != 0 {
		t.Fatal("cancel scratch not cleaned")
	}
}
