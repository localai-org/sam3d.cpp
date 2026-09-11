package main

// Video is a bounded sequence of the same validated single-image calls. No
// alternate inference implementation, inferred calibration or temporal network.
import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"image"
	"io"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
)

const maxTrackFrames = 1800
const trackFrameBytes = 16 + (18439*3+127*3+3)*4
const liveRateJitterSeconds = 0.005

type trackSample struct {
	Time     float64  `json:"time"`
	Settings settings `json:"settings"`
}
type track struct {
	ID         string            `json:"id"`
	Name       string            `json:"name"`
	Mode       string            `json:"mode"`
	Hz         float64           `json:"hz"`
	Width      int               `json:"width"`
	Height     int               `json:"height"`
	Created    time.Time         `json:"created"`
	State      string            `json:"state"`
	Precision  string            `json:"precision"`
	Provenance map[string]string `json:"provenance"`
	Samples    []trackSample     `json:"samples"`
	Count      int               `json:"count"`
	Skeleton   bool              `json:"skeleton"`
	Start      float64           `json:"start,omitempty"`
	Recording  *track            `json:"recording,omitempty"`
	busy       bool
	last       time.Time
	lastTime   float64
	lastSubmit float64
	inflight   int
	next       uint64
	cancels    map[uint64]context.CancelFunc
}
type frameRequest struct {
	ctx     context.Context
	t       *track
	take    *track
	packed  []byte
	preview []byte
	s       settings
	stamp   float64
	reply   chan frameReply
	timings map[string]float64
}
type frameReply struct {
	data []byte
	err  error
}

func elapsedMS(start time.Time) float64 {
	return float64(time.Since(start)) / float64(time.Millisecond)
}
func timingHeader(values map[string]float64) string {
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	parts := make([]string, 0, len(keys))
	for _, key := range keys {
		parts = append(parts, fmt.Sprintf("%s;dur=%.3f", key, values[key]))
	}
	return strings.Join(parts, ", ")
}

func (a *app) trackDir(id string) string { return filepath.Join(a.cfg.data, "tracks", id) }
func (a *app) saveTrack(t *track) error {
	if t.Mode == "live" {
		return nil
	}
	return jsonFile(filepath.Join(a.trackDir(t.ID), "track.json"), t)
}
func (a *app) loadTracks() error {
	root := filepath.Join(a.cfg.data, "tracks")
	if e := os.MkdirAll(root, 0700); e != nil {
		return e
	}
	entries, e := os.ReadDir(root)
	if e != nil {
		return e
	}
	for _, entry := range entries {
		if !entry.IsDir() || !validID.MatchString(entry.Name()) {
			continue
		}
		b, e := os.ReadFile(filepath.Join(root, entry.Name(), "track.json"))
		if e != nil {
			return e
		}
		var t track
		if json.Unmarshal(b, &t) != nil || t.ID != entry.Name() || (t.Mode != "offline" && t.Mode != "take") || len(t.Samples) > maxTrackFrames {
			return fmt.Errorf("invalid saved track %s", entry.Name())
		}
		if t.State == "recording" {
			t.State = "interrupted"
			if e = a.saveTrack(&t); e != nil {
				return e
			}
		}
		a.tracks[t.ID] = &t
	}
	return nil
}
func strictJSON(r io.Reader, v any) error {
	d := json.NewDecoder(r)
	d.DisallowUnknownFields()
	if e := d.Decode(v); e != nil {
		return e
	}
	if d.Decode(new(any)) != io.EOF {
		return fmt.Errorf("trailing JSON")
	}
	return nil
}
func (a *app) trackRoutes(m *http.ServeMux) {
	a.skeletonRoutes(m)
	m.HandleFunc("POST /api/tracks", a.createTrack)
	m.HandleFunc("POST /api/tracks/{id}/frame", a.submitFrame)
	m.HandleFunc("POST /api/tracks/{id}/finish", func(w http.ResponseWriter, r *http.Request) {
		a.mu.Lock()
		defer a.mu.Unlock()
		t := a.tracks[r.PathValue("id")]
		if t == nil {
			fail(w, 404, "track not found")
			return
		}
		if t.Mode == "take" {
			fail(w, 409, "stop the take through its live session's record/stop endpoint")
			return
		}
		if e := a.finishRecording(t, "complete"); e != nil {
			fail(w, 500, e)
			return
		}
		t.State = "complete"
		for _, cancel := range t.cancels {
			cancel()
		}
		if e := a.saveTrack(t); e != nil {
			fail(w, 500, e)
			return
		}
		send(w, 200, t)
		if t.Mode == "live" {
			delete(a.tracks, t.ID)
		}
	})
	m.HandleFunc("GET /api/tracks", func(w http.ResponseWriter, r *http.Request) {
		a.mu.Lock()
		defer a.mu.Unlock()
		list := []*track{}
		for _, t := range a.tracks {
			if t.Mode != "live" {
				list = append(list, t)
			}
		}
		sort.Slice(list, func(i, j int) bool { return list[i].Created.After(list[j].Created) })
		send(w, 200, list)
	})
	m.HandleFunc("GET /api/tracks/{id}", func(w http.ResponseWriter, r *http.Request) {
		a.mu.Lock()
		defer a.mu.Unlock()
		t := a.tracks[r.PathValue("id")]
		if t == nil {
			fail(w, 404, "track not found")
			return
		}
		send(w, 200, t)
	})
	m.HandleFunc("GET /tracks/{id}/{name}", func(w http.ResponseWriter, r *http.Request) {
		id, name := r.PathValue("id"), r.PathValue("name")
		if !validID.MatchString(id) {
			http.NotFound(w, r)
			return
		}
		a.mu.Lock()
		t := a.tracks[id]
		allowed := t != nil && t.Mode != "live" && (name == "track.json" || name == "faces.bin" || name == "preview.jpg")
		if t != nil && t.Mode != "live" && ((len(name) == 10 && strings.HasSuffix(name, ".bin") && t.Mode == "offline") || (len(name) == 11 && strings.HasSuffix(name, ".pose") && t.Skeleton)) {
			index, e := strconv.Atoi(name[:6])
			allowed = e == nil && index >= 0 && index < len(t.Samples) && name == fmt.Sprintf("%06d%s", index, filepath.Ext(name))
		}
		a.mu.Unlock()
		if !allowed {
			http.NotFound(w, r)
			return
		}
		http.ServeFile(w, r, filepath.Join(a.trackDir(id), name))
	})
}
func (a *app) createTrack(w http.ResponseWriter, r *http.Request) {
	var in struct {
		Name   string  `json:"name"`
		Mode   string  `json:"mode"`
		Hz     float64 `json:"hz"`
		Width  int     `json:"width"`
		Height int     `json:"height"`
	}
	if strictJSON(http.MaxBytesReader(w, r.Body, 4096), &in) != nil || (in.Mode != "offline" && in.Mode != "live") || in.Hz < 1 || in.Hz > 30 || in.Width < 8 || in.Height < 8 || in.Width > 1280 || in.Height > 1280 || in.Width*in.Height > 1000000 {
		fail(w, 400, "mode offline/live; rate 1–30 Hz; frames 8–1280 px and at most 1 MP required")
		return
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	// Expire abandoned live sessions. They never create saved frame directories.
	live := 0
	offline := 0
	for id, t := range a.tracks {
		if t.Mode != "live" {
			offline++
			continue
		}
		if !t.busy && time.Since(t.last) > time.Minute {
			if e := a.finishRecording(t, "interrupted"); e != nil {
				fail(w, 500, e)
				return
			}
			delete(a.tracks, id)
		} else {
			live++
		}
	}
	if live >= 4 || (in.Mode == "offline" && offline >= a.cfg.maxJobs) {
		fail(w, 429, "track/session limit reached")
		return
	}
	used, e := storageSize(a.cfg.data)
	if e != nil || used+(16<<20) > a.cfg.storage {
		fail(w, 507, "data storage budget exhausted")
		return
	}
	var nonce [12]byte
	if _, e = rand.Read(nonce[:]); e != nil {
		fail(w, 500, e)
		return
	}
	if len(in.Name) > 160 {
		in.Name = in.Name[:160]
	}
	t := &track{ID: hex.EncodeToString(nonce[:]), Name: in.Name, Mode: in.Mode, Hz: in.Hz, Width: in.Width, Height: in.Height, Created: time.Now().UTC(), State: "recording", Skeleton: true, Precision: a.cfg.precisionName(), Provenance: a.provenance, Samples: []trackSample{}, last: time.Now()}
	if t.Mode == "offline" {
		if e = os.Mkdir(a.trackDir(t.ID), 0700); e != nil {
			fail(w, 500, e)
			return
		}
		if e = a.saveTrack(t); e != nil {
			os.RemoveAll(a.trackDir(t.ID))
			fail(w, 500, e)
			return
		}
	}
	a.tracks[t.ID] = t
	send(w, 201, t)
}
func (a *app) submitFrame(w http.ResponseWriter, r *http.Request) {
	started := time.Now()
	timings := map[string]float64{}
	uploadHeld := false
	select {
	case a.uploads <- struct{}{}:
		uploadHeld = true
		defer func() {
			if uploadHeld {
				<-a.uploads
			}
		}()
	default:
		fail(w, 429, "another image/frame is being processed")
		return
	}
	var s settings
	stamp, e := strconv.ParseFloat(r.URL.Query().Get("time"), 64)
	if e != nil || math.IsNaN(stamp) || math.IsInf(stamp, 0) || stamp < 0 || stamp > 86400 || len(r.URL.RawQuery) > 4096 || strictJSON(strings.NewReader(r.URL.Query().Get("settings")), &s) != nil {
		fail(w, 400, "invalid frame time/settings")
		return
	}
	a.mu.Lock()
	t := a.tracks[r.PathValue("id")]
	if t == nil || t.Mode == "take" || t.State != "recording" {
		a.mu.Unlock()
		fail(w, 409, "track is not recording")
		return
	}
	if !a.workerLive {
		a.mu.Unlock()
		fail(w, 429, "inference worker is not ready")
		return
	}
	limit := 1
	if t.Mode == "live" {
		limit = 2
	}
	// Browser timers and request delivery can move a nominally capped frame a
	// few milliseconds earlier. Keep a small admission tolerance; the selected
	// timestamps and bounded pipeline still constrain sustained work.
	if t.inflight >= limit || len(a.queue) > 0 || (t.Mode == "live" && (t.Count > 0 || t.inflight > 0) && time.Since(t.last).Seconds() < 1/t.Hz-liveRateJitterSeconds) {
		a.mu.Unlock()
		fail(w, 429, "live pipeline full or rate cap reached; retry with latest frame")
		return
	}
	previous := t.lastTime
	hasPrevious := t.Count > 0
	if t.inflight > 0 {
		previous = t.lastSubmit
		hasPrevious = true
	}
	if (hasPrevious && stamp-previous < 1/t.Hz-0.001) || (t.Mode == "offline" && t.Count >= maxTrackFrames) {
		a.mu.Unlock()
		fail(w, 400, "timestamps must increase at the selected sampling interval; offline limit is 1800 frames")
		return
	}
	if e = validateSettings(s, t.Width, t.Height); e != nil {
		a.mu.Unlock()
		fail(w, 400, e)
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), a.cfg.timeout)
	defer cancel()
	take := t.Recording
	if take != nil && stamp < take.Start {
		take = nil
	}
	requestID := t.next
	t.next++
	if t.cancels == nil {
		t.cancels = map[uint64]context.CancelFunc{}
	}
	t.cancels[requestID] = cancel
	t.inflight++
	t.busy = true
	t.last = time.Now()
	t.lastSubmit = stamp
	a.mu.Unlock()
	defer func() {
		a.mu.Lock()
		delete(t.cancels, requestID)
		t.inflight--
		t.busy = t.inflight > 0
		a.mu.Unlock()
	}()
	data, e := io.ReadAll(http.MaxBytesReader(w, r.Body, 2<<20))
	timings["read"] = elapsedMS(started)
	if e != nil {
		fail(w, 413, "frame exceeds 2 MiB")
		return
	}
	decodeStarted := time.Now()
	c, format, e := image.DecodeConfig(bytes.NewReader(data))
	if e != nil || (format != "jpeg" && format != "png") || c.Width != t.Width || c.Height != t.Height {
		fail(w, 400, "frame dimensions/format do not match session")
		return
	}
	im, e := decodeImage(data)
	if e != nil {
		fail(w, 400, e)
		return
	}
	timings["decode"] = elapsedMS(decodeStarted)
	packStarted := time.Now()
	packed := packImage(im, s)
	timings["pack"] = elapsedMS(packStarted)
	req := &frameRequest{ctx: ctx, t: t, take: take, packed: packed, preview: data, s: s, stamp: stamp, reply: make(chan frameReply, 1), timings: timings}
	// Live mode permits one decoded frame to wait while the preceding frame is
	// inferred. The upload token is retained through this ordered enqueue, so
	// concurrent handlers cannot reverse frame timestamps.
	select {
	case a.frames <- req:
		<-a.uploads
		uploadHeld = false
	default:
		fail(w, 429, "inference worker is busy; retry shortly")
		return
	}
	select {
	case reply := <-req.reply:
		if reply.err != nil {
			fail(w, 500, reply.err)
			return
		}
		a.mu.Lock()
		if req.take != nil {
			w.Header().Set("X-Take-ID", req.take.ID)
			w.Header().Set("X-Take-Count", strconv.Itoa(req.take.Count))
			w.Header().Set("X-Take-State", req.take.State)
		}
		a.mu.Unlock()
		w.Header().Set("Content-Type", "application/octet-stream")
		timings["server"] = elapsedMS(started)
		w.Header().Set("Server-Timing", timingHeader(timings))
		w.Write(reply.data)
	case <-ctx.Done():
		// Wait for the consumer to finish cancellation before releasing session
		// ownership. Otherwise a second request could race result persistence.
		<-req.reply
		fail(w, 408, ctx.Err())
	}
}
func compactFrame(b *bodyResult, stamp float64, faces bool) []byte {
	var out bytes.Buffer
	out.WriteString("S3DTRK01")
	binary.Write(&out, binary.LittleEndian, stamp)
	for _, key := range []string{"vertices", "joints", "camera_translation"} {
		binary.Write(&out, binary.LittleEndian, b.Tensors[key])
	}
	if faces {
		binary.Write(&out, binary.LittleEndian, b.Faces)
	}
	return out.Bytes()
}
func (a *app) runFrame(server context.Context, r *frameRequest) {
	inputStarted := time.Now()
	ctx, cancel := context.WithCancel(r.ctx)
	stop := context.AfterFunc(server, cancel)
	defer stop()
	defer cancel()
	var reply frameReply
	defer func() { r.reply <- reply }()
	// Scratch never lives in history, including on disconnect/error. os.MkdirTemp
	// supplies the exact private directory cleaned by this request only.
	dir, e := os.MkdirTemp(a.cfg.data, ".video-frame-")
	if e != nil {
		reply.err = e
		return
	}
	defer os.RemoveAll(dir)
	if e = os.WriteFile(filepath.Join(dir, "image.input"), r.packed, 0600); e == nil {
		r.timings["input_file"] = elapsedMS(inputStarted)
		workerStarted := time.Now()
		e = a.runResidentPaths(ctx, dir, func(string) {}, func(line string) {
			var value float64
			if n, err := fmt.Sscanf(line, "TIMING body_infer_ms %f", &value); err == nil && n == 1 && value >= 0 && !math.IsInf(value, 0) && !math.IsNaN(value) {
				r.timings["infer"] = value
			}
		})
		r.timings["worker"] = elapsedMS(workerStarted)
	}
	if !a.cfg.persistent {
		defer a.stopResident()
	}
	if e != nil {
		reply.err = e
		return
	}
	resultStarted := time.Now()
	f, e := os.Open(filepath.Join(dir, "result.bin"))
	if e != nil {
		reply.err = e
		return
	}
	b, e := parseResult(f)
	f.Close()
	if e != nil {
		reply.err = e
		return
	}
	// Reject extreme geometry rather than interpolating catastrophic estimates.
	for _, key := range []string{"vertices", "joints", "camera_translation"} {
		for _, v := range b.Tensors[key] {
			if math.Abs(float64(v)) > 100 {
				reply.err = fmt.Errorf("extreme %s; reselect the person", key)
				return
			}
		}
	}
	r.timings["parse"] = elapsedMS(resultStarted)
	persistStarted := time.Now()
	a.mu.Lock()
	defer a.mu.Unlock()
	if e = ctx.Err(); e != nil || r.t.State != "recording" {
		reply.err = context.Canceled
		return
	}
	t := r.t
	first := t.Count == 0
	if t.Mode == "offline" {
		used, err := storageSize(a.cfg.data)
		if err != nil || used+(16<<20) > a.cfg.storage {
			reply.err = fmt.Errorf("data storage budget exhausted; saved frames are retained")
			return
		}
		destination := a.trackDir(t.ID)
		if first {
			var indices bytes.Buffer
			binary.Write(&indices, binary.LittleEndian, b.Faces)
			if e = os.WriteFile(filepath.Join(destination, "faces.bin"), indices.Bytes(), 0600); e == nil {
				e = os.WriteFile(filepath.Join(destination, "preview.jpg"), r.preview, 0600)
			}
		}
		if e == nil {
			e = saveSkeletonFrame(destination, t.Count, b, r.stamp)
		}
		if e == nil {
			e = os.WriteFile(filepath.Join(destination, fmt.Sprintf("%06d.bin", t.Count)), compactFrame(b, r.stamp, false), 0600)
		}
		if e != nil {
			reply.err = e
			return
		}
		t.Samples = append(t.Samples, trackSample{r.stamp, r.s})
	}
	if r.take != nil && r.take.State == "recording" {
		if e = a.appendRecording(t, r.take, b, r); e != nil {
			reply.err = e
			return
		}
	}
	t.Count++
	t.lastTime = r.stamp
	if e = a.saveTrack(t); e != nil {
		reply.err = e
		return
	}
	r.timings["persist"] = elapsedMS(persistStarted)
	encodeStarted := time.Now()
	reply.data = compactFrame(b, r.stamp, first)
	r.timings["encode"] = elapsedMS(encodeStarted)
}
