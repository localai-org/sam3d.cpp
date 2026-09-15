package main

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

func saveSkeletonFrame(dir string, index int, b *bodyResult, stamp float64) error {
	f, e := skeletonFromResult(b, stamp)
	if e != nil {
		return e
	}
	data, e := encodeSkeleton(f)
	if e != nil {
		return e
	}
	return os.WriteFile(filepath.Join(dir, fmt.Sprintf("%06d.pose", index)), data, 0600)
}

// Call with a.mu held. A live session owns at most one independent saved take.
func (a *app) finishRecording(session *track, state string) error {
	t := session.Recording
	if t == nil {
		return nil
	}
	t.State = state
	if e := a.saveTrack(t); e != nil {
		return e
	}
	session.Recording = nil
	return nil
}
func (a *app) appendRecording(session, t *track, b *bodyResult, r *frameRequest) error {
	used, e := storageSize(a.cfg.data)
	if e != nil {
		return e
	}
	if used+(2<<20) > a.cfg.storage {
		if e = a.finishRecording(session, "interrupted"); e != nil {
			return e
		}
		return fmt.Errorf("recording stopped: storage budget exhausted; completed poses retained")
	}
	dir := a.trackDir(t.ID)
	if t.Count == 0 {
		if e = os.WriteFile(filepath.Join(dir, "preview.jpg"), r.preview, 0600); e != nil {
			return e
		}
	}
	stamp := r.stamp - t.Start
	if e = saveSkeletonFrame(dir, t.Count, b, stamp); e != nil {
		return e
	}
	t.Samples = append(t.Samples, trackSample{stamp, r.s})
	t.Count++
	if t.Count >= maxTrackFrames {
		return a.finishRecording(session, "complete")
	}
	return a.saveTrack(t)
}
func (a *app) skeletonRoutes(m *http.ServeMux) {
	m.HandleFunc("POST /api/tracks/{id}/record", func(w http.ResponseWriter, r *http.Request) {
		var in struct {
			Name string  `json:"name"`
			Time float64 `json:"time"`
		}
		if strictJSON(http.MaxBytesReader(w, r.Body, 4096), &in) != nil || !finite64(in.Time) || in.Time < 0 || in.Time > 86400 || len(in.Name) > 160 {
			fail(w, 400, "invalid recording name/time")
			return
		}
		a.mu.Lock()
		defer a.mu.Unlock()
		session := a.tracks[r.PathValue("id")]
		if session == nil || session.Mode != "live" || session.State != "recording" || session.Recording != nil {
			fail(w, 409, "live tracking must be active with no recording in progress")
			return
		}
		saved := 0
		for _, t := range a.tracks {
			if t.Mode != "live" {
				saved++
			}
		}
		if saved >= a.cfg.maxJobs {
			fail(w, 429, "saved take limit reached")
			return
		}
		used, e := storageSize(a.cfg.data)
		if e != nil || used+(2<<20) > a.cfg.storage {
			fail(w, 507, "storage budget exhausted")
			return
		}
		var nonce [12]byte
		if _, e = rand.Read(nonce[:]); e != nil {
			fail(w, 500, e)
			return
		}
		name := strings.TrimSpace(in.Name)
		if name == "" {
			name = "Webcam take " + time.Now().Format("15:04:05")
		}
		t := &track{ID: hex.EncodeToString(nonce[:]), Name: name, Mode: "take", Hz: session.Hz, Width: session.Width, Height: session.Height, Created: time.Now().UTC(), State: "recording", Precision: session.Precision, Provenance: session.Provenance, Skeleton: true, Start: in.Time, Samples: []trackSample{}}
		if e = os.Mkdir(a.trackDir(t.ID), 0700); e != nil {
			fail(w, 500, e)
			return
		}
		if e = a.saveTrack(t); e != nil {
			os.RemoveAll(a.trackDir(t.ID))
			fail(w, 500, e)
			return
		}
		a.tracks[t.ID] = t
		session.Recording = t
		send(w, 201, t)
	})
	m.HandleFunc("POST /api/tracks/{id}/record/stop", func(w http.ResponseWriter, r *http.Request) {
		a.mu.Lock()
		defer a.mu.Unlock()
		session := a.tracks[r.PathValue("id")]
		if session == nil || session.Recording == nil {
			fail(w, 409, "no active recording")
			return
		}
		t := session.Recording
		if e := a.finishRecording(session, "complete"); e != nil {
			fail(w, 500, e)
			return
		}
		send(w, 200, t)
	})
	m.HandleFunc("PATCH /api/tracks/{id}", func(w http.ResponseWriter, r *http.Request) {
		var in struct {
			Name string `json:"name"`
		}
		if strictJSON(http.MaxBytesReader(w, r.Body, 4096), &in) != nil || strings.TrimSpace(in.Name) == "" || len(in.Name) > 160 {
			fail(w, 400, "name must contain 1–160 bytes")
			return
		}
		a.mu.Lock()
		defer a.mu.Unlock()
		t := a.tracks[r.PathValue("id")]
		if t == nil || t.Mode == "live" {
			fail(w, 404, "take not found")
			return
		}
		t.Name = strings.TrimSpace(in.Name)
		if e := a.saveTrack(t); e != nil {
			fail(w, 500, e)
			return
		}
		send(w, 200, t)
	})
	m.HandleFunc("DELETE /api/tracks/{id}", func(w http.ResponseWriter, r *http.Request) {
		a.mu.Lock()
		defer a.mu.Unlock()
		t := a.tracks[r.PathValue("id")]
		if t == nil || t.Mode == "live" {
			fail(w, 404, "take not found")
			return
		}
		if t.State == "recording" || t.busy {
			fail(w, 409, "stop recording before deleting this take")
			return
		}
		if e := os.RemoveAll(a.trackDir(t.ID)); e != nil {
			fail(w, 500, e)
			return
		}
		delete(a.tracks, t.ID)
		send(w, 200, map[string]bool{"deleted": true})
	})
	m.HandleFunc("GET /api/tracks/{id}/skeleton.glb", a.exportTrackSkeleton)
	m.HandleFunc("GET /api/jobs/{id}/skeleton.glb", func(w http.ResponseWriter, r *http.Request) {
		id := r.PathValue("id")
		if !validID.MatchString(id) {
			fail(w, 404, "photo not found")
			return
		}
		a.mu.Lock()
		j := a.jobs[id]
		ok := j != nil && j.State == "complete"
		a.mu.Unlock()
		if !ok {
			fail(w, 409, "photo is not complete")
			return
		}
		file, e := os.Open(filepath.Join(a.dir(id), "result.json"))
		if e != nil {
			fail(w, 404, e)
			return
		}
		defer file.Close()
		var b bodyResult
		if e = json.NewDecoder(file).Decode(&b); e != nil {
			fail(w, 422, e)
			return
		}
		f, e := skeletonFromResult(&b, 0)
		if e != nil {
			fail(w, 422, e)
			return
		}
		data, e := skeletonGLB([]skeletonFrame{f}, movementQuery(r))
		if e != nil {
			fail(w, 400, e)
			return
		}
		sendGLB(w, id, data)
	})
}
func movementQuery(r *http.Request) string {
	v := r.URL.Query().Get("movement")
	if v == "" {
		v = "in-place"
	}
	return v
}
func sendGLB(w http.ResponseWriter, id string, data []byte) {
	w.Header().Set("Content-Type", "model/gltf-binary")
	w.Header().Set("Content-Disposition", fmt.Sprintf(`attachment; filename="sam3d-%s.glb"`, id))
	w.Write(data)
}
func (a *app) exportTrackSkeleton(w http.ResponseWriter, r *http.Request) {
	start, end := 0., 86400.
	for key, dest := range map[string]*float64{"start": &start, "end": &end} {
		if text := r.URL.Query().Get(key); text != "" {
			v, e := strconv.ParseFloat(text, 64)
			if e != nil || !finite64(v) || v < 0 || v > 86400 {
				fail(w, 400, "invalid trim interval")
				return
			}
			*dest = v
		}
	}
	if start > end {
		fail(w, 400, "trim start must precede end")
		return
	}
	frames, e := func() ([]skeletonFrame, error) {
		a.mu.Lock()
		defer a.mu.Unlock()
		t := a.tracks[r.PathValue("id")]
		if t == nil || t.Mode == "live" || !t.Skeleton {
			return nil, fmt.Errorf("skeleton unavailable; reprocess this sequence")
		}
		if t.State == "recording" {
			return nil, fmt.Errorf("stop recording before exporting")
		}
		var frames []skeletonFrame
		for i, s := range t.Samples {
			if s.Time < start || s.Time > end {
				continue
			}
			f, e := readSkeleton(filepath.Join(a.trackDir(t.ID), fmt.Sprintf("%06d.pose", i)))
			if e != nil {
				return nil, e
			}
			if f.Time != s.Time {
				return nil, fmt.Errorf("skeleton timestamp mismatch")
			}
			frames = append(frames, f)
		}
		return frames, nil
	}()
	if e != nil {
		fail(w, 409, e)
		return
	}
	data, e := skeletonGLB(frames, movementQuery(r))
	if e != nil {
		fail(w, 400, e)
		return
	}
	sendGLB(w, r.PathValue("id"), data)
}
