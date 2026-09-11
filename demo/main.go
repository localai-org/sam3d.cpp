// Optional local web application. Inference is the native public-C-API CLI;
// Go handles bounded uploads, one worker, persistence and static mesh export.
package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"embed"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"image"
	_ "image/jpeg"
	"image/png"
	"io"
	"io/fs"
	"log"
	"math"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
)

//go:embed web
var assets embed.FS

const maxUpload = 20 << 20

var validID = regexp.MustCompile(`^[a-f0-9]{24}$`)

type settings struct {
	Box    [4]float32 `json:"box"`
	Camera [4]float32 `json:"camera"`
}
type job struct {
	ID          string            `json:"id"`
	Name        string            `json:"name"`
	State       string            `json:"state"`
	Stage       string            `json:"stage"`
	Error       string            `json:"error,omitempty"`
	Created     time.Time         `json:"created"`
	Started     time.Time         `json:"started"`
	Finished    time.Time         `json:"finished"`
	Width       int               `json:"width"`
	Height      int               `json:"height"`
	Settings    settings          `json:"settings"`
	SourceSHA   string            `json:"source_sha256"`
	InputSHA    string            `json:"native_input_sha256"`
	ResultSHA   string            `json:"result_sha256,omitempty"`
	GLBSHA      string            `json:"glb_sha256,omitempty"`
	Backend     string            `json:"backend"`
	Precision   string            `json:"precision,omitempty"`
	Provenance  map[string]string `json:"provenance"`
	Preparation string            `json:"preparation_note,omitempty"`
}
type config struct {
	bodyMode                                                                           string
	precision                                                                          string
	ffmpeg                                                                             string
	addr, data, runner, module, backend, description, backbone, branch, mhr, reference string
	device, threads, memory, reserve, maxJobs                                          int
	storage                                                                            int64
	timeout                                                                            time.Duration
	persistent                                                                         bool
	workerIdle                                                                         time.Duration
}
type app struct {
	cfg        config
	mu         sync.Mutex
	jobs       map[string]*job
	queue      chan string
	uploads    chan struct{}
	current    string
	cancel     context.CancelFunc
	provenance map[string]string
	native     *residentWorker // owned exclusively by the single queue consumer
	tracks     map[string]*track
	frames     chan *frameRequest
	workerLive bool
}

func digest(path string) (string, error) {
	f, e := os.Open(path)
	if e != nil {
		return "", e
	}
	defer f.Close()
	h := sha256.New()
	if _, e = io.Copy(h, f); e != nil {
		return "", e
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}
func jsonFile(path string, v any) error {
	b, e := json.MarshalIndent(v, "", "  ")
	if e != nil {
		return e
	}
	tmp := path + ".tmp"
	if e = os.WriteFile(tmp, append(b, '\n'), 0600); e != nil {
		return e
	}
	return os.Rename(tmp, path)
}
func (a *app) dir(id string) string { return filepath.Join(a.cfg.data, id) }
func (a *app) save(j *job) error    { return jsonFile(filepath.Join(a.dir(j.ID), "job.json"), j) }
func (a *app) stage(id, s string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	j := a.jobs[id]
	j.Stage = s
	if e := a.save(j); e != nil {
		log.Printf("persist stage: %v", e)
	}
}
func send(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}
func fail(w http.ResponseWriter, status int, e any) {
	send(w, status, map[string]any{"error": fmt.Sprint(e)})
}
func validateSettings(s settings, w, h int) error {
	for _, v := range append(s.Box[:], s.Camera[:]...) {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return fmt.Errorf("nonfinite box or camera")
		}
	}
	if s.Box[0] < 0 || s.Box[1] < 0 || s.Box[2] > float32(w) || s.Box[3] > float32(h) || s.Box[2]-s.Box[0] < 8 || s.Box[3]-s.Box[1] < 8 {
		return fmt.Errorf("select a person box at least 8 pixels wide/high inside the image")
	}
	if s.Camera[0] < 1 || s.Camera[1] < 1 || s.Camera[0] > 100000 || s.Camera[1] > 100000 || s.Camera[2] < 0 || s.Camera[2] > float32(w) || s.Camera[3] < 0 || s.Camera[3] > float32(h) {
		return fmt.Errorf("invalid camera: focal lengths 1–100000 px and principal point inside image required")
	}
	return nil
}
func decodeImage(data []byte) (image.Image, error) {
	c, format, e := image.DecodeConfig(bytes.NewReader(data))
	if e != nil {
		return nil, fmt.Errorf("upload a valid PNG or JPEG")
	}
	if (format != "jpeg" && format != "png") || c.Width < 8 || c.Height < 8 || c.Width > 32766 || c.Height > 32766 || int64(c.Width)*int64(c.Height) > 16000000 {
		return nil, fmt.Errorf("image must be 8–32766 pixels per side, at most 16 megapixels")
	}
	im, _, e := image.Decode(bytes.NewReader(data))
	return im, e
}
func storageSize(root string) (int64, error) {
	var n int64
	e := filepath.WalkDir(root, func(p string, d fs.DirEntry, e error) error {
		if e != nil {
			return e
		}
		if !d.IsDir() {
			v, e := d.Info()
			if e != nil {
				return e
			}
			n += v.Size()
		}
		return nil
	})
	return n, e
}
func (a *app) submit(w http.ResponseWriter, r *http.Request) {
	select {
	case a.uploads <- struct{}{}:
		defer func() { <-a.uploads }()
	default:
		fail(w, 429, "another image is being decoded; please retry shortly")
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, maxUpload+65536)
	if e := r.ParseMultipartForm(maxUpload + 65536); e != nil {
		fail(w, 400, "invalid upload or file exceeds 20 MiB")
		return
	}
	if r.MultipartForm != nil {
		defer r.MultipartForm.RemoveAll()
	}
	f, h, e := r.FormFile("image")
	if e != nil {
		fail(w, 400, "image is required")
		return
	}
	defer f.Close()
	data, e := io.ReadAll(io.LimitReader(f, maxUpload+1))
	if e != nil || len(data) > maxUpload {
		fail(w, 413, "image exceeds 20 MiB")
		return
	}
	var s settings
	dec := json.NewDecoder(strings.NewReader(r.FormValue("settings")))
	dec.DisallowUnknownFields()
	if e = dec.Decode(&s); e != nil {
		fail(w, 400, "invalid box/camera settings")
		return
	}
	if dec.Decode(new(any)) != io.EOF {
		fail(w, 400, "trailing settings")
		return
	}
	im, e := decodeImage(data)
	if e != nil {
		fail(w, 400, e)
		return
	}
	iw, ih := im.Bounds().Dx(), im.Bounds().Dy()
	if e = validateSettings(s, iw, ih); e != nil {
		fail(w, 400, e)
		return
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if len(a.jobs) >= a.cfg.maxJobs {
		fail(w, 507, "history limit reached; archive results from the data directory before restarting")
		return
	}
	if len(a.queue) >= cap(a.queue) {
		fail(w, 429, "inference queue is full; wait for the current jobs")
		return
	}
	used, e := storageSize(a.cfg.data)
	if e != nil {
		fail(w, 500, e)
		return
	}
	if used+int64(len(a.queue)+2)*(160<<20) > a.cfg.storage {
		fail(w, 507, "data storage budget exhausted; archive old results before restarting")
		return
	}
	var nonce [12]byte
	if _, e = rand.Read(nonce[:]); e != nil {
		fail(w, 500, e)
		return
	}
	id := hex.EncodeToString(nonce[:])
	dir := a.dir(id)
	if e = os.Mkdir(dir, 0700); e != nil {
		fail(w, 500, e)
		return
	}
	success := false
	defer func() {
		if !success {
			os.RemoveAll(dir)
		}
	}() // Only this newly-created, validated job directory.
	// Store a canonical opaque PNG of the exact RGB passed to inference. EXIF is
	// not silently applied; the client previews this orientation before submission.
	packed := packImage(im, s)
	rgb := image.NewRGBA(image.Rect(0, 0, iw, ih))
	for i, j := 52, 0; i < len(packed); i, j = i+3, j+4 {
		copy(rgb.Pix[j:j+3], packed[i:i+3])
		rgb.Pix[j+3] = 255
	}
	var imageBytes bytes.Buffer
	if e = png.Encode(&imageBytes, rgb); e != nil {
		fail(w, 500, e)
		return
	}
	if e = os.WriteFile(filepath.Join(dir, "input.png"), imageBytes.Bytes(), 0600); e != nil {
		fail(w, 500, e)
		return
	}
	if e = os.WriteFile(filepath.Join(dir, "image.input"), packed, 0600); e != nil {
		fail(w, 500, e)
		return
	}
	sourceHash := sha256.Sum256(data)
	inputHash := sha256.Sum256(packed)
	name := filepath.Base(strings.ReplaceAll(h.Filename, "\\", "/"))
	if len(name) > 160 {
		name = name[:160]
	}
	j := &job{ID: id, Name: name, State: "queued", Stage: "Queued for the single inference worker", Created: time.Now().UTC(), Width: iw, Height: ih, Settings: s, SourceSHA: hex.EncodeToString(sourceHash[:]), InputSHA: hex.EncodeToString(inputHash[:]), Backend: a.cfg.backend, Precision: a.cfg.precisionName(), Provenance: a.provenance}
	j.Preparation = r.FormValue("preparation_note")
	if len(j.Preparation) > 512 {
		j.Preparation = j.Preparation[:512]
	}
	if e = a.save(j); e != nil {
		fail(w, 500, e)
		return
	}
	a.jobs[id] = j
	a.queue <- id
	success = true
	send(w, 202, j)
}
func (a *app) routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /api/config", func(w http.ResponseWriter, r *http.Request) {
		send(w, 200, map[string]any{"model": "SAM 3D Body · DINOv3 · " + a.cfg.inferenceLabel(), "body_inference": a.cfg.bodyModeName(), "precision": a.cfg.precisionName(), "backend": a.cfg.backend, "scope": "Body pose branch — single-person video uses independent frame estimates, not a temporal model or automatic detector", "reference": a.cfg.reference != "", "max_upload_bytes": maxUpload})
	})
	a.trackRoutes(mux)
	mux.HandleFunc("POST /api/jobs", a.submit)
	mux.HandleFunc("POST /api/prepare", a.preparePhoto)
	mux.HandleFunc("GET /api/jobs", func(w http.ResponseWriter, r *http.Request) {
		a.mu.Lock()
		defer a.mu.Unlock()
		jobs := make([]*job, 0, len(a.jobs))
		for _, j := range a.jobs {
			jobs = append(jobs, j)
		}
		sort.Slice(jobs, func(i, j int) bool { return jobs[i].Created.After(jobs[j].Created) })
		send(w, 200, jobs)
	})
	mux.HandleFunc("GET /api/jobs/{id}", func(w http.ResponseWriter, r *http.Request) {
		a.mu.Lock()
		defer a.mu.Unlock()
		j := a.jobs[r.PathValue("id")]
		if j == nil {
			fail(w, 404, "job not found")
			return
		}
		send(w, 200, j)
	})
	mux.HandleFunc("DELETE /api/jobs/{id}", func(w http.ResponseWriter, r *http.Request) {
		a.mu.Lock()
		defer a.mu.Unlock()
		j := a.jobs[r.PathValue("id")]
		if j == nil {
			fail(w, 404, "job not found")
			return
		}
		if j.State == "queued" {
			j.State = "cancelled"
			j.Stage = "Cancelled"
			j.Finished = time.Now().UTC()
			if e := a.save(j); e != nil {
				fail(w, 500, e)
				return
			}
		} else if a.current == j.ID && a.cancel != nil {
			a.cancel()
		}
		send(w, 200, j)
	})
	allowed := map[string]bool{"input.png": true, "result.json": true, "body.glb": true, "body.obj": true, "job.json": true}
	mux.HandleFunc("GET /files/{id}/{name}", func(w http.ResponseWriter, r *http.Request) {
		id, name := r.PathValue("id"), r.PathValue("name")
		if !validID.MatchString(id) || !allowed[name] {
			http.NotFound(w, r)
			return
		}
		if name == "body.glb" || name == "body.obj" || name == "job.json" {
			w.Header().Set("Content-Disposition", fmt.Sprintf(`attachment; filename="%s-%s"`, id, name))
		}
		http.ServeFile(w, r, filepath.Join(a.dir(id), name))
	})
	mux.HandleFunc("GET /reference/{name}", func(w http.ResponseWriter, r *http.Request) {
		name := r.PathValue("name")
		if a.cfg.reference == "" || (name != "input.png" && name != "result.json" && name != "manifest.json") {
			http.NotFound(w, r)
			return
		}
		http.ServeFile(w, r, filepath.Join(a.cfg.reference, name))
	})
	web, _ := fs.Sub(assets, "web")
	mux.Handle("GET /", http.FileServer(http.FS(web)))
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-Content-Type-Options", "nosniff")
		w.Header().Set("Cache-Control", "no-store")
		w.Header().Set("Referrer-Policy", "no-referrer")
		w.Header().Set("Content-Security-Policy", "default-src 'self'; script-src 'self'; worker-src 'self'; style-src 'self'; img-src 'self' blob: data:; media-src 'self' blob:; connect-src 'self'; object-src 'none'; frame-ancestors 'none'")
		// No cross-origin writes to this unauthenticated local application.
		if r.Method != "GET" && r.Method != "HEAD" {
			origin := r.Header.Get("Origin")
			if origin != "" {
				u, e := url.Parse(origin)
				if e != nil || u.Host != r.Host || (u.Scheme != "http" && u.Scheme != "https") {
					fail(w, 403, "cross-origin request rejected")
					return
				}
			}
			if r.Header.Get("Sec-Fetch-Site") == "cross-site" {
				fail(w, 403, "cross-site request rejected")
				return
			}
		}
		mux.ServeHTTP(w, r)
	})
}
func (a *app) worker(ctx context.Context) {
	a.mu.Lock()
	a.workerLive = true
	a.mu.Unlock()
	defer func() {
		a.mu.Lock()
		a.workerLive = false
		a.mu.Unlock()
	}()
	defer a.stopResident()
	interval := a.cfg.workerIdle
	if interval <= 0 {
		interval = 2 * time.Minute
	}
	tick := time.NewTicker(interval)
	defer tick.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-tick.C:
			if a.native != nil && time.Since(a.native.lastUsed) >= interval {
				a.stopResident()
			}
		case request := <-a.frames:
			a.runFrame(ctx, request)
		case id := <-a.queue:
			a.mu.Lock()
			j := a.jobs[id]
			if j.State != "queued" {
				a.mu.Unlock()
				continue
			}
			runctx, cancel := context.WithTimeout(ctx, a.cfg.timeout)
			a.current = id
			a.cancel = cancel
			j.State = "running"
			j.Stage = "Checking memory budget"
			j.Started = time.Now().UTC()
			a.save(j)
			a.mu.Unlock()
			e := a.run(runctx, id)
			cancel()
			a.mu.Lock()
			a.current = ""
			a.cancel = nil
			j.Finished = time.Now().UTC()
			if e != nil {
				j.State = "failed"
				j.Stage = "Generation failed"
				j.Error = e.Error()
				if errors.Is(e, context.Canceled) {
					j.State = "cancelled"
					j.Stage = "Cancelled"
				}
			} else {
				j.State = "complete"
				j.Stage = "Body ready — static mesh and skeleton"
			}
			if e = a.save(j); e != nil {
				log.Printf("save completed job: %v", e)
			}
			a.mu.Unlock()
		}
	}
}
func checkMemory(memory, reserve int) error {
	if memory == 0 {
		return nil
	}
	f, e := os.Open("/proc/meminfo")
	if e != nil {
		return e
	}
	defer f.Close()
	s := bufio.NewScanner(f)
	for s.Scan() {
		fields := strings.Fields(s.Text())
		if len(fields) == 3 && fields[0] == "MemAvailable:" {
			kb, e := strconv.ParseInt(fields[1], 10, 64)
			if e != nil {
				return e
			}
			if kb < int64(memory+reserve)*1024 {
				return fmt.Errorf("not enough available RAM: need %d MiB job budget + %d MiB host headroom", memory, reserve)
			}
			return nil
		}
	}
	return fmt.Errorf("cannot check host memory headroom")
}
func (a *app) runOnce(ctx context.Context, id string) error {
	if e := checkMemory(a.cfg.memory, a.cfg.reserve); e != nil {
		return e
	}
	c := a.cfg
	dir := a.dir(id)
	out := filepath.Join(dir, "result.bin")
	args := c.precisionArgs([]string{c.module, c.backend, strconv.Itoa(c.device), c.description, c.backbone, c.branch, c.mhr, filepath.Join(dir, "image.input"), out, strconv.Itoa(c.threads)})
	args = c.bodyInferenceArgs(args)
	executable := c.runner
	unit := "sam3d-demo-job-" + id
	if c.memory > 0 {
		args = append([]string{"--user", "--scope", "--quiet", "--collect", "--unit=" + unit, "-p", fmt.Sprintf("MemoryMax=%dM", c.memory), "-p", fmt.Sprintf("MemoryHigh=%dM", c.memory*5/6), "-p", "MemorySwapMax=0", "-p", "OOMPolicy=kill", executable}, args...)
		executable = "systemd-run"
	}
	cmd := exec.Command(executable, args...)
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmd.Env = c.nativeEnvironment(os.Environ())
	pipe, e := cmd.StderrPipe()
	if e != nil {
		return e
	}
	cmd.Stdout = io.Discard
	logFile, e := os.Create(filepath.Join(dir, "inference.log"))
	if e != nil {
		return e
	}
	defer logFile.Close()
	if e = cmd.Start(); e != nil {
		return fmt.Errorf("start native worker: %w", e)
	}
	done := make(chan error, 1)
	go func() {
		scanner := bufio.NewScanner(pipe)
		scanner.Buffer(make([]byte, 4096), 65536)
		written := 0
		for scanner.Scan() {
			line := scanner.Text()
			if written < 1<<20 {
				n, _ := fmt.Fprintln(logFile, line)
				written += n
			}
			if strings.HasPrefix(line, "STAGE ") {
				a.stage(id, strings.TrimPrefix(line, "STAGE "))
			}
		}
		scanErr := scanner.Err()
		err := cmd.Wait()
		if err == nil {
			err = scanErr
		}
		done <- err
	}()
	select {
	case e = <-done:
	case <-ctx.Done():
		// The scope owns native children independently of the systemd-run client.
		if c.memory > 0 {
			stopCtx, stop := context.WithTimeout(context.Background(), 10*time.Second)
			exec.CommandContext(stopCtx, "systemctl", "--user", "stop", unit+".scope").Run()
			stop()
		}
		syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
		<-done
		return ctx.Err()
	}
	if e != nil {
		logFile.Sync()
		data, _ := os.ReadFile(filepath.Join(dir, "inference.log"))
		if len(data) > 4096 {
			data = data[len(data)-4096:]
		}
		return fmt.Errorf("native worker: %w\n%s", e, strings.TrimSpace(string(data)))
	}
	return nil
}
func (a *app) run(ctx context.Context, id string) error {
	var e error
	if a.cfg.persistent {
		e = a.runResident(ctx, id)
	} else {
		e = a.runOnce(ctx, id)
	}
	if e != nil {
		return e
	}
	dir := a.dir(id)
	out := filepath.Join(dir, "result.bin")
	a.stage(id, "Validating mesh and preparing downloads")
	f, e := os.Open(out)
	if e != nil {
		return e
	}
	b, e := parseResult(f)
	f.Close()
	if e != nil {
		return e
	}
	if e = jsonFile(filepath.Join(dir, "result.json"), b); e != nil {
		return e
	}
	if e = writeGLB(filepath.Join(dir, "body.glb"), b); e != nil {
		return e
	}
	if e = writeOBJ(filepath.Join(dir, "body.obj"), b); e != nil {
		return e
	}
	resultSHA, e := digest(out)
	if e != nil {
		return e
	}
	glbSHA, e := digest(filepath.Join(dir, "body.glb"))
	if e != nil {
		return e
	}
	a.mu.Lock()
	a.jobs[id].ResultSHA = resultSHA
	a.jobs[id].GLBSHA = glbSHA
	a.mu.Unlock()
	return nil
}
func loadApp(c config) (*app, error) {
	if c.precision != "" && c.precision != "f32" && c.precision != "bf16" {
		return nil, fmt.Errorf("precision must be f32 or bf16")
	}
	if e := os.MkdirAll(c.data, 0700); e != nil {
		return nil, e
	}
	a := &app{cfg: c, jobs: map[string]*job{}, queue: make(chan string, 2), uploads: make(chan struct{}, 1), provenance: map[string]string{}, tracks: map[string]*track{}, frames: make(chan *frameRequest, 1)}
	if e := a.loadTracks(); e != nil {
		return nil, e
	}
	dirs, e := os.ReadDir(c.data)
	if e != nil {
		return nil, e
	}
	for _, d := range dirs {
		if !d.IsDir() || !validID.MatchString(d.Name()) {
			continue
		}
		b, e := os.ReadFile(filepath.Join(c.data, d.Name(), "job.json"))
		if e != nil {
			return nil, e
		}
		var j job
		if e = json.Unmarshal(b, &j); e != nil || j.ID != d.Name() {
			return nil, fmt.Errorf("invalid saved job %s", d.Name())
		}
		if j.State == "running" || j.State == "queued" {
			j.State = "failed"
			j.Stage = "Interrupted"
			j.Error = "Server stopped before completion; select this input and generate again"
			j.Finished = time.Now().UTC()
			if e = a.save(&j); e != nil {
				return nil, e
			}
		}
		a.jobs[j.ID] = &j
	}
	return a, nil
}
func main() {
	c := config{}
	flag.StringVar(&c.bodyMode, "body-inference", "standard", "standard, no-correctives, fast512, fast448 or fast384; fast modes change estimates")
	flag.StringVar(&c.ffmpeg, "ffmpeg", "ffmpeg", "optional FFmpeg executable for photo conversion (requires fd: protocol)")
	flag.StringVar(&c.addr, "listen", "127.0.0.1:8097", "HTTP bind address; no authentication, expose only to a trusted network")
	flag.StringVar(&c.data, "data", "generated/demo", "private persistent input/output directory")
	flag.StringVar(&c.runner, "runner", "build/vulkan-optimized/bin/sam3d-body-infer", "native public C API executable")
	flag.BoolVar(&c.persistent, "persistent-worker", true, "reuse the bounded native model session across images")
	flag.DurationVar(&c.workerIdle, "worker-idle", 2*time.Minute, "release resident RAM/VRAM after this idle period (within one timer interval)")
	flag.StringVar(&c.module, "module", "", "explicit GGML backend module")
	flag.StringVar(&c.backend, "backend", "Vulkan", "CPU or Vulkan")
	flag.StringVar(&c.precision, "precision", "f32", "f32 or bf16 encoder (decoder/MHR remain F32); Vulkan BF16 requires the patched CM2 build")
	flag.StringVar(&c.description, "device-name", "-", "required exact GGML device name, or -")
	flag.IntVar(&c.device, "device", 0, "backend device index")
	flag.IntVar(&c.threads, "threads", 6, "CPU threads")
	flag.StringVar(&c.backbone, "backbone", "", "backbone GGUF")
	flag.StringVar(&c.branch, "branch", "", "body pose branch GGUF")
	flag.StringVar(&c.mhr, "mhr", "", "MHR GGUF")
	flag.StringVar(&c.reference, "reference", "", "optional prepared upstream comparison directory")
	flag.IntVar(&c.memory, "memory-mib", 6144, "hard per-worker RAM cap via systemd user scope; 0 only with an external memory limit")
	flag.IntVar(&c.reserve, "reserve-mib", 10240, "additional required free host RAM before starting a worker")
	flag.IntVar(&c.maxJobs, "max-jobs", 100, "maximum retained history entries")
	flag.Int64Var(&c.storage, "storage-bytes", 2<<30, "maximum data budget, including space reserved for pending jobs")
	flag.DurationVar(&c.timeout, "timeout", 10*time.Minute, "maximum job runtime")
	flag.Parse()
	if e := c.validateBodyMode(); e != nil {
		log.Fatal(e)
	}
	if (c.backend != "CPU" && c.backend != "Vulkan") || c.threads < 1 || c.threads > 256 || c.device < 0 || c.maxJobs < 1 || c.storage < 320<<20 || c.timeout < time.Second || c.memory < 0 || (c.memory > 0 && c.memory < 64) || c.reserve < 64 || c.workerIdle < time.Second {
		log.Fatal("invalid runtime limits or backend")
	}
	for _, p := range []*string{&c.data, &c.runner, &c.module, &c.backbone, &c.branch, &c.mhr} {
		if *p == "" {
			log.Fatal("--module, --backbone, --branch and --mhr are required")
		}
		v, e := filepath.Abs(*p)
		if e != nil {
			log.Fatal(e)
		}
		*p = v
	}
	if c.reference != "" {
		var e error
		c.reference, e = filepath.Abs(c.reference)
		if e != nil {
			log.Fatal(e)
		}
	}
	a, e := loadApp(c)
	if e != nil {
		log.Fatal(e)
	}
	for label, path := range map[string]string{"runner": c.runner, "module": c.module, "backbone": c.backbone, "branch": c.branch, "mhr": c.mhr} {
		sha, e := digest(path)
		if e != nil {
			log.Fatal(e)
		}
		a.provenance[label+"_sha256"] = sha
	}
	a.provenance["precision"] = c.precisionName()
	a.provenance["precision_scope"] = c.precisionLabel()
	a.provenance["body_inference"] = c.bodyModeName()
	if c.memory > 0 {
		if _, e = exec.LookPath("systemd-run"); e != nil {
			log.Fatal("systemd-run required for memory cap; use an external cap and explicitly set --memory-mib 0 otherwise")
		}
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	workerDone := make(chan struct{})
	go func() { a.worker(ctx); close(workerDone) }()
	server := &http.Server{Addr: c.addr, Handler: a.routes(), ReadHeaderTimeout: 10 * time.Second, ReadTimeout: 60 * time.Second, WriteTimeout: 60 * time.Second, IdleTimeout: 90 * time.Second, MaxHeaderBytes: 16384}
	go func() {
		<-ctx.Done()
		shutdown, done := context.WithTimeout(context.Background(), 15*time.Second)
		defer done()
		server.Shutdown(shutdown)
	}()
	log.Printf("SAM 3D Body demo: http://%s (body pose branch; one %d MiB-bounded %s worker)", c.addr, c.memory, c.backend)
	if e = server.ListenAndServe(); e != http.ErrServerClosed {
		log.Fatal(e)
	}
	<-workerDone
}
