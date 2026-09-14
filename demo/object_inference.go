package main

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"
)

// submitObject accepts an opaque scene image plus a same-sized binary mask.
// The browser owns interactive selection; the server canonicalizes both into
// the small S3DOBJ01 container consumed by the native runtime.
func (a *app) submitObject(w http.ResponseWriter, r *http.Request) {
	select {
	case a.uploads <- struct{}{}:
		defer func() { <-a.uploads }()
	default:
		fail(w, 429, "another image is being decoded; please retry shortly")
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, 2*maxUpload+131072)
	if err := r.ParseMultipartForm(2*maxUpload + 131072); err != nil {
		fail(w, 400, "invalid image/mask upload")
		return
	}
	if r.MultipartForm != nil {
		defer r.MultipartForm.RemoveAll()
	}
	readPart := func(name string) ([]byte, *multipart.FileHeader, error) {
		f, h, err := r.FormFile(name)
		if err != nil {
			return nil, nil, fmt.Errorf("%s is required", name)
		}
		defer f.Close()
		data, err := io.ReadAll(io.LimitReader(f, maxUpload+1))
		if err != nil || len(data) > maxUpload {
			return nil, nil, fmt.Errorf("%s exceeds 20 MiB", name)
		}
		return data, h, nil
	}
	imageData, header, err := readPart("image")
	if err != nil {
		fail(w, 400, err)
		return
	}
	maskData, _, err := readPart("mask")
	if err != nil {
		fail(w, 400, err)
		return
	}
	source, err := decodeImage(imageData)
	if err != nil {
		fail(w, 400, err)
		return
	}
	mask, err := decodeImage(maskData)
	if err != nil {
		fail(w, 400, "mask must be a valid PNG")
		return
	}
	wid, hei := source.Bounds().Dx(), source.Bounds().Dy()
	if mask.Bounds().Dx() != wid || mask.Bounds().Dy() != hei {
		fail(w, 400, "mask dimensions must match the image")
		return
	}
	rgba := image.NewNRGBA(image.Rect(0, 0, wid, hei))
	draw.Draw(rgba, rgba.Bounds(), source, source.Bounds().Min, draw.Src)
	binaryMask := image.NewNRGBA(rgba.Bounds())
	minX, minY, maxX, maxY := wid, hei, -1, -1
	packed := make([]byte, 16+wid*hei*4)
	copy(packed[:8], "S3DOBJ01")
	binary.LittleEndian.PutUint32(packed[8:12], uint32(wid))
	binary.LittleEndian.PutUint32(packed[12:16], uint32(hei))
	for y := 0; y < hei; y++ {
		for x := 0; x < wid; x++ {
			i := y*wid + x
			r, g, b, _ := rgba.At(x, y).RGBA()
			_, _, _, alpha := mask.At(x+mask.Bounds().Min.X, y+mask.Bounds().Min.Y).RGBA()
			selected := alpha >= 0x8000
			packed[16+4*i+0] = byte(r >> 8)
			packed[16+4*i+1] = byte(g >> 8)
			packed[16+4*i+2] = byte(b >> 8)
			if selected {
				packed[16+4*i+3] = 255
				binaryMask.SetNRGBA(x, y, color.NRGBA{R: 255, G: 255, B: 255, A: 255})
				if x < minX {
					minX = x
				}
				if y < minY {
					minY = y
				}
				if x > maxX {
					maxX = x
				}
				if y > maxY {
					maxY = y
				}
			}
		}
	}
	if maxX-minX < 2 || maxY-minY < 2 {
		fail(w, 400, "paint an object mask at least three pixels wide and high")
		return
	}

	a.mu.Lock()
	defer a.mu.Unlock()
	if len(a.jobs) >= a.cfg.maxJobs || len(a.queue) >= cap(a.queue) {
		fail(w, 429, "inference queue or history is full")
		return
	}
	used, err := storageSize(a.cfg.data)
	if err != nil {
		fail(w, 500, err)
		return
	}
	if used+int64(len(a.queue)+2)*(160<<20) > a.cfg.storage {
		fail(w, 507, "data storage budget exhausted")
		return
	}
	var nonce [12]byte
	if _, err = rand.Read(nonce[:]); err != nil {
		fail(w, 500, err)
		return
	}
	id := hex.EncodeToString(nonce[:])
	dir := a.dir(id)
	if err = os.Mkdir(dir, 0700); err != nil {
		fail(w, 500, err)
		return
	}
	success := false
	defer func() {
		if !success {
			os.RemoveAll(dir)
		}
	}()
	var scenePNG, maskPNG bytes.Buffer
	if err = png.Encode(&scenePNG, rgba); err == nil {
		err = png.Encode(&maskPNG, binaryMask)
	}
	if err == nil {
		err = os.WriteFile(filepath.Join(dir, "input.png"), scenePNG.Bytes(), 0600)
	}
	if err == nil {
		err = os.WriteFile(filepath.Join(dir, "mask.png"), maskPNG.Bytes(), 0600)
	}
	if err == nil {
		err = os.WriteFile(filepath.Join(dir, "object.input"), packed, 0600)
	}
	if err != nil {
		fail(w, 500, err)
		return
	}
	inputHash := sha256.Sum256(packed)
	sourceHash := sha256.Sum256(imageData)
	name := filepath.Base(strings.ReplaceAll(header.Filename, "\\", "/"))
	if len(name) > 160 {
		name = name[:160]
	}
	j := &job{ID: id, Kind: "object", Name: name, State: "queued",
		Stage: "Queued for SAM 3D Objects inference", Created: time.Now().UTC(),
		Width: wid, Height: hei, SourceSHA: hex.EncodeToString(sourceHash[:]),
		InputSHA: hex.EncodeToString(inputHash[:]),
		Backend:  a.cfg.backend, Precision: "f16 weights / f32 compute", Provenance: a.provenance}
	j.Preparation = r.FormValue("preparation_note")
	if len(j.Preparation) > 512 {
		j.Preparation = j.Preparation[:512]
	}
	if err = a.save(j); err != nil {
		fail(w, 500, err)
		return
	}
	a.jobs[id] = j
	a.queue <- id
	success = true
	send(w, 202, j)
}

func (a *app) runObject(ctx context.Context, id string) error {
	if err := checkMemory(a.cfg.memory, a.cfg.reserve); err != nil {
		return err
	}
	a.stopResident() // Object inference owns the GPU and streams several models.
	dir := a.dir(id)
	out := filepath.Join(dir, "object.glb")
	args := []string{a.cfg.module, a.cfg.backend, strconv.Itoa(a.cfg.device),
		a.cfg.description, a.cfg.objectModels, filepath.Join(dir, "object.input"), out,
		strconv.Itoa(a.cfg.threads), "f16", "42"}
	executable := a.cfg.objectRunner
	unit := "sam3d-demo-object-" + id
	if a.cfg.memory > 0 {
		args = append([]string{"--user", "--scope", "--quiet", "--collect", "--unit=" + unit,
			"-p", fmt.Sprintf("MemoryMax=%dM", a.cfg.memory),
			"-p", fmt.Sprintf("MemoryHigh=%dM", a.cfg.memory*5/6),
			"-p", "MemorySwapMax=0", "-p", "OOMPolicy=kill", executable}, args...)
		executable = "systemd-run"
	}
	a.stage(id, "Running SAM 3D Objects generators")
	cmd := exec.Command(executable, args...)
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmd.Env = a.cfg.nativeEnvironment(os.Environ())
	var output bytes.Buffer
	cmd.Stdout = &output
	cmd.Stderr = &output
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("start native object worker: %w", err)
	}
	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()
	var runErr error
	select {
	case runErr = <-done:
	case <-ctx.Done():
		if a.cfg.memory > 0 {
			stopCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			exec.CommandContext(stopCtx, "systemctl", "--user", "stop", unit+".scope").Run()
			cancel()
		}
		if cmd.Process != nil {
			syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
		}
		<-done
		return ctx.Err()
	}
	outputBytes := output.Bytes()
	if len(outputBytes) > 1<<20 {
		outputBytes = outputBytes[len(outputBytes)-(1<<20):]
	}
	_ = os.WriteFile(filepath.Join(dir, "inference.log"), outputBytes, 0600)
	if runErr != nil {
		return fmt.Errorf("native object worker: %w\n%s", runErr, strings.TrimSpace(string(outputBytes)))
	}
	info, err := os.Stat(out)
	if err != nil || info.Size() < 256 {
		return fmt.Errorf("native object worker did not produce a valid GLB")
	}
	header := make([]byte, 4)
	f, openErr := os.Open(out)
	if openErr != nil {
		return openErr
	}
	_, readErr := io.ReadFull(f, header)
	f.Close()
	if readErr != nil || !bytes.Equal(header, []byte("glTF")) {
		return fmt.Errorf("native object worker produced an invalid GLB header")
	}
	sha, err := digest(out)
	if err != nil {
		return err
	}
	a.mu.Lock()
	a.jobs[id].ResultSHA = sha
	a.mu.Unlock()
	return nil
}
