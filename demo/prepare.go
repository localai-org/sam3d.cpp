package main

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"image"
	"io"
	"net/http"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"syscall"
	"time"
)

const maxOriginalUpload = 128 << 20

// Never retain unbounded output, including stderr from a malformed input.
type cappedBuffer struct {
	bytes.Buffer
	limit    int
	exceeded bool
}

func (b *cappedBuffer) Write(p []byte) (int, error) {
	n := len(p)
	left := b.limit - b.Len()
	if left < n {
		b.exceeded = true
		p = p[:left]
	}
	b.Buffer.Write(p)
	return n, nil
}

func conversionArgs(edge int) []string {
	// Input is an inherited regular-file descriptor (seekable, unlike pipe:).
	// Neither arbitrary file paths nor URLs are accessible via input protocols.
	// No shell, user filenames, playlists, scripts, devices or external filters.
	return []string{"-hide_banner", "-loglevel", "error", "-nostdin", "-max_alloc", "268435456",
		"-threads", "2", "-filter_threads", "1", "-protocol_whitelist", "fd,pipe",
		"-format_whitelist", "jpeg_pipe,png_pipe,webp_pipe,bmp_pipe,tiff_pipe,gif,gif_pipe,ico,mov,jpegxl_pipe,exr_pipe,heif,avif",
		"-fd", "0", "-i", "fd:", "-map", "0:v:0", "-frames:v", "1", "-an", "-sn", "-dn",
		"-vf", fmt.Sprintf("scale=w='min(%d,iw)':h='min(%d,ih)':force_original_aspect_ratio=decrease,format=rgba,premultiply=inplace=1,format=rgb24", edge, edge),
		"-threads", "2", "-c:v", "png", "-f", "image2pipe", "pipe:1"}
}
func (a *app) convertPhoto(ctx context.Context, input *os.File, edge int) ([]byte, error) {
	exe := a.cfg.ffmpeg
	if exe == "" {
		exe = "ffmpeg"
	}
	path, e := exec.LookPath(exe)
	if e != nil {
		return nil, fmt.Errorf("FFmpeg is unavailable: install it or configure --ffmpeg (%w)", e)
	}
	if _, e = input.Seek(0, io.SeekStart); e != nil {
		return nil, e
	}
	args := conversionArgs(edge)
	var nonce [12]byte
	if _, e = rand.Read(nonce[:]); e != nil {
		return nil, e
	}
	unit := "sam3d-photo-" + hex.EncodeToString(nonce[:])
	if a.cfg.memory > 0 {
		args = append([]string{"--user", "--scope", "--quiet", "--collect", "--unit=" + unit, "-p", "MemoryMax=768M", "-p", "MemorySwapMax=0", "-p", "OOMPolicy=kill", path}, args...)
		path = "systemd-run"
	}
	cmd := exec.Command(path, args...)
	cmd.Stdin = input
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	output := &cappedBuffer{limit: maxUpload}
	diagnostics := &cappedBuffer{limit: 8192}
	cmd.Stdout = output
	cmd.Stderr = diagnostics
	if e = cmd.Start(); e != nil {
		return nil, fmt.Errorf("start FFmpeg: %w", e)
	}
	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()
	select {
	case e = <-done:
	case <-ctx.Done():
		if a.cfg.memory > 0 {
			stopCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			exec.CommandContext(stopCtx, "systemctl", "--user", "stop", unit+".scope").Run()
			cancel()
		}
		syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
		<-done
		return nil, fmt.Errorf("FFmpeg stopped: %w", ctx.Err())
	}
	if e != nil {
		return nil, fmt.Errorf("FFmpeg conversion failed: %w: %s", e, strings.TrimSpace(diagnostics.String()))
	}
	if output.exceeded {
		return nil, errConvertedSize
	}
	if _, e = decodeImage(output.Bytes()); e != nil {
		return nil, fmt.Errorf("FFmpeg produced an unusable image: %w", e)
	}
	return output.Bytes(), nil
}

var errConvertedSize = fmt.Errorf("FFmpeg PNG exceeds 20 MiB")

// Photo preparation happens BEFORE person selection and intrinsics entry. This
// avoids silently changing the coordinate system of an already-selected box.
func (a *app) preparePhoto(w http.ResponseWriter, r *http.Request) {
	select {
	case a.uploads <- struct{}{}:
		defer func() { <-a.uploads }()
	default:
		fail(w, 429, "another photo is being prepared; please retry shortly")
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, maxOriginalUpload+65536)
	reader, e := r.MultipartReader()
	if e != nil {
		fail(w, 400, "expected a photo upload")
		return
	}
	// Spool to a private, bounded, disposable file; never use the user's filename.
	f, e := os.CreateTemp(a.cfg.data, ".photo-*")
	if e != nil {
		fail(w, 500, e)
		return
	}
	defer func() { f.Close(); os.Remove(f.Name()) }()
	var size int64
	found := false
	for count := 0; ; count++ {
		part, err := reader.NextPart()
		if err == io.EOF {
			break
		}
		if err != nil || count > 2 {
			fail(w, 400, "invalid photo upload or file exceeds 128 MiB")
			return
		}
		if part.FormName() != "image" || found {
			part.Close()
			fail(w, 400, "upload exactly one image")
			return
		}
		found = true
		size, e = io.Copy(f, io.LimitReader(part, maxOriginalUpload+1))
		part.Close()
		if e != nil || size > maxOriginalUpload {
			fail(w, 413, "original photo exceeds the 128 MiB conversion limit")
			return
		}
	}
	if !found || size == 0 {
		fail(w, 400, "image is required")
		return
	}
	f.Seek(0, io.SeekStart)
	cfg, format, decodeErr := image.DecodeConfig(f)
	f.Seek(0, io.SeekStart)
	acceptable := decodeErr == nil && (format == "jpeg" || format == "png") && size <= maxUpload && cfg.Width >= 8 && cfg.Height >= 8 && cfg.Width <= 32766 && cfg.Height <= 32766 && int64(cfg.Width)*int64(cfg.Height) <= 16000000
	if acceptable && r.URL.Query().Get("force") != "1" {
		mime := "image/png"
		if format == "jpeg" {
			mime = "image/jpeg"
		}
		w.Header().Set("Content-Type", mime)
		w.Header().Set("X-Sam3d-Converted", "false")
		w.Header().Set("Content-Length", strconv.FormatInt(size, 10))
		io.Copy(w, f)
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), 45*time.Second)
	defer cancel()
	output, e := a.convertPhoto(ctx, f, 4000)
	if e == errConvertedSize {
		output, e = a.convertPhoto(ctx, f, 2500)
	} // <=6.25 MP RGB PNG fits the byte budget.
	if e != nil {
		fail(w, 422, fmt.Sprintf("Could not prepare photo with FFmpeg: %v", e))
		return
	}
	result, _, e := image.DecodeConfig(bytes.NewReader(output))
	if e != nil {
		fail(w, 500, e)
		return
	}
	note := fmt.Sprintf("FFmpeg re-encoded this photo as PNG (%d x %d).", result.Width, result.Height)
	if decodeErr == nil && (cfg.Width != result.Width || cfg.Height != result.Height) {
		note = fmt.Sprintf("FFmpeg resized/reoriented this photo from %d x %d to %d x %d and re-encoded it as PNG.", cfg.Width, cfg.Height, result.Width, result.Height)
	}
	note += " Only the first image/frame is used. Select the person on this converted preview."
	w.Header().Set("Content-Type", "image/png")
	w.Header().Set("X-Sam3d-Converted", "true")
	w.Header().Set("X-Sam3d-Notice", note)
	w.Header().Set("Content-Length", strconv.Itoa(len(output)))
	w.Write(output)
}
