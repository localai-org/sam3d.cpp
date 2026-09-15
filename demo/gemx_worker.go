package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"syscall"
)

const gemxJoints = 77

type gemxWorker struct {
	cmd    *exec.Cmd
	input  io.WriteCloser
	events chan string
	done   chan error
}

type gemxPose struct {
	Positions, Rotations, LocalTranslations, Keypoints []float32
	Parents                                            []int32
	Camera                                             []float32
}

func (c config) gemxEnabled() bool {
	return c.gemxRunner != "" && c.gemxDenoiser != "" && c.gemxVitpose != "" && c.gemxYolox != ""
}

func (a *app) startGemxWorker(ctx context.Context) error {
	c := a.cfg
	args := []string{"--worker", c.gemxDenoiser, c.gemxVitpose, c.gemxYolox, c.module, c.backend,
		strconv.Itoa(c.device), c.description, strconv.Itoa(c.threads), "120"}
	cmd := exec.Command(c.gemxRunner, args...)
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmd.Env = c.nativeEnvironment(os.Environ())
	input, e := cmd.StdinPipe()
	if e != nil {
		return e
	}
	stdout, e := cmd.StdoutPipe()
	if e != nil {
		input.Close()
		return e
	}
	stderr, e := cmd.StderrPipe()
	if e != nil {
		input.Close()
		stdout.Close()
		return e
	}
	if e = cmd.Start(); e != nil {
		input.Close()
		stdout.Close()
		stderr.Close()
		return fmt.Errorf("start GEM-X worker: %w", e)
	}
	w := &gemxWorker{cmd: cmd, input: input, events: make(chan string, 8), done: make(chan error, 1)}
	a.gemx = w
	var readers sync.WaitGroup
	readers.Add(2)
	go func() {
		defer readers.Done()
		s := bufio.NewScanner(stdout)
		for s.Scan() {
			w.events <- s.Text()
		}
	}()
	go func() {
		defer readers.Done()
		s := bufio.NewScanner(stderr)
		for s.Scan() {
			// Keep native diagnostics visible without allowing a full pipe to
			// block cancellation.
			fmt.Fprintln(os.Stderr, "GEM-X:", s.Text())
		}
	}()
	go func() {
		readers.Wait()
		w.done <- cmd.Wait()
	}()
	select {
	case line := <-w.events:
		if line != "READY" {
			a.stopGemxWorker()
			return fmt.Errorf("unexpected GEM-X worker reply %q", line)
		}
		return nil
	case e := <-w.done:
		a.gemx = nil
		return fmt.Errorf("GEM-X worker exited during startup: %v", e)
	case <-ctx.Done():
		a.stopGemxWorker()
		return ctx.Err()
	}
}

func (a *app) stopGemxWorker() {
	w := a.gemx
	if w == nil {
		return
	}
	a.gemx = nil
	w.input.Close()
	select {
	case <-w.done:
		return
	default:
	}
	syscall.Kill(-w.cmd.Process.Pid, syscall.SIGKILL)
	<-w.done
}

func gemxRequest(fields ...string) ([]byte, error) {
	var out bytes.Buffer
	for _, value := range fields {
		if len(value) < 1 || len(value) > 4096 || strings.ContainsRune(value, 0) {
			return nil, fmt.Errorf("invalid GEM-X worker field")
		}
		binary.Write(&out, binary.LittleEndian, uint32(len(value)))
		out.WriteString(value)
	}
	return out.Bytes(), nil
}

type gemxPending struct {
	worker *gemxWorker
}

func (a *app) beginGemxFrame(ctx context.Context, stream, dir string) ([4]float32, *gemxPending, error) {
	var box [4]float32
	if a.gemx == nil {
		if e := a.startGemxWorker(ctx); e != nil {
			return box, nil, e
		}
	}
	w := a.gemx
	frame, e := gemxRequest(stream, filepath.Join(dir, "image.input"),
		filepath.Join(dir, "result.bin"), filepath.Join(dir, "gemx.pose"))
	if e != nil {
		return box, nil, e
	}
	sent := make(chan error, 1)
	go func() { _, e := w.input.Write(frame); sent <- e }()
	select {
	case e = <-sent:
		if e != nil {
			a.stopGemxWorker()
			return box, nil, e
		}
	case <-ctx.Done():
		a.stopGemxWorker()
		return box, nil, ctx.Err()
	}
	select {
	case line := <-w.events:
		var x0, y0, x1, y1 float64
		if n, scanErr := fmt.Sscanf(line, "BOX %f %f %f %f", &x0, &y0, &x1, &y1); scanErr != nil || n != 4 ||
			math.IsNaN(x0) || math.IsNaN(y0) || math.IsNaN(x1) || math.IsNaN(y1) ||
			math.IsInf(x0, 0) || math.IsInf(y0, 0) || math.IsInf(x1, 0) || math.IsInf(y1, 0) ||
			x1-x0 < 8 || y1-y0 < 8 {
			a.stopGemxWorker()
			return box, nil, fmt.Errorf("unexpected GEM-X worker reply %q", line)
		}
		box = [4]float32{float32(x0), float32(y0), float32(x1), float32(y1)}
		return box, &gemxPending{worker: w}, nil
	case e = <-w.done:
		a.gemx = nil
		return box, nil, fmt.Errorf("GEM-X worker exited: %v", e)
	case <-ctx.Done():
		a.stopGemxWorker()
		return box, nil, ctx.Err()
	}
}

func (a *app) finishGemxFrame(ctx context.Context, pending *gemxPending) error {
	if pending == nil || pending.worker == nil || a.gemx != pending.worker {
		return fmt.Errorf("GEM-X worker changed while processing a frame")
	}
	w := pending.worker
	select {
	case line := <-w.events:
		if line != "DONE" {
			a.stopGemxWorker()
			return fmt.Errorf("unexpected GEM-X worker reply %q", line)
		}
		return nil
	case e := <-w.done:
		a.gemx = nil
		return fmt.Errorf("GEM-X worker exited: %v", e)
	case <-ctx.Done():
		a.stopGemxWorker()
		return ctx.Err()
	}
}

func parseGemxPose(r io.Reader) (*gemxPose, error) {
	var magic [8]byte
	if _, e := io.ReadFull(r, magic[:]); e != nil || string(magic[:]) != "GEMPOSE1" {
		return nil, fmt.Errorf("invalid GEM-X pose magic")
	}
	readFloats := func(n int) ([]float32, error) {
		v := make([]float32, n)
		if e := binary.Read(r, binary.LittleEndian, v); e != nil {
			return nil, e
		}
		for _, x := range v {
			if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) || math.Abs(float64(x)) > 10000 {
				return nil, fmt.Errorf("invalid GEM-X pose value")
			}
		}
		return v, nil
	}
	p := &gemxPose{}
	var e error
	if p.Positions, e = readFloats(gemxJoints * 3); e != nil {
		return nil, e
	}
	if p.Rotations, e = readFloats(gemxJoints * 4); e != nil {
		return nil, e
	}
	if p.LocalTranslations, e = readFloats(gemxJoints * 3); e != nil {
		return nil, e
	}
	p.Parents = make([]int32, gemxJoints)
	if e = binary.Read(r, binary.LittleEndian, p.Parents); e != nil {
		return nil, e
	}
	if p.Parents[0] != -1 {
		return nil, fmt.Errorf("invalid GEM-X root")
	}
	for i := 1; i < gemxJoints; i++ {
		if p.Parents[i] < 0 || p.Parents[i] >= int32(i) {
			return nil, fmt.Errorf("invalid GEM-X topology")
		}
	}
	if p.Camera, e = readFloats(3); e != nil {
		return nil, e
	}
	if p.Keypoints, e = readFloats(gemxJoints * 3); e != nil {
		return nil, e
	}
	var tail [1]byte
	if n, e := r.Read(tail[:]); n != 0 || e != io.EOF {
		return nil, fmt.Errorf("trailing GEM-X pose data")
	}
	return p, nil
}

func compactGemxFrame(p *gemxPose, stamp float64) []byte {
	var out bytes.Buffer
	out.WriteString("GEMTRK01")
	binary.Write(&out, binary.LittleEndian, stamp)
	binary.Write(&out, binary.LittleEndian, p.Positions)
	binary.Write(&out, binary.LittleEndian, p.Parents)
	binary.Write(&out, binary.LittleEndian, p.Camera)
	return out.Bytes()
}
