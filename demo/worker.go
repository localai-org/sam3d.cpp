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
	"time"
)

// Only the queue consumer owns process state. The pipe reader owns no job;
// bounded log/stage events are consumed by whichever request is active.
type workerEvent struct {
	line     string
	protocol bool
}
type residentWorker struct {
	cmd      *exec.Cmd
	input    io.WriteCloser
	events   chan workerEvent
	done     chan struct{}
	err      error // read only after done is closed
	unit     string
	lastUsed time.Time
}

func (a *app) startResident() error {
	if e := checkMemory(a.cfg.memory, a.cfg.reserve); e != nil {
		return e
	}
	c := a.cfg
	args := c.precisionArgs([]string{"--worker", c.module, c.backend, strconv.Itoa(c.device), c.description, c.backbone, c.branch, c.mhr, strconv.Itoa(c.threads)})
	args = c.bodyInferenceArgs(args)
	executable := c.runner
	w := &residentWorker{events: make(chan workerEvent, 64), done: make(chan struct{}), lastUsed: time.Now()}
	if c.memory > 0 {
		w.unit = fmt.Sprintf("sam3d-demo-worker-%d-%d.scope", os.Getpid(), time.Now().UnixNano())
		args = append([]string{"--user", "--scope", "--quiet", "--collect", "--unit=" + w.unit, "-p", fmt.Sprintf("MemoryMax=%dM", c.memory), "-p", fmt.Sprintf("MemoryHigh=%dM", c.memory*5/6), "-p", "MemorySwapMax=0", "-p", "OOMPolicy=kill", executable}, args...)
		executable = "systemd-run"
	}
	cmd := exec.Command(executable, args...)
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmd.Env = c.nativeEnvironment(os.Environ())
	in, e := cmd.StdinPipe()
	if e != nil {
		return e
	}
	stdout, e := cmd.StdoutPipe()
	if e != nil {
		in.Close()
		return e
	}
	stderr, e := cmd.StderrPipe()
	if e != nil {
		in.Close()
		stdout.Close()
		return e
	}
	if e = cmd.Start(); e != nil {
		in.Close()
		stdout.Close()
		stderr.Close()
		return fmt.Errorf("start resident worker: %w", e)
	}
	w.cmd = cmd
	w.input = in
	a.native = w
	var readers sync.WaitGroup
	var readErrors [2]error
	for index, pipe := range []io.Reader{stderr, stdout} {
		readers.Add(1)
		go func(index int, pipe io.Reader) {
			defer readers.Done()
			scanner := bufio.NewScanner(pipe)
			scanner.Buffer(make([]byte, 4096), 65536)
			for scanner.Scan() {
				event := workerEvent{scanner.Text(), index == 1}
				// Protocol messages must never be lost. stderr is diagnostic:
				// dropping excess lines prevents a verbose driver deadlocking
				// cancellation or an idle resident session.
				if event.protocol {
					w.events <- event
				} else {
					select {
					case w.events <- event:
					default:
					}
				}
			}
			readErrors[index] = scanner.Err()
		}(index, pipe)
	}
	go func() {
		readers.Wait()
		w.err = cmd.Wait()
		if w.err == nil {
			for _, e := range readErrors {
				if e != nil {
					w.err = e
				}
			}
		}
		close(w.done)
	}()
	return nil
}

func (a *app) stopResident() {
	w := a.native
	if w == nil {
		return
	}
	a.native = nil
	w.input.Close()
	select {
	case <-w.done:
		return
	default:
	}
	if w.unit != "" {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		exec.CommandContext(ctx, "systemctl", "--user", "stop", w.unit).Run()
		cancel()
	}
	// Reap on every cancellation/failure/shutdown; never leave a hidden GPU job.
	syscall.Kill(-w.cmd.Process.Pid, syscall.SIGKILL)
	// Drain protocol events while waiting, including a racing DONE notification.
	for {
		select {
		case <-w.done:
			return
		case <-w.events:
		}
	}
}

func workerFrame(input, output string) ([]byte, error) {
	var b bytes.Buffer
	for _, path := range []string{input, output} {
		if len(path) < 1 || len(path) > 4096 || strings.ContainsRune(path, 0) {
			return nil, fmt.Errorf("invalid worker path")
		}
		binary.Write(&b, binary.LittleEndian, uint32(len(path)))
		b.WriteString(path)
	}
	return b.Bytes(), nil
}

func (a *app) runResident(ctx context.Context, id string) (err error) {
	return a.runResidentPaths(ctx, a.dir(id), func(s string) { a.stage(id, s) })
}

// Shared by still jobs and video frames; called only by the one worker loop.
func (a *app) runResidentPaths(ctx context.Context, dir string, stage func(string), timing ...func(string)) (err error) {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}
	if a.native != nil {
		select {
		case <-a.native.done:
			a.stopResident()
		default:
		}
	}
	fresh := a.native == nil
	if fresh {
		if err = a.startResident(); err != nil {
			return err
		}
	}
	defer func() {
		if err != nil {
			a.stopResident()
		}
	}()
	w := a.native
	file, e := os.Create(filepath.Join(dir, "inference.log"))
	if e != nil {
		return e
	}
	defer file.Close()
	var tail string
	written := 0
	record := func(line string) {
		tail += line + "\n"
		if len(tail) > 4096 {
			tail = tail[len(tail)-4096:]
		}
		if written < 1<<20 {
			n, _ := fmt.Fprintln(file, line)
			written += n
		}
		if strings.HasPrefix(line, "STAGE ") {
			stage(strings.TrimPrefix(line, "STAGE "))
		}
		if strings.HasPrefix(line, "TIMING ") && len(timing) > 0 {
			timing[0](line)
		}
	}
	wait := func(expected string) error {
		for {
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-w.done:
				// Read queued diagnostics before reporting an abnormal process exit.
				for {
					select {
					case ev := <-w.events:
						if !ev.protocol {
							record(ev.line)
						}
					default:
						return fmt.Errorf("resident worker exited: %v\n%s", w.err, strings.TrimSpace(tail))
					}
				}
			case ev := <-w.events:
				if !ev.protocol {
					record(ev.line)
					continue
				}
				if expected == "DONE" && strings.HasPrefix(ev.line, "TIMING body_infer_ms ") {
					value, e := strconv.ParseFloat(strings.TrimPrefix(ev.line, "TIMING body_infer_ms "), 64)
					if e != nil || value < 0 || math.IsNaN(value) || math.IsInf(value, 0) {
						return fmt.Errorf("invalid native timing reply")
					}
					record(ev.line)
					continue
				}
				if ev.line != expected {
					return fmt.Errorf("unexpected native worker reply %q", ev.line)
				}
				return nil
			}
		}
	}
	if fresh {
		stage("Starting resident native worker")
		if err = wait("READY"); err != nil {
			return err
		}
	}
	stage("Estimating body with resident model")
	frame, e := workerFrame(filepath.Join(dir, "image.input"), filepath.Join(dir, "result.bin"))
	if e != nil {
		return e
	}
	// Separate writer so even an unresponsive worker cannot block cancellation.
	sent := make(chan error, 1)
	go func() { _, e := w.input.Write(frame); sent <- e }()
	select {
	case err = <-sent:
		if err != nil {
			return err
		}
	case <-ctx.Done():
		return ctx.Err()
	}
	if err = wait("DONE"); err != nil {
		return err
	}
	w.lastUsed = time.Now()
	return nil
}
