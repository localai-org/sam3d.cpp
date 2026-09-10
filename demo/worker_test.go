package main

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// A real child process exercises pipes, cancellation, reuse and restart. It
// deliberately does not load weights; numerical acceptance is a separate gate.
func TestMain(m *testing.M) {
	if len(os.Args) > 1 && os.Args[1] == "--worker" && os.Getenv("SAM3D_FAKE_WORKER") == "1" {
		fmt.Println("READY")
		for {
			paths := make([]string, 2)
			for i := range paths {
				var n uint32
				if e := binary.Read(os.Stdin, binary.LittleEndian, &n); e != nil {
					os.Exit(0)
				}
				if n < 1 || n > 4096 {
					os.Exit(2)
				}
				b := make([]byte, n)
				if _, e := io.ReadFull(os.Stdin, b); e != nil {
					os.Exit(2)
				}
				paths[i] = string(b)
			}
			contents, e := os.ReadFile(paths[0])
			if e != nil {
				os.Exit(3)
			}
			if string(contents) == "hang" {
				for {
					time.Sleep(time.Second)
				}
			}
			if string(contents) == "fail" {
				fmt.Fprintln(os.Stderr, "intentional worker failure")
				os.Exit(7)
			}
			if string(contents) == "bad-protocol" {
				fmt.Println("INVALID")
				continue
			}
			fmt.Fprintln(os.Stderr, "STAGE Fake inference")
			if strings.HasPrefix(string(contents), "S3DIMG01") {
				if os.Getenv("SAM3D_FAKE_IMAGE_HANG") == "1" {
					for {
						time.Sleep(time.Second)
					}
				}
				contents = syntheticResult()
			}
			if e := os.WriteFile(paths[1], contents, 0600); e != nil {
				os.Exit(4)
			}
			if strings.HasPrefix(string(contents), "timing:") {
				fmt.Println("TIMING body_infer_ms " + strings.TrimPrefix(string(contents), "timing:"))
			} else {
				fmt.Println("TIMING body_infer_ms 12.500000")
			}
			fmt.Println("DONE")
		}
	}
	os.Exit(m.Run())
}

func TestResidentTimingProtocol(t *testing.T) {
	t.Setenv("SAM3D_FAKE_WORKER", "1")
	exe, e := os.Executable()
	if e != nil {
		t.Fatal(e)
	}
	dir := t.TempDir()
	a := &app{cfg: config{runner: exe, backend: "CPU", threads: 1}}
	defer a.stopResident()
	for _, value := range []string{"1.25", "2.75", "NaN", "Inf", "-1", "bad", "3.5"} {
		if e := os.WriteFile(filepath.Join(dir, "image.input"), []byte("timing:"+value), 0600); e != nil {
			t.Fatal(e)
		}
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		var timings []string
		e := a.runResidentPaths(ctx, dir, func(string) {}, func(line string) { timings = append(timings, line) })
		cancel()
		valid := value == "1.25" || value == "2.75" || value == "3.5"
		if valid {
			if e != nil || len(timings) != 1 || timings[0] != "TIMING body_infer_ms "+value {
				t.Fatalf("timing %q: %v %v", value, timings, e)
			}
		} else if e == nil || a.native != nil {
			t.Fatalf("invalid timing retained worker: %q %v", value, e)
		}
	}
}
func TestWorkerFrame(t *testing.T) {
	for _, path := range []string{"", strings.Repeat("x", 4097), "x\x00y"} {
		if _, e := workerFrame(path, "out"); e == nil {
			t.Fatal("accepted bad path")
		}
	}
	frame, e := workerFrame("a\nb", "c d")
	if e != nil || len(frame) != 14 {
		t.Fatalf("framing: %x %v", frame, e)
	}
}
func TestResidentReuseCancelAndRecovery(t *testing.T) {
	t.Setenv("SAM3D_FAKE_WORKER", "1")
	exe, e := os.Executable()
	if e != nil {
		t.Fatal(e)
	}
	a := &app{cfg: config{runner: exe, data: t.TempDir(), backend: "CPU", threads: 1}, jobs: map[string]*job{}}
	defer a.stopResident()
	id := "012345678901234567890123"
	if e = os.Mkdir(a.dir(id), 0700); e != nil {
		t.Fatal(e)
	}
	a.jobs[id] = &job{ID: id}
	run := func(value string, timeout time.Duration) error {
		if e := os.WriteFile(filepath.Join(a.dir(id), "image.input"), []byte(value), 0600); e != nil {
			t.Fatal(e)
		}
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		defer cancel()
		return a.runResident(ctx, id)
	}
	for _, value := range []string{"one", "two", "one"} {
		previous := a.native
		if e = run(value, 5*time.Second); e != nil {
			t.Fatal(e)
		}
		if previous != nil && previous != a.native {
			t.Fatal("worker not reused")
		}
		contents, _ := os.ReadFile(filepath.Join(a.dir(id), "result.bin"))
		if string(contents) != value {
			t.Fatal("stale output")
		}
	}
	old := a.native
	if e = run("hang", 100*time.Millisecond); !errors.Is(e, context.DeadlineExceeded) {
		t.Fatalf("timeout: %v", e)
	}
	if a.native != nil {
		t.Fatal("cancelled worker retained")
	}
	select {
	case <-old.done:
	default:
		t.Fatal("cancelled process not reaped")
	}
	for _, mode := range []string{"fail", "bad-protocol"} {
		if e = run(mode, 5*time.Second); e == nil {
			t.Fatal("accepted failed worker")
		}
		if a.native != nil {
			t.Fatal("failed worker retained")
		}
		if e = run("recovered", 5*time.Second); e != nil {
			t.Fatal(e)
		}
	}
	a.stopResident()
	if a.native != nil {
		t.Fatal("shutdown retained worker")
	}
}

func TestResidentIdleRelease(t *testing.T) {
	t.Setenv("SAM3D_FAKE_WORKER", "1")
	exe, e := os.Executable()
	if e != nil {
		t.Fatal(e)
	}
	a := &app{cfg: config{runner: exe, backend: "CPU", threads: 1, workerIdle: 20 * time.Millisecond}, queue: make(chan string)}
	if e = a.startResident(); e != nil {
		t.Fatal(e)
	}
	w := a.native
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { a.worker(ctx); close(done) }()
	select {
	case <-w.done:
	case <-time.After(5 * time.Second):
		cancel()
		<-done
		t.Fatal("idle worker was not released")
	}
	cancel()
	<-done
	if a.native != nil {
		t.Fatal("idle model retained")
	}
}
