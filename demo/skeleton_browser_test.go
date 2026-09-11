package main

import (
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func TestSkeletonBrowser(t *testing.T) {
	node := os.Getenv("SAM3D_TEST_NODE")
	if node == "" {
		t.Skip("set SAM3D_TEST_NODE and SAM3D_TEST_CHROMIUM for real-browser recording QA")
	}
	chrome := os.Getenv("SAM3D_TEST_CHROMIUM")
	if chrome == "" {
		chrome = "chromium"
	}
	a, _ := trackingWorker(t)
	server := httptest.NewServer(a.routes())
	defer server.Close()
	script, _ := filepath.Abs("../tests/test_skeleton_browser.mjs")
	cmd := exec.Command(node, script, server.URL, chrome)
	out, e := cmd.CombinedOutput()
	if e != nil {
		t.Fatalf("browser: %v\n%s", e, out)
	}
	t.Log(string(out))
}
