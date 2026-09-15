package main

import (
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
)

func TestBodyInferenceModes(t *testing.T) {
	for _, mode := range []string{"", "standard", "no-correctives", "fast512", "fast448", "fast384"} {
		c := config{bodyMode: mode, precision: "bf16"}
		if e := c.validateBodyMode(); e != nil {
			t.Fatal(e)
		}
		base := []string{"--worker", "module"}
		args := c.bodyInferenceArgs(c.precisionArgs(base))
		if !reflect.DeepEqual(args[:3], []string{"--worker", "module", "--bf16"}) {
			t.Fatal(args)
		}
		if strings.HasPrefix(mode, "fast") && args[len(args)-1] != "--body-crop-size="+mode[4:] {
			t.Fatal(args)
		}
	}
	if (config{bodyMode: "fas448"}).validateBodyMode() == nil {
		t.Fatal("invalid mode accepted")
	}
	a := &app{cfg: config{bodyMode: "fast448"}}
	w := httptest.NewRecorder()
	a.routes().ServeHTTP(w, httptest.NewRequest("GET", "/api/config", nil))
	if !strings.Contains(w.Body.String(), "fast448 (approximate)") {
		t.Fatal(w.Body.String())
	}
}
