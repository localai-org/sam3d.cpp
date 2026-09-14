package main

import (
	"bytes"
	"encoding/binary"
	"testing"
)

func TestGemxPoseAndTrackFrame(t *testing.T) {
	var source bytes.Buffer
	source.WriteString("GEMPOSE1")
	positions := make([]float32, gemxJoints*3)
	rotations := make([]float32, gemxJoints*4)
	translations := make([]float32, gemxJoints*3)
	parents := make([]int32, gemxJoints)
	keypoints := make([]float32, gemxJoints*3)
	parents[0] = -1
	for i := 1; i < gemxJoints; i++ {
		parents[i] = int32(i - 1)
		rotations[i*4+3] = 1
	}
	rotations[3] = 1
	camera := []float32{0, 0, 2}
	for _, values := range []any{positions, rotations, translations, parents, camera, keypoints} {
		if e := binary.Write(&source, binary.LittleEndian, values); e != nil {
			t.Fatal(e)
		}
	}
	pose, e := parseGemxPose(bytes.NewReader(source.Bytes()))
	if e != nil {
		t.Fatal(e)
	}
	frame := compactGemxFrame(pose, .25)
	const expected = 16 + gemxJoints*3*4 + gemxJoints*4 + 3*4
	if len(frame) != expected || string(frame[:8]) != "GEMTRK01" {
		t.Fatalf("unexpected GEM-X track frame: %d bytes", len(frame))
	}
}
