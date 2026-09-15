package main

import (
	"encoding/binary"
	"fmt"
	"image"
	"image/color"
	"math"
)

func patchPackedBox(packed []byte, box [4]float32) error {
	if len(packed) < 52 || string(packed[:8]) != "S3DIMG01" {
		return fmt.Errorf("invalid packed image")
	}
	for i, v := range box {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return fmt.Errorf("invalid tracked box")
		}
		binary.LittleEndian.PutUint32(packed[20+4*i:], math.Float32bits(v))
	}
	return nil
}

// GEM-X derives a 192:256 person crop, enlarges it by 1.2, then passes the
// resulting square box to SAM 3D Body. Body applies its own published 1.25
// preprocessing padding. Keep the detector box for tracking and display, but
// use this box for the Body pose token consumed by GEM-X.
func gemxBodyBox(box [4]float32) [4]float32 {
	w, h := box[2]-box[0], box[3]-box[1]
	size := max(h, w/.75) * 1.2
	cx, cy := (box[0]+box[2])*.5, (box[1]+box[3])*.5
	return [4]float32{cx - size*.5, cy - size*.5, cx + size*.5, cy + size*.5}
}

// Same native wire bytes and black-alpha compositing as Image.At().RGBA(),
// without per-pixel interface boxing or bytes.Buffer method calls for common
// JPEG/PNG layouts. Image bounds, subimage strides and chroma offsets matter.
func packImage(im image.Image, s settings) []byte {
	bounds := im.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	out := make([]byte, 52+w*h*3)
	copy(out, "S3DIMG01")
	binary.LittleEndian.PutUint32(out[8:], uint32(w))
	binary.LittleEndian.PutUint32(out[12:], uint32(h))
	binary.LittleEndian.PutUint32(out[16:], uint32(w*3))
	for i, v := range s.Box {
		binary.LittleEndian.PutUint32(out[20+4*i:], math.Float32bits(v))
	}
	for i, v := range s.Camera {
		binary.LittleEndian.PutUint32(out[36+4*i:], math.Float32bits(v))
	}
	dest := out[52:]
	i := 0
	switch p := im.(type) {
	case *image.YCbCr:
		for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
			yi := p.YOffset(bounds.Min.X, y)
			for x := bounds.Min.X; x < bounds.Max.X; x++ {
				ci := p.COffset(x, y)
				r, g, b := color.YCbCrToRGB(p.Y[yi], p.Cb[ci], p.Cr[ci])
				yi++
				dest[i], dest[i+1], dest[i+2] = r, g, b
				i += 3
			}
		}
	case *image.RGBA:
		for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
			row := p.Pix[p.PixOffset(bounds.Min.X, y):][:w*4]
			for x := 0; x < len(row); x += 4 {
				dest[i], dest[i+1], dest[i+2] = row[x], row[x+1], row[x+2]
				i += 3
			}
		}
	case *image.NRGBA:
		for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
			row := p.Pix[p.PixOffset(bounds.Min.X, y):][:w*4]
			for x := 0; x < len(row); x += 4 {
				r, g, b, _ := (color.NRGBA{R: row[x], G: row[x+1], B: row[x+2], A: row[x+3]}).RGBA()
				dest[i], dest[i+1], dest[i+2] = byte(r>>8), byte(g>>8), byte(b>>8)
				i += 3
			}
		}
	case *image.Gray:
		for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
			row := p.Pix[p.PixOffset(bounds.Min.X, y):][:w]
			for _, v := range row {
				dest[i], dest[i+1], dest[i+2] = v, v, v
				i += 3
			}
		}
	default:
		for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
			for x := bounds.Min.X; x < bounds.Max.X; x++ {
				r, g, b, _ := im.At(x, y).RGBA()
				dest[i], dest[i+1], dest[i+2] = byte(r>>8), byte(g>>8), byte(b>>8)
				i += 3
			}
		}
	}
	return out
}
