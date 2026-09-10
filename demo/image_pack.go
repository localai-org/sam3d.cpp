package main

import (
	"encoding/binary"
	"image"
	"image/color"
	"math"
)

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
