package main

import (
	"bytes"
	"encoding/binary"
	"image"
	"image/color"
	"image/jpeg"
	"math"
	"math/rand"
	"testing"
)

// Frozen original implementation, including the wire header and 16-bit RGBA
// truncation. Keep independent of the optimized implementation.
func originalPackImage(im image.Image, s settings) []byte {
	w, h := im.Bounds().Dx(), im.Bounds().Dy()
	var out bytes.Buffer
	out.Grow(52 + w*h*3)
	out.WriteString("S3DIMG01")
	binary.Write(&out, binary.LittleEndian, []uint32{uint32(w), uint32(h), uint32(w * 3)})
	binary.Write(&out, binary.LittleEndian, s.Box)
	binary.Write(&out, binary.LittleEndian, s.Camera)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r, g, b, _ := im.At(im.Bounds().Min.X+x, im.Bounds().Min.Y+y).RGBA()
			out.WriteByte(byte(r >> 8))
			out.WriteByte(byte(g >> 8))
			out.WriteByte(byte(b >> 8))
		}
	}
	return out.Bytes()
}

func TestPackImageExact(t *testing.T) {
	random := rand.New(rand.NewSource(12))
	s := settings{Box: [4]float32{math.Float32frombits(0x80000000), .25, 17, 19}, Camera: [4]float32{987.25, 999, 5.5, 7.5}}
	check := func(im image.Image) {
		t.Helper()
		if !bytes.Equal(packImage(im, s), originalPackImage(im, s)) {
			t.Fatalf("packing differs for %T bounds %v", im, im.Bounds())
		}
	}
	for _, origin := range []image.Point{{0, 0}, {3, 5}, {-3, -5}} {
		bounds := image.Rectangle{Min: origin, Max: origin.Add(image.Pt(19, 23))}
		crop := bounds.Inset(1)
		for _, ratio := range []image.YCbCrSubsampleRatio{image.YCbCrSubsampleRatio444, image.YCbCrSubsampleRatio422, image.YCbCrSubsampleRatio420, image.YCbCrSubsampleRatio440, image.YCbCrSubsampleRatio411, image.YCbCrSubsampleRatio410} {
			im := image.NewYCbCr(bounds, ratio)
			random.Read(im.Y)
			random.Read(im.Cb)
			random.Read(im.Cr)
			check(im)
			check(im.SubImage(crop))
		}
		rgba := image.NewRGBA(bounds)
		random.Read(rgba.Pix)
		check(rgba)
		check(rgba.SubImage(crop))
		nrgba := image.NewNRGBA(bounds)
		random.Read(nrgba.Pix)
		check(nrgba)
		check(nrgba.SubImage(crop))
		gray := image.NewGray(bounds)
		random.Read(gray.Pix)
		check(gray)
		check(gray.SubImage(crop))
		rgba64 := image.NewRGBA64(bounds)
		random.Read(rgba64.Pix)
		check(rgba64)
		check(rgba64.SubImage(crop))
		palette := image.NewPaletted(bounds, color.Palette{color.Transparent, color.NRGBA{200, 100, 50, 127}, color.White})
		for i := range palette.Pix {
			palette.Pix[i] = byte(i % 3)
		}
		check(palette)
	}
	// Every alpha/channel pair, especially 0, 1, 254 and 255.
	alpha := image.NewNRGBA(image.Rect(0, 0, 256, 256))
	for a := 0; a < 256; a++ {
		for c := 0; c < 256; c++ {
			alpha.SetNRGBA(c, a, color.NRGBA{uint8(c), uint8(255 - c), uint8(c), uint8(a)})
		}
	}
	check(alpha)
	var encoded bytes.Buffer
	if err := jpeg.Encode(&encoded, alpha, nil); err != nil {
		t.Fatal(err)
	}
	decoded, err := jpeg.Decode(&encoded)
	if err != nil {
		t.Fatal(err)
	}
	check(decoded)
}

func TestYCbCrEightBitEquivalence(t *testing.T) {
	// Exhaust all 24 input bits. The old path truncates 16-bit RGBA; replacing it
	// with a superficially similar conversion must not change rounding/clipping.
	for y := 0; y < 256; y++ {
		for cb := 0; cb < 256; cb++ {
			for cr := 0; cr < 256; cr++ {
				c := color.YCbCr{uint8(y), uint8(cb), uint8(cr)}
				r, g, b, _ := c.RGBA()
				rr, gg, bb := color.YCbCrToRGB(c.Y, c.Cb, c.Cr)
				if rr != byte(r>>8) || gg != byte(g>>8) || bb != byte(b>>8) {
					t.Fatal("YCbCr rounding changed", c)
				}
			}
		}
	}
}

var packBenchmark []byte

func BenchmarkPackImage(b *testing.B) {
	im := image.NewYCbCr(image.Rect(0, 0, 960, 540), image.YCbCrSubsampleRatio420)
	random := rand.New(rand.NewSource(12))
	random.Read(im.Y)
	random.Read(im.Cb)
	random.Read(im.Cr)
	for _, test := range []struct {
		name string
		pack func(image.Image, settings) []byte
	}{{"original", originalPackImage}, {"typed", packImage}} {
		b.Run(test.name, func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(960 * 540 * 3)
			for i := 0; i < b.N; i++ {
				packBenchmark = test.pack(im, settings{})
			}
		})
	}
}
