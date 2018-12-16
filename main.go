package main

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"

	"github.com/fogleman/gg"
)

const (
	width    = 2048
	height   = 2048
	scale    = 16
	gridGap  = 2
	gridSize = 8
)

type vector mat.Dense

func vec(x, y float64) *vector {
	return (*vector)(mat.NewDense(2, 1, []float64{x, y}))
}

func (v *vector) x() float64 {
	return (*mat.Dense)(v).At(0, 0)
}

func (v *vector) y() float64 {
	return (*mat.Dense)(v).At(1, 0)
}

func (v *vector) add(o *vector) *vector {
	c := new(vector)
	(*mat.Dense)(c).Add((*mat.Dense)(v), (*mat.Dense)(o))
	return c
}

func (v *vector) sub(o *vector) *vector {
	c := new(vector)
	(*mat.Dense)(c).Sub((*mat.Dense)(v), (*mat.Dense)(o))
	return c
}

func (v *vector) mul(s float64) *vector {
	c := new(vector)
	(*mat.Dense)(c).Scale(s, (*mat.Dense)(v))
	return c
}

func (v *vector) length() float64 {
	return math.Sqrt(v.x()*v.x() + v.y()*v.y())
}

func (v *vector) norm() *vector {
	return v.mul(1 / v.length())
}

func (v *vector) comp() (x, y float64) {
	return v.x(), v.y()
}

func (v *vector) dot(o *vector) float64 {
	return v.x()*o.x() + v.y()*o.y()
}

func (v *vector) String() string {
	return fmt.Sprintf("[%v, %v]", v.x(), v.y())
}

type line struct {
	start, direction *vector
	length           float64 // The amount of 'direction's to be added to 'start' over the entire line.
	orientation      int     // If 1, any point above this line will be in the medium. Otherwise, any point above this line will be in a vacuum.
	refractiveIndex  float64
}

func (l *line) draw(dc *gg.Context) {
	end := l.start.add(l.direction.mul(l.length))

	dc.MoveTo(pt(l.start.x(), l.start.y()))
	dc.LineTo(pt(end.x(), end.y()))
	dc.SetRGB(0.0, 0.0, 0.0)
	dc.SetLineWidth(scale / 6)
	dc.Stroke()
}

// cartesian returns an equation in the form ax - by = c.
func (l *line) cartesian() (a, b, c float64) {
	return l.direction.y(),
		l.direction.x(),
		l.direction.y()*l.start.x() - l.direction.x()*l.start.y()
}

type ray struct {
	start, direction *vector
}

type raySegment struct {
	start, direction *vector
	length           float64
}

func (r *raySegment) draw(dc *gg.Context) {
	end := r.start.add(r.direction.mul(r.length))

	dc.MoveTo(pt(r.start.x(), r.start.y()))
	dc.LineTo(pt(end.x(), end.y()))
	dc.SetRGB(1.0, 0.0, 0.0)
	dc.SetLineWidth(scale / 4)
	dc.Stroke()

	x, y := pt(r.start.comp())
	dc.DrawCircle(x, y, 5)
	x, y = pt(end.comp())
	dc.DrawCircle(x, y, 5)
}

func (r *ray) draw(dc *gg.Context, length float64) {
	dx, dy := r.direction.norm().comp()

	dc.MoveTo(pt(r.start.x(), r.start.y()))
	dc.LineTo(pt(r.start.x()+length*dx, r.start.y()+length*dy))
	dc.SetRGB(1.0, 0.0, 0.0)
	dc.SetLineWidth(scale / 4)
	dc.Stroke()
}

func (r *ray) intersect(l *line) (ok bool, x, y, distance float64) {
	lineDist := -(l.start.x()*r.direction.y() - l.start.y()*r.direction.x() - r.direction.y()*r.start.x() + r.direction.x()*r.start.y()) / (l.direction.x()*r.direction.y() - l.direction.y()*r.direction.x())

	rayDist := -(-l.direction.y()*l.start.x() + l.direction.y()*r.start.x() + l.direction.x()*l.start.y() - l.direction.x()*r.start.y()) / (r.direction.x()*l.direction.y() - l.direction.x()*r.direction.y())

	if lineDist < 0 || lineDist > l.length || rayDist < 0 {
		return false, -1, -1, -1
	}

	pos := l.start.add(l.direction.mul(lineDist))

	return true, pos.x(), pos.y(), pos.sub(r.start).length()
}

func (r *ray) refract(l *line, n float64) (ok bool, cropped *raySegment, continuation *ray) {
	ok, intX, intY, dist := r.intersect(l)
	if !ok {
		return false, nil, nil
	}

	var (
		n1 = n
		n2 = l.refractiveIndex

		a = l.start.x()
		b = l.start.y()
		c = l.direction.x()
		d = l.direction.y()
		k = d*a - c*b
		s = d*r.start.x() - c*r.start.y()
	)

	if (k-s)*float64(l.orientation) > 0 {
		n1, n2 = n2, n1
	}

	var (
		parallel = l.direction
		matrix   = mat.NewDense(2, 2, []float64{
			-parallel.y(), parallel.x(),
			parallel.x(), parallel.y(),
		})
		inverse = mat.NewDense(2, 2, nil)
		iM      = new(vector)
	)

	if err := inverse.Inverse(matrix); err != nil {
		fmt.Println(err)
	}

	(*mat.Dense)(iM).Mul(inverse, (*mat.Dense)(r.direction))

	var (
		R     = iM.y() / iM.x()
		X     = (n1 / n2) * (R / math.Sqrt(1+R*R))
		theta = math.Abs(math.Asin(X))
		rMx   = math.Copysign(math.Cos(theta), iM.x())
		rMy   = math.Copysign(math.Sin(theta), iM.y())
		rM    = vec(rMx, rMy)
		rT    = new(vector)
	)

	// If X < -1 or X > 1, total internal reflection should occur.
	if X > 1 || X < -1 {
		rM = vec(-iM.x(), iM.y())
	}

	(*mat.Dense)(rT).Mul(matrix, (*mat.Dense)(rM))

	cropped = &raySegment{
		start:     r.start,
		direction: r.direction,
		length:    dist / r.direction.length(),
	}

	continuation = &ray{
		start:     vec(intX, intY),
		direction: rT,
	}

	return
}

func (r *ray) project(lines []*line, max int) (segments []*raySegment, final *ray) {
	final = r
	ignore := -1

	for count := 0; count < max; count++ {
		var (
			intersection = -1
			distance     = math.Inf(1)
		)

		for i, l := range lines {
			if ok, _, _, dist := final.intersect(l); ok && dist < distance && i != ignore {
				intersection = i
				distance = dist
			}
		}

		if intersection == -1 {
			break
		}

		ignore = intersection

		_, cropped, continuation := final.refract(lines[intersection], 1)

		final = continuation
		segments = append(segments, cropped)
	}

	return
}

func newLine(start, end *vector, o int, n float64) *line {
	var (
		diff      = end.sub(start)
		direction = diff.norm()
		length    = diff.length()
	)

	return &line{
		start:           start,
		direction:       direction,
		length:          length,
		orientation:     o,
		refractiveIndex: n,
	}
}

func lensWidthAt(r, c, k, K float64) float64 {
	if 1-(K+1)*c*c*r*r < 0 {
		return -1
	}

	return k + (c*r*r)/(1+math.Sqrt(1-(K+1)*c*c*r*r))
}

func makeLens(resolution, sx, sy, tx, ty, index, c, k, K float64) []*line {
	var (
		lines = make([]*line, 0)
		prev  = -1.0
	)

	for x := 0.0; lensWidthAt(x, c, k, K) > 0; x += resolution {
		width := lensWidthAt(x, c, k, K)

		if prev >= 0 {
			lines = append(lines, newLine(
				vec((x-resolution)*sx+tx, prev*sy+ty),
				vec(x*sx+tx, width*sy+ty),
				-1, index,
			))

			lines = append(lines, newLine(
				vec(-((x-resolution)*sx+tx), prev*sy+ty),
				vec(-(x*sx+tx), width*sy+ty),
				1, index,
			))

			lines = append(lines, newLine(
				vec((x-resolution)*sx+tx, -prev*sy+ty),
				vec(x*sx+tx, -width*sy+ty),
				1, index,
			))

			lines = append(lines, newLine(
				vec(-((x-resolution)*sx+tx), -prev*sy+ty),
				vec(-(x*sx+tx), -width*sy+ty),
				-1, index,
			))
		}

		prev = width
	}

	return lines
}

func pt(x, y float64) (float64, float64) {
	return (x*scale + width/2), height - (y*scale + height/2)
}

func main() {
	dc := gg.NewContext(width, height)
	dc.DrawRectangle(0, 0, width, height)
	dc.SetRGB(1, 1, 1)
	dc.Fill()

	dc.SetRGB(0.9, 0.8, 0.9)
	dc.SetLineWidth(1.0)

	for x := gridGap * scale; x < width; x += gridGap * scale {
		for y := gridGap * scale; y < height; y += gridGap * scale {
			dc.MoveTo(float64(x-gridSize), float64(y))
			dc.LineTo(float64(x+gridSize), float64(y))
			dc.Stroke()

			dc.MoveTo(float64(x), float64(y-gridSize))
			dc.LineTo(float64(x), float64(y+gridSize))
			dc.Stroke()
		}
	}

	n := 4.4

	/* lines := []*line{
		// Bottom and top
		newLine(vec(-40, 16), vec(40, 24), 1, n),
		newLine(vec(-40, 32), vec(40, 40), -1, n),

		// Left and right
		newLine(vec(-40, 16), vec(-40, 32), -1, n),
		newLine(vec(40, 24), vec(40, 40), 1, n),
	} */

	lines := makeLens(0.05, 4, 1, 0, -40, n, -0.04, 8.2, 1.9)

	for x := -30.0; x <= 30.0; x += 5 {
		ray := &ray{
			start:     vec(x, -100),
			direction: vec(0, 1),
		}

		segments, final := ray.project(lines, 32)

		for _, seg := range segments {
			seg.draw(dc)
		}

		final.draw(dc, 500)
	}

	for _, line := range lines {
		line.draw(dc)
	}

	dc.SavePNG("out.png")
}
