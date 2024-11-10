// Operation between a float32 and float64

package main

import (
	"fmt"
)

func main() {

	var x float32
	var y float64

	var avg float64

	avg = (x + y) / 2

	fmt.Printf("Average is %v, type=%T\n", avg, avg)

	var i int
	var u uint

	var sa int

	sa = i + u

	fmt.Printf("Sum is %v, type=%T\n", sa, sa)
}
