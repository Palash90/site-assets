// Automatic type recognition

package main

import (
	"fmt"
)

func main() {

	x := 1
	y := 2
	avg := (x + y) / 2

	// Something like python tuples
	divider := 2
	x1, x2 := 1.0, 2.0

	// following line will throw an error due to type mismatch
	// avg1 := (x1 + x2) / divider

	// However, the following line will not throw an error. Because, it infers 2 as 2.0 (my guess)
	avg1 := (x1 + x2) / 2

	fmt.Printf("divider=%v, type=%T\n", divider, divider)
	fmt.Printf("avg=%v, type=%T\n", avg, avg)
	fmt.Printf("avg1=%v, type=%T\n", avg1, avg1)
}
