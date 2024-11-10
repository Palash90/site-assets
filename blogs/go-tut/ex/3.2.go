// Get average

package main

import (
	"fmt"
)

func main() {

	var x int
	var y int

	x = 1
	y = 2

	var avg int

	avg = (x + y) / 2

	// Average is 1 because, division between two integers results in integer.
	fmt.Printf("Average is %v\n", avg)

}
