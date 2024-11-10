// Default integer types in Go and Unused Variable Declaration

package main

import (
	"fmt"
)

// The following program will not even compile due to unused variable declaration.
func main() {

	var x int // Variable definition
	var y int // The int type has different register format based on 32 bit or 64 bit systems. Most like the early age C programs, where you get an integer of either 16 bit size or a 32 bit size based on platform.

	x = 1
	y = 1

	// Use _ to discard values returned by function
	len, _ := fmt.Println("x=%v type of x=%T", x, x)
	fmt.Println("Printed length", len)
}
