// Type check system

package main

import (
	"fmt"
)

func main() {
	var y float32

	y = 1
	fmt.Printf("y=%v, type=%T\n", y, y)

	// If you uncomment the following two lines, compilation will fail due to type check fail
	//var x int
	//y=x

	fmt.Printf("y=%v, type=%T\n", y, y)
}
