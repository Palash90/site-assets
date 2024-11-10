// Get float division

package main

import (
	"fmt"
)

func main() {

	var num1 float32
	var num2 float32

	num1 = 1
	num2 = 2

	var avg = (num1 + num2) / 2

	fmt.Printf("Average is %v, type %T", avg, avg)
	fmt.Println()
}
