// If statement
// Usage: go run 4.1.go 8

package main

// I am using os package to get the console arguments
// Using strconv to convert string to integer
import (
	"fmt"
	"os"
	"strconv"
)

func main() {
	// Go way of Integer.parseInt() and Getting console arguments
	a, error := strconv.Atoi(os.Args[1])
	fmt.Printf("Error %v\n", error)

	if a < 0 || a > 20 {
		fmt.Println("Input out of range")
	} else if a < 5 {
		fmt.Println("Input is too small")
	} else if a >= 5 && a <= 10 {
		fmt.Println("Input is just right")
	} else {
		fmt.Println("Input is too big")
	}

	if half := a / 2; half < 3 {

		fmt.Println("Very less half life")
	}

	// following line will throw error due to type mismatch
	//	if half1 := a / 2; half1 < 3.5 {

	//		fmt.Println("Very less half life")
	//	}

}
