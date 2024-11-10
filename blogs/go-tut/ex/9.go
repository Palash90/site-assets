// Functions

package main

import "fmt"

func main() {

	fmt.Println(add(1, 2))
	fmt.Println(divide(9, 0))
	fmt.Println(divide(10, 3))

	// Assigning three values
	quotient, remainder, error := divide(5, 0)
	fmt.Println("5 divided by 0 results in quotient=", quotient, "remainder=", remainder, "error=", error)

	// Discarding values returned by functions
	quotient, _, _ = divide(10, 3)
	fmt.Println("10 divided by 3 results in quotient=", quotient)
}

func add(addend1 int, addend2 int) int {

	return addend1 + addend2
}

func divide(dividend int, divisor int) (int, int, error) {

	if divisor == 0 {
		return 0, 0, fmt.Errorf("Divide by Zero")
	} else {

		return dividend / divisor, dividend % divisor, nil
	}
}
