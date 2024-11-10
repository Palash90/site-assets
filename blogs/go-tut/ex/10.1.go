// Pointers

package main

import "fmt"

func main() {
	a := 1
	fmt.Println("a=", a)

	doubleInt(a)
	fmt.Println("After doubleInt a=", a)

	// Passing pointer
	doubleIntPtr(&a)
	fmt.Println("After doubleInt a=", a)

	// Passing pointer a regular value. The following line will throw an error due to type check
	//	doubleIntPtr(a)

}

func doubleInt(a int) {
	a *= 2
}

// Pointer definition
func doubleIntPtr(a *int) {

	// Dereferencing pointer
	*a *= 2
}
