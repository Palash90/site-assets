// Check ppointers with array and slice

package main

import "fmt"

func main() {
	arr := [3]int{1, 2, 3}
	fmt.Println("Initial arr =", arr)

	doubleSizedArray(arr)
	fmt.Println("After doubleSizedArray arr=", arr)

	doubleSizedArrayPtr(&arr)
	fmt.Println("After doubleSizedArrayPtr arr=", arr)


	// Following line throws an error due to type check
	// doubleUnsizedArray(arr)

	slc := []int{5, 10, 15}
	fmt.Println("slc=", slc)

	// Following line throws an error due to type check
	// doubleSizedArray(slc)

	doubleUnsizedArray(slc)
	fmt.Println("slc=", slc)

	s := "From Main"
	fmt.Println("s=",s)
	stringConcat(s)
	fmt.Println("After StringConcat s=", s)

	sp := "From Main pointer"
	fmt.Println(sp)
	stringConcatPtr(&sp)
	fmt.Println(sp)

}

// Arrays are passed by value
func doubleSizedArray(arr [3]int) {
	for a := range arr {
		a *= 2
	}
}

// Array pointer
func doubleSizedArrayPtr(arr *[3]int) {
	for a := range *arr {
	(*arr)[a] *= 2
	}
}


// Slices are passed by reference
func doubleUnsizedArray(arr []int) {
	for a := range arr {
		arr[a] *= 2
	}
}

// Like Array String is also passed by value
func stringConcat(str string) {
	str += "-Added By function"
}

// String pointer
func stringConcatPtr(str *string) {
	*str += "-Added By function"
}
