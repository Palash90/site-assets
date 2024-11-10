// Slices and array

package main

import (
	"fmt"
)

func main() {
	// Array Declaration
	var arr = [5]int{1, 2, 3} // Shorthand of initialization

	// Slice declaraion, notice the length is omitted
	var sl = []int{8, 5, 4, 3, 2}

	fmt.Printf("arr =%v, %T\n", arr, arr)
	fmt.Printf("sl=%v, %T\n", sl, sl)

	// Use of built in functions on slice
	fmt.Println("Length of sl before append", len(sl))

	// Adding an element to slice using built in append
	sl1 := append(sl, 4)

	// Append does not change the existing slice, it returns a new slice
	fmt.Println("Length of sl after append", len(sl))
	fmt.Println("Length of sl1 after append", len(sl1))
	fmt.Println("sl, sl1", sl, sl1)

	// Array and slice traversing using simple for loop
	fmt.Println("Array elements")
	for i := 0; i < len(arr); i++ {
		fmt.Println(arr[i])
	}

	fmt.Println("Slice elements")
	for i := 0; i < len(sl); i++ {
		fmt.Println(sl[i])
	}

	// Use of range
	// Range with appended array, range gives indexes
	fmt.Println("Appended slice elements")
	for i := range sl1 {
		fmt.Println("Position=", i, "value=", sl1[i])
	}

	// Use of range with two variables, this gives both the index and the value.
	fmt.Println("Appended slice elements with two context range")
	for i, v := range sl1 {
		fmt.Println("Position=", i, "value=", v)
	}

	// Discarding the index
	fmt.Println("-----------------")
	for _, v := range sl1 {
		fmt.Println("value=", v)
	}

	// Some slice operations
	fmt.Println("arr=", arr, "\tsl=", sl)
	fmt.Println("arr[0]=", arr[0], "\tsl[2]=", sl[2])
	fmt.Println("arr[2:]=", arr[2:], "\tsl[1:]=", sl[1:])
	fmt.Println("arr[:2]=", arr[:2], "\tsl[:3]=", sl[:3])

	fmt.Println("arr[1:2]=", arr[1:2], "\tsl[1:3]=", sl[1:3])

}
