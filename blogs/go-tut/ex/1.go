// Line comment
/*
 * Block Comment
 */

// Package definition. Main means the code execution starts with this definition

// package mains
package main

import (
	"fmt"
)

// The main function definition is important, otherwise, it won't run. Try renaming the func 'mains'
func main() {

	fmt.Println("Hello") // Semi-colon is unnecessary but can work with one. Go fmt removed the Semi-colon.
	fmt.Println("Printed without Semi colon")
	fmt.Println("UTF Out of Box - 😊")
}
