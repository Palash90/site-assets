// Switch Expression

package main

import (
	"fmt"

	"os"
	"strconv"
)

func main() {

	x, _ := strconv.Atoi(os.Args[1])

	switch x {

	case 1:
		fmt.Println("One")
	case 2:
		fmt.Println("Two")
	default:
		fmt.Println("Undefined Choice")
	}

	// Switch without expression

	switch {
	case x <= 1:
		fmt.Println("OK")
	case x > 1 && x <= 3:
		fmt.Println("Cool")
	case x > 3:
		fmt.Println("Bad")
	}
	switch x {

	case 1, 2:
		fmt.Println("Less than 3")
	default:
		fmt.Println("Out of range")
	}

	switch even := x%2 == 0; even {

	case true:
		fmt.Println("Even")
	case false:
		fmt.Println("Odd")
	}
}
