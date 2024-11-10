// Fizz Buzz with user input from command line arguments
// Usage: go run fizzbuzz.go 20

package main

import (
	"fmt"
	"os"
	"strconv"
)

func main() {

	till, _ := strconv.Atoi(os.Args[1])

	i := 1

	for {

		if i > till {
			break
		}

		if i%3 == 0 && i%5 == 0 {
			fmt.Println("Fizz Buzz")
		} else if i%3 == 0 {
			fmt.Println("Fizz")
		} else if i%5 == 0 {
			fmt.Println("Buzz")
		} else {
			fmt.Println(i)
		}

		i++
	}
}
