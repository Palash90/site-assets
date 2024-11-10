// Different flavours of for loop

package main

import (
	"fmt"
)

func main() {
	fmt.Println("Regular for loop")
	// Regular for loop
	for i := 0; i < 3; i++ {
		fmt.Println(i)
	}

	// For loop with break
	fmt.Println("\n for loop with break")
	for i := 0; i < 5; i++ {
		if i == 2 {
			break
		}
		fmt.Println(i)
	}

	// For loop with continue
	fmt.Println("\n for loop with continue")
	for i := 0; i < 5; i++ {
		if i == 2 {
			continue
		}
		fmt.Println(i)
	}

	// For as while
	i := 0
	for i < 3 {
		fmt.Println(i)
		i++
	}

	// For as while true
	i = 0
	for {
		if i == 3 {
			break
		}
		fmt.Println(i)
		i++
	}

}
