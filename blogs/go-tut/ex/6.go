// Strings in go

package main

import (
	"fmt"
)

func main() {

	fav := "Harry Potter"

	// built in len function
	fmt.Println("fav=", fav, ", Length", len(fav))

	// extracting a single character from string
	fmt.Printf("fav[4]=%v %T\n", fav[4], fav[4])

	// extracting slices, the indexing concept is a little confusing. So, check yourself and get your own understanding
	fmt.Println("fav[4:8]", fav[4:8])
	fmt.Println("fav[4:]", fav[4:])
	fmt.Println("fav[:4]", fav[:4])

	// If you uncomment the following line, you will have an error
	//fav[4]=56

	// String concatenation
	fmt.Println(fav + " and Ron")

	// Multi Line String declaration
	an := `
	1. Harry Potter
	2. Lord Voldemort
	`

	fmt.Println(an)

	name := "Palash"

	// Use of Sprintf and C# style variable declaration within function
	var placeHolder = fmt.Sprintf("%v %T", name, name)

	fmt.Println(placeHolder)
}
