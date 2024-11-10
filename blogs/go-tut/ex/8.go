// Map in go

package main

import (
	"fmt"
)

func main() {

	m := map[string]string{

		"a": "A",

		"b": "B",

		"c": "C", // The trailing comma is a must. Without this the program will not compile

	}

	fmt.Println(m["a"])

	// The following line will not return <nil> or throw an error, it simply returns the default value of the type.
	// For string type, the default value is empty string, so the following line will return empty string
	fmt.Println(m["d"])

	// Followin is the way to check if the key returned any value using the double context return
	v, keyExists := m["d"]

	fmt.Println("Key exists-", keyExists, "value-", v)

	fmt.Println(len(m))

	// Upsert a value
	m["c"] = "Third Character of English alphabet"

	m["d"] = "D"

	fmt.Println("c is", m["c"], "d is", m["d"])

	//Delete a value
	delete(m, "d")

	fmt.Println(m)

	// Iterating a map using for and range
	fmt.Println("single context range only returns keys")
	for v := range m {
		fmt.Println(v)
	}

	fmt.Println("Double context range returns keys and values")
	for i, v := range m {
		fmt.Println(i, v)
	}
}
