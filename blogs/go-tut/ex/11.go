// Use of defer

package main

import "fmt"

func main() {
	worker()
}

func worker() {

	// Some code to acquire some resource

	fmt.Println("Resource A acquired")
	fmt.Println("Resource B is dependent on Resource A")
	fmt.Println("Resource B is acquired")

	defer cleanResource("Resource A")
	defer cleanResource("Resource B")

	fmt.Println("Working with Resource B")
	fmt.Println("Working with Resource A")

	fmt.Println("Worker complete")

}

func cleanResource(resource string) {
	fmt.Println("Cleaning up", resource)
}
