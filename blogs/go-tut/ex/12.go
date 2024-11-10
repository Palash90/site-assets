// Basic structs

package main

import (
	"fmt"
	"math/rand"
)

type Product struct {
	// Any variable name with Init Cap can be accessed publicly
	Name string

	Quantity int

	Price float64

	Sell bool

	// Any variable name starting with small letter is treated as default modifier of Java
	shelf int
}

// This is a method of struct Product, this will not change the existing value of the struct
// The values of the struct will be copied in this method
func (p Product) Value() float64 {

	if p.Sell {
		return -(float64(p.Quantity) * p.Price)
	} else {
		return float64(p.Quantity) * p.Price
	}
}

// This method is a pointer based method and can change the value of the passed pointer
// The method name and attribute name cannot be the same. Had to rename to SellProduct
func (p *Product) SellProduct() {
	p.Sell = true
}

// This function creates a Product instance and returns. This is similar to constructor in Java/C#

func NewProduct(name string, quantity int, price float64, sell bool) (*Product, error) {
	if name == "" {
		return nil, fmt.Errorf("Product name cannot be empty")
	}

	if quantity < 0 {
		return nil, fmt.Errorf("Quantity should be more than 0, was provided %v", quantity)
	}

	if price < 0 {
		return nil, fmt.Errorf("Price of a product cannot be negative")
	}
	p := &Product{
		Name:     name,
		Quantity: quantity,
		Price:    price,
		Sell:     sell,
		shelf:    rand.Intn(5),
	}

	return p, nil
}

func main() {

	// Empty definition of struct, all values will be assigned default value for the corresponding type
	p1 := Product{}

	// The following will just print the values of the attributes in the struct
	fmt.Println(p1)
	// Use +v to better visualize the struct in a JSON like format
	fmt.Printf("%+v\n", p1)

	// Like Java/C/C# access variables in the struct
	p1.Name = "Diya"
	p1.Quantity = 100
	p1.Price = 2
	p1.Sell = false
	p1.shelf = 4

	fmt.Printf("%+v\n", p1)

	// Declaring and initializing the struct
	// Like assigning in map, you need the traling comma in each attribute
	p2 := Product{
		Name:     "Rangoli",
		Quantity: 50,
		Price:    10,
		Sell:     true,
		shelf:    5,
	}

	fmt.Printf("%+v\n", p2)

	// Declaring struct with only some attributes, attributes that are unassigned gets a default value
	p3 := Product{
		Name:     "Oil",
		Quantity: 50,
	}

	fmt.Printf("%+v\n", p3)

	fmt.Println("Total Inventory Value before sell", p3.Value()+p2.Value()+p1.Value())
	p1.SellProduct()
	fmt.Println("Total Inventory Value after sell", p3.Value()+p2.Value()+p1.Value())

	// Creating Product instance with constructor function
	p4, err := NewProduct("Dry Fruits", 50, 400, false)
	if err != nil {
		fmt.Println(err)
	}
	fmt.Printf("Product from constructor function %+v\n", p4)

	fmt.Println("Total Inventory Value before sell", p3.Value()+p2.Value()+p1.Value()+p4.Value())
	p4.SellProduct()
	fmt.Println("Total Inventory Value after sell", p3.Value()+p2.Value()+p1.Value()+p4.Value())
}
