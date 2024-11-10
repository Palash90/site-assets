// Different integer definitions of Go

package main

import (
	"fmt"
)

func main() {

	var i8 int8
	var i16 int16
	var i32 int32
	var i64 int64

	fmt.Println("Before initialization of integers")
	fmt.Printf("i8=%v, type is %T\n", i8, i8)
	fmt.Printf("i16=%v, type is %T\n", i16, i16)
	fmt.Printf("i32=%v, type is %T\n", i32, i32)
	fmt.Printf("i64=%v, type is %T\n", i64, i64)
	fmt.Println("\n")

	i8 = 1
	i16 = 2
	i32 = 4
	i64 = 8

	fmt.Println("After initialization of integers")
	fmt.Printf("i8=%v, type is %T\n", i8, i8)
	fmt.Printf("i16=%v, type is %T\n", i16, i16)
	fmt.Printf("i32=%v, type is %T\n", i32, i32)
	fmt.Printf("i64=%v, type is %T\n", i64, i64)
	fmt.Println("\n")

	var ui8 uint8
	var ui16 uint16
	var ui32 uint32
	var ui64 uint64

	fmt.Println("Before initialization of unsigned integers")
	fmt.Printf("ui8=%v, type is %T\n", ui8, ui8)
	fmt.Printf("ui16=%v, type is %T\n", ui16, ui16)
	fmt.Printf("ui32=%v, type is %T\n", ui32, ui32)
	fmt.Printf("ui64=%v, type is %T\n", ui64, ui64)
	fmt.Println("\n")

	ui8 = 8
	ui16 = 16
	ui32 = 32
	ui64 = 64

	fmt.Println("After initialization of unsigned integers")
	fmt.Printf("ui8=%v, type is %T\n", ui8, ui8)
	fmt.Printf("ui16=%v, type is %T\n", ui16, ui16)
	fmt.Printf("ui32=%v, type is %T\n", ui32, ui32)
	fmt.Printf("ui64=%v, type is %T\n", ui64, ui64)
	fmt.Println("\n")

	var f32 float32
	var f64 float64

	fmt.Println("Before Initialization of floats")
	fmt.Printf("f32=%v, type=%T\n", f32, f32)
	fmt.Printf("f64=%v, type=%T\n", f64, f64)
	fmt.Println("\n")

	f32 = 3.2
	f64 = 6.4

	fmt.Println("Before Initialization of floats")
	fmt.Printf("f32=%v, type=%T\n", f32, f32)
	fmt.Printf("f64=%v, type=%T\n", f64, f64)
	fmt.Println("\n")

	var by byte // Alias for uint8

	by = 2

	fmt.Printf("by=%v, type=%T\n", by, by)
	fmt.Println()
}
