---
title: "Multi dimensional array"
date: "2020-03-26T13:57:00+05:30"
recovered_from: "web.archive.org (snapshot 20200923012245)"
---

Multi-dimensional arrays are one of the most confusing element for beginners. In hope of helping the beginners, I am writing this post. To make things simple, I will assume we use only and only integer numbers.

**Array (1D Array)

To understand multi-dimensional array, we need to first understand an array. An array is a data structure that contains a group of elements. So, a collection of multiple items make an array.

![image](https://cdn.programiz.com/sites/tutorial2program/files/c-arrays.jpg)

Follow the example -

2 - a single element

[2, 3, 4] - an array of elements

[2] - an array containing only one element

So, an array can be thought of as a row of elements. Where each element has a index associated with it. Arrays can give access to its elements using an index. Array indexes start from 0.

Follow the program -

```
var a = [1,2,3]

console.log(a)

console.log(a[0])

```

**2D Array

**

A 2D Array is an array of multiple 1D Arrays. Now we know 1D array, so, we can stack them together to form a 2D array.

In 1D array, we saw that each element has an index and we can access the elements using the index. In case 2D Arrays, we do the same. The only difference is that, for 1D array, we access a single element using index, in 2D arrays, we access an array using an index.

Follow the program -

```
var _2d = [[1,2],[3,4],[5,6]]

console.log(_2d) //output [[1,2],[3,4],[5,6]]

console.log(_2d[0]) //output [1,2]

console.log(_2d[1]) //output [3,4]

console.log(_2d[2]) //output [5,6]

```

We saw how we can access an array from a 2D Array, now we will see how we can access a single element from a 2D Array. To access a single element from a 2D array, we need two indexes, first one is to access the nested 1D array and the second one to get access to the single element.

```
console.log(_2d[0][0]) // output 1

console.log(_2d[0][1]) // output 2

console.log(_2d[1][0]) // output 3

console.log(_2d[1][1]) // output 4

console.log(_2d[2][0]) // output 5

console.log(_2d[2][1]) // output 6

```

Remember that, we can also change the values of the 2D arrays using the similar way we did in 1D array

```
_2d[0][0] = 100

console.log(_2d) //output [[100,2],[3,4],[5,6]]

```

**3D Array

A 3D Array is an array of multiple 2D Arrays. we can extend the same process of defining and using a 2D array to define and use 3D Array.

Here, the array index gives you access to a 2D Array.

```
var _3d= [ [ [1,2], [3,4], [5,6] ], [ [7,8], [9,10], [11,12] ] ]

console.log(_3d) // output [ [ [1,2], [3,4], [5,6] ], [ [7,8], [9,10], [11,12] ] ]

console.log(_3d[0]) //output [ [1,2], [3,4], [5,6] ]

conosle.log(_3d[1]) //output [ [7,8], [9,10], [11,12] ]

```

Now to access a single element in the array, now we need 3 indexes, the first one is to get the 2D Array, the second is to access the nested 1D array, the third is to get the element in the 1D array.

```
var _3d= [ [ [1,2], [3,4], [5,6] ], [ [7,8], [9,10], [11,12] ] ]

console.log(_3d) // output [ [ [1,2], [3,4], [5,6] ], [ [7,8], [9,10], [11,12] ] ]

conosle.log(_3d[0][0][0]) // output 1

conosle.log(_3d[0][0][1]) // output 2

conosle.log(_3d[0][1][0]) // output 3

conosle.log(_3d[0][1][1]) // output 4

conosle.log(_3d[0][2][0]) // output 5

conosle.log(_3d[0][2][1]) // output 6

conosle.log(_3d[1][0][0]) // output 7

conosle.log(_3d[1][0][1]) // output 8

conosle.log(_3d[1][1][0]) // output 9

conosle.log(_3d[1][1][1]) // output 10

conosle.log(_3d[1][2][0]) // output 11

conosle.log(_3d[1][2][1]) // output 12

```

The modification also works the same way.

```
console.log(_3d) // output [ [ [1,2], [3,4], [5,6] ], [ [7,8], [9,10], [11,12] ] ]

_3d[0][0][1]=50

console.log(_3d) // output [ [ [1,50], [3,4], [5,6] ], [ [7,8], [9,10], [11,12] ] ]

```

**4D Array

Moving on to the next stage, 4 dimensional array. Now, we know what a 3 dimensional array looks like. Now we will extend it to create 4 dimensional array.

A 4 dimensional array is an array of a three dimensional array.

```
var _4d = [[[[1,2], [3,4]], [[5,6], [7,8]]],  [[[9,10], [11,12]], [[13,14], [15,16]]]]

```

Here, the array index gives you access to a 3D Array.

```
console.log(_4d[0])//output [ [ [ 1, 2 ], [ 3, 4 ] ], [ [ 5, 6 ], [ 7, 8 ] ] ]

```

Now to access a single element in the array, now we need 4 indexes, the first one to get the nested 3D Array, the second one is to get the nested 3D Array, the third is to access the nested 1D array, the fourth is to get the element in the 1D array.

```
console.log(_4d[0][0][0][0]) //output 1

console.log(_4d[0][0][0][1]) //output 2

console.log(_4d[0][0][1][0]) //output 3

console.log(_4d[0][0][1][1]) //output 4

console.log(_4d[0][1][0][0]) //output 5

console.log(_4d[0][1][0][1]) //output 6

console.log(_4d[0][1][1][0]) //output 7

console.log(_4d[0][1][1][1]) //output 8

console.log(_4d[1][0][0][0]) //output 9

console.log(_4d[1][0][0][1]) //output 10

console.log(_4d[1][0][1][0]) //output 11

console.log(_4d[1][0][1][1]) //output 12

console.log(_4d[1][1][0][0]) //output 13

console.log(_4d[1][1][0][1]) //output 14

console.log(_4d[1][1][1][0]) //output 15

console.log(_4d[1][1][1][1]) //output 16

```

Modification of the array elements can be done in the same way.

**Multi-dimensional array

Based on the concept of extending, we can simply stack arrays of dimension n and get n+1 dimensional array.

For example-

We stacked array of dimension 0 (a single element) to form a one dimensional array.

We stacked array of dimensional 1 to form a two dimensional array

We stacked array of dimensional 2 to form a three dimensional array

We stacked array of dimensional 3 to form a four dimensional array

In the same way we can stack n dimensional array to get an array of n+1.

Hope this post helps you have a clear concept of arrays.

<div class='
