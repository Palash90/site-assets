The Motivation
==============

To make some progress in my Rust Learning Journey, I was doing some research about projects to write. The project should give some learning opportuinities of Data Structures, Algorithms and System Architecture. It should also have some provision to become a distributed one in future. After few days of studying, I finally fixed on Mini Redis.

Reasons are obvious -
1. It is easy to implement
1. It is memory bound
1. I may have to learn and implement many Data Structures (I know I will re-invent the wheel, even though Rust's `std` has many required Data Structures for the job)
1. With some few tweaks, it can even be re-modeled as persistent database
1. I can integrate my other distributed system project - [Distributed File System](https://github.com/Palash90/dist-fs)

Then, I tried learning more about Redis. One thing that caught my eyes were how it handles requests single threadedly. Later I found that, JavaScript also works in similar fashion. In fact, I was a little heart-broken when I knew that `setTimeOut` is actually not part of JavaScript, but __Web API__. Having implemented and used servers earlier, one thing got stuck in my head, "for a client to handle, you need a `thread` making the architecture, one thread per client".

So, the idea of single thread really caught me into the loop. So much so that, I myself tried to implement a mini version of the same. I built a functional one but by no means, it is perfect. From a real world use case scenario, this is another toy project.

__NOTE:__ This is not the exact same replica of Redis (or JS) Event Loop, Redis handles things differently and more efficiently and at huge scale. What I have done here is just for learning and demonstration purpose. You can grab some idea by going through my journey and this post may help you to understand how Redis actually handles things in real working piece of software which handles thousands of requests per second with just one single thread.

The Plan
========
Alright. The disclaimers are at place and now I am ready to describe you my adventure and I strongly recommend you to start along with an open mind and an open IDE.

I did not use much of the library support except for the basic `std`. Also, almost have never experimented anything outside [The Book](https://doc.rust-lang.org/stable/book/). So, if you have prior experience with The Book, it should be fairly easy to follow along.

First thing I will show, how to implement the TCP Server and how to respond to client. Then I will show you how a server can interact with multiple clients using `thread`. However, as I am trying to make a single threaded server, I will pivot back to single thread but with some design change.

Lastly, I will open up discussion for future extensions like `async/await` or `epoll`.

Let's start.

The Client
==========
An artist is nothing without his/her fan base. Similarly a server is nothing without its clients. So, I will first start with the client.

The client is not doing much here. It is just opening a connection to server and endlessly exchanging information until prompted to `quit`.

## The Imports
```rust
use std::io::{self, Read, Write};
use std::net::TcpStream;
```
The first line imports all the necessary functionality for input and output, like reading, flushing and writing to streams.

The second line imports the actual TCP Stream, that is responsible for the network connection.

## The function body
Once we are done with the imports, we need to start the connection between the server and the client. The following line does the job.

```rust
let mut stream = TcpStream::connect("127.0.0.1:7878")?;
```
You might notice the `?`. If you are not familiar with error propagation in rust, this operator propagates the error.

Once the connection is established, the client needs to send the server some request. To do this, I am writing an endless loop. Inside the loop body, I am printing some arbitary prompt to make it look like an REPL.

```rust
let mut input = String::new();
print!("mem-db$: ");
io::stdout().flush();
```
The flush operation is used here to immediately return to the prompt, which is not what `println!()` does automatically.

After this, we are opening up the standard input for user input.

```rust
io::stdin()
            .read_line(&mut input)
            .expect("Failed to read line");
```

And once the input is provided by the user, it is matched against `quit` command or not.

```rust
if input.trim().eq("quit") {
    println!("Good Bye");
    break;
}
```

If not, the client simply sends the command to the server.

```rust
stream.write_all(input.as_bytes())?;
```

Then the client waits for server's response and prints back to the console

```rust
let mut buffer = [0; 512];
let n = stream.read(&mut buffer)?;
println!("Received: {}", String::from_utf8_lossy(&buffer[..n]));
```

Here is the whole code of the simple client.
```rust
use std::io::{self, Read, Write};
use std::net::TcpStream;

fn main() -> io::Result<()> {
    // Connect to the server
    let mut stream = TcpStream::connect("127.0.0.1:7878")?;

    

    loop {
        let mut input = String::new();
        print!("mem-db$: ");
        io::stdout().flush();
        
        io::stdin()
            .read_line(&mut input)
            .expect("Failed to read line");

        if input.trim().eq("quit") {
            println!("Good Bye");
            break;
        }

        stream.write_all(input.as_bytes())?;

        // Read the response from the server
        let mut buffer = [0; 512];
        let n = stream.read(&mut buffer)?;
        println!("Received: {}", String::from_utf8_lossy(&buffer[..n]));
    }

    Ok(())
}
```