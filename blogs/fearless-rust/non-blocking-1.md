The Motivation
==============

To make some progress in my Rust Learning Journey, I was doing some research about projects to write. The project should give some learning opportuinities of Data Structures, Algorithms and System Architecture. It should also have some provision to become a distributed one in future. After few days of studying, I finally fixed on Mini Redis.

Reasons are obvious -
1. It is easy to implement
1. It is memory bound
1. I may have to learn and implement many Data Structures (I know I will re-invent the wheel, even though Rust's `std` has many required Data Structures for the job)
1. With some few tweaks, it can even be re-modeled as persistent database
1. I can integrate my other distributed system project - [Distributed File System](https://github.com/Palash90/dist-fs)

With that, I tried learning more about Redis. One thing that caught my eyes were how it handles requests single threadedly. Later I found that, JavaScript also works in similar fashion. In fact, I was a little heart-broken when I knew that `setTimeout` is actually not part of JavaScript, but __Web API__. Having implemented and used servers earlier, one thing got stuck in my head, "for a client to handle, you need a `thread`. This makes the architecture, one thread per client".

So, the idea of single thread handling multiplel clients really caught me into the loop. So much so that, I myself tried to implement a mini version of the same. I built a functional one but by no means, it is perfect. From a real world use case scenario, this is just another toy project.

__DISCLAIMER:__ This is not the exact same replica of Redis (or JS) Event Loop, Redis handles things differently and more efficiently and at huge scale. What I have done here is just for learning and demonstration purpose. You can grab some idea by going through my journey and this post may help you to understand how Redis actually handles things in real working piece of software which handles thousands of requests per second with just one single thread.

The Plan
========
Alright. The disclaimers are at place and now I am ready to describe you my adventure and I strongly recommend you to start along with an open mind and an open IDE.

I did not use much of the library support except for the basic `std`. Also, almost have never experimented anything outside [The Book](https://doc.rust-lang.org/stable/book/). So, if you have prior experience with The Book, it should be fairly easy to follow along.

First thing I will show, how to implement the simple TCP Server and how to respond to client. Then I will show you how a server can interact with multiple clients using `thread`. However, multi-threaded approach has some problems which Redis tries to solve using single thread (or rather I say a Fixed Number of Threads). Hence, I will too pivot back to single thread but with some design change.

Lastly, I will open up discussion for future extensions like `async/await` and/or `epoll`.

Let's start.

The Client
==========
An artist is nothing without his/her fan base. Similarly a server is nothing without its clients. So, I will start with implementing the client.

The client is not doing much here. It is just opening a connection to server and endlessly exchanging information until prompted to `quit`.

## The Imports
```rust
use std::io::{self, Read, Write};
use std::net::TcpStream;
```
The first line imports all the necessary functionality for input and output, like reading, flushing and writing to streams.

The second line imports the actual TCP Stream, that is responsible for the network connection.

## The function body
Now the connection between the server and the client should be established. The following line does the job.

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

After this, the standard input is opened for user input.

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

    // An endless interaction to server
    loop {
        // User inputs the command and the client collects it
        let mut input = String::new();
        print!("mem-db$: ");
        io::stdout().flush();
        
        io::stdin()
            .read_line(&mut input)
            .expect("Failed to read line");

        // Check if `quit` command issued
        if input.trim().eq("quit") {
            println!("Good Bye");
            break;
        }

        // Send the command to server
        stream.write_all(input.as_bytes())?;

        // Read the response from the server
        let mut buffer = [0; 512];
        let n = stream.read(&mut buffer)?;
        println!("Received: {}", String::from_utf8_lossy(&buffer[..n]));
    }

    Ok(())
}
```
The Server
==========
Once done with the client, its time to write the first version of the server code. The server needs to listen to the port where it is hosted. So that, when there is some input sent by client, the server can get it and process it. Once the process is done, the server sends back the response on the same open channel. This is pretty common in [TCP protocol] (https://en.wikipedia.org/wiki/Transmission_Control_Protocol) and you can read more about it in the linked wikipedia page.

In Rust, `TcpListener` can be used to listen to the port. So along with the `Read` and `Write`, the `TcpListener` needs to be imported.

```rust
use std::io::{Read, Write};
use std::net::TcpListener;
```

In the client code, I mentioned the port `7878`. In order for client to connect to server properly, the server also needs to listen to the same port. This can be achieved, with the following code,

```rust
let listener = TcpListener::bind("127.0.0.1:7878")?;
println!("Server listening on port 7878");
```
Once connected, it lets user know that the port is free and the server is able to get hold of the port. There are multiple clients which can connect to the server. To facilitate that, the server needs to keep track of all the incoming connections and start serving them. The following code achieves the same.

```rust
for stream in listener.incoming() {
}
```
When there is any connection available waiting on the port, the server is notified and passed over the data stream. To handle the data from the stream, a buffer is initiated too. There is a guess work in the maximum size of data that is sent from the client and that here is 512. On the other hand, if there is any error, that is handled too.

```rust
match stream {
    Ok(mut stream) => {
        println!("Connnected");
        let mut buffer = [0u8; 512];
        /**
            Code to read and process the incoming data
        */
    }
    Err(e) => {
        eprintln!("Failed to accept connection: {}", e);
    }
}
```

Now, once the listener connects us to the underlying data stream, the server can read and process commands sent by the client. However, there is a chance that the stream is not properly read or client stops sending data or some network crash. To handle this, the stream read call wraps the read data in a `Result` enum. The following code matches on the `Result` and takes corresponding action.

```rust
match stream.read(&mut buffer) {
    Ok(0) => {
        // Client disconnected, drop the stream and exit the connection
        println!("Client disconnected, removing stream");
        break;
    }
    Ok(n) => {
        // Server reads the stream
        let op = String::from_utf8_lossy(&buffer[..n]).into_owned();
        println!("Received: {:?}", op);

        stream.write_all(&buffer[..n]).unwrap();
        stream.flush().unwrap();
    }
    Err(e) => {
        // Some other error occurred, drop the connection
        println!("Error reading from stream: {:?}", e);
    }
}
```
The server is kept simple here and simply returns what the client has sent, effectively making it an `echo` server. That's everything put together. The following is the complete code of the server.

```rust
use std::io::{Read, Write};
use std::net::TcpListener;

fn main() -> std::io::Result<()> {
    let listener = TcpListener::bind("127.0.0.1:7878")?;
    println!("Server listening on port 7878");
    
    for stream in listener.incoming() {
        match stream {
            Ok(mut stream) => {
                println!("Connnected");
                let mut buffer = [0u8; 512];  

                match stream.read(&mut buffer) {
                    Ok(0) => {
                        // Client disconnected, drop the stream and exit the connection
                        println!("Client disconnected, removing stream");
                        break;
                    }
                    Ok(n) => {
                        // Read the stream for client's input
                        let op = String::from_utf8_lossy(&buffer[..n]).into_owned();
                        println!("Received: {:?}", op);

                        // Return the same input back like an `echo` server
                        stream.write_all(&buffer[..n]).unwrap();
                        stream.flush().unwrap();
                    }
                    Err(e) => {
                        // Some other error occurred, drop the connection
                        println!("Error reading from stream: {:?}", e);
                    }
                }
            }
            Err(e) => {
                eprintln!("Failed to accept connection: {}", e);
            }
        }
    }
    

    Ok(())
}
```

Now let's run both the server and client in different terminals and observer the interaction.

## The Server Log
```shell
Server listening on port 7878
Connnected
Received: "hello\r\n"
Connnected
Received: "Hello from Client 2\r\n"
```

## Log from First Client
```shell
mem-db$: hello
Received: hello

mem-db$:
Error: Os { code: 10053, kind: ConnectionAborted, message: "An established connection was aborted by the software in your host machine." }
error: process didn't exit successfully: `target\debug\client.exe` (exit code: 1)
```

## Log from Second Client
```shell
mem-db$: Hello from Client 2
Received: Hello from Client 2

mem-db$:
Error: Os { code: 10053, kind: ConnectionAborted, message: "An established connection was aborted by the software in your host machine." }
error: process didn't exit successfully: `target\debug\client.exe` (exit code: 1)
```
What just happened here?
The client crashed just after one interaction, instead of interacting indefinitely. However, server keeps on accepting new client connections. Let's fix this.

If you go back and observe the server code, you will find that, once the server sends the response, it eventually gets out of the `for` loop where the control enters into another open connection, leaving the active connection unattended. The OS eventually closes the unused connection. To avoid this happening, the server needs to listen to the client stream continuously and this can be achieved with an indefinite loop. Inside the loop, the stream read and process instructions can run.

```rust
loop{
    match stream.read(&mut buffer) {
        Ok(0) => {
            // Client disconnected, drop the stream and exit the connection
            println!("Client disconnected, removing stream");
            break;
        }
        Ok(n) => {
            let op = String::from_utf8_lossy(&buffer[..n]).into_owned();
            println!("Received: {:?}", op);
    
            stream.write_all(&buffer[..n]).unwrap();
            stream.flush().unwrap();
        }
        Err(e) => {
            // Some other error occurred, drop the connection
            println!("Error reading from stream: {:?}", e);
        }
    }
}
```

Now the problem gets solved and connection between server and client is kept open. Let's run the server and two clients once again.

## Server log
```shell
Server listening on port 7878
Connnected
Received: "hello from client 1\r\n"
Received: "Hi Server\r\n"
Received: "How are you doing\r\n"
```

## Client 1 Log
```shell
mem-db$: hello from client 1
Received: hello from client 1

mem-db$: Hi Server
Received: Hi Server

mem-db$: How are you doing
Received: How are you doing

mem-db$:
```

## Client 2 Log
```shell
mem-db$: Hello from client 2
```

The server is now only handling the first client. It is ignoring the second client. If you read the server code again, you will notice that, the server waits indefinitely for first client, until it disconnects. 

So, now let's quit the first client and observe what happens.

## Server Log
```shell
Server listening on port 7878
Connnected
Received: "hello from client 1\r\n"
Received: "Hi Server\r\n"
Received: "How are you doing\r\n"
Client disconnected, removing stream
Connnected
Received: "Hello from client 2\r\n"
```

## Client 1 Log
```shell
mem-db$: hello from client 1
Received: hello from client 1

mem-db$: Hi Server
Received: Hi Server

mem-db$: How are you doing
Received: How are you doing

mem-db$: quit
Good Bye
```

## Client 2 Log
```shell
mem-db$: Hello from client 2
Received: Hello from client 2

mem-db$:
```
Now the server interacts with client 2 after client 1 disconnects. Now, if the first client is restarted, it will keep on waiting.

## Client 1 Log after restart
```shell
mem-db$: Hello from client 1 again

```
Now, server will respond to the first client once the second client quits the interaction.

This is what, is known as Blocking I/O problem. The main thread waits for client input or closed connection, while other clients keep on waiting for server response. This wastes so many CPU cycles waiting.

To get rid of this problem, servers started using multi-threaded environments. For each new connection a new thread was spawned unblocking all the threads.

If the indefinite loop is moved into its own thread, then for each client, the server can wait indefinitely for its client input.
```rust
thread::spawn(move || loop {
    match stream.read(&mut buffer) {
        Ok(0) => {
            // Client disconnected, drop the stream and exit the connection
            println!("Client disconnected, removing stream");
            break;
        }
        Ok(n) => {
            let op = String::from_utf8_lossy(&buffer[..n]).into_owned();
            println!("Received: {:?}", op);

            stream.write_all(&buffer[..n]).unwrap();
            stream.flush().unwrap();
        }
        Err(e) => {
            // Some other error occurred, drop the connection
            println!("Error reading from stream: {:?}", e);
        }
    }
});
```
Now, the server interacts with each client simultaneously and the server and client log shows that. You can even see in the server log line by line interaction by the clients. I am not going to show the client logs here. That won't be interesting.

## Multi-Threaded Server Log
```shell
Server listening on port 7878
Connnected
Received: "Hello from client 1\r\n"
Connnected
Received: "Hello from client 2\r\n"
Received: "Client 1: Hi Server, how are you?\r\n"
Received: "Client 2: Server dude, you are playing next level\r\n"
```

# The problem with multi-threading
