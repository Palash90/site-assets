# Writing a very Simple Terminal Plotter in Rust

In the journey of writing this guide - [Machine Learning from First Principles](https://ai.palashkantikundu.in/), I set a challenging constraint: **zero third-party libraries**. This project is about a minimalistic, systems-level understanding—building tensors, matrix operations, and backpropagation in Rust so you can inspect every memory access and gradient step.

But even when building from scratch, you can't fly blind. Visualization is a necessity in ML. To stay true to the "zero dependency" rule, I had to build my own plotting tool using nothing but the Rust standard library and terminal ANSI codes.

## The Philosophy: Radical Transparency

Most developers reach for a plotting library immediately. However, when the goal is **mastery over production**, adding a massive dependency tree feels like a cheat. By building our own plotter, we ensure that the tools we use to verify our math are just as transparent as the math itself.

## The End Result

Before jumping into implementation, here is a glimpse what it does:


![Image description](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/irby8b338hd3gfppefne.gif)


And here is a more complex one:


![Image description](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/l6zusgb7343qxh7d82bt.gif)


## Defining the Data: The `Trace` Struct

Before we can render a single pixel, we need a way to describe our data. The `Trace` struct acts as our container for data series, allowing us to toggle between scatter plots and line graphs.

```rust
pub struct Trace {
    pub name: String,
    pub x: Vec<f32>,
    pub y: Vec<f32>,
    pub color: PlotColor,
    pub is_line: bool,
}

```

This allows us to overlay multiple metrics like "Training Loss" vs. "Validation Loss" using a variety of ANSI-powered colors.


## The Rendering Engine: `render_plot`

The heart of the tool is the `render_plot` function. It constructs an entire coordinate system within a 2D grid of strings.

### 1. Mapping and Normalization

Since terminal dimensions are fixed (e.g., 80x40), but data values can be anything, we use a `map_val` helper to scale our floats into grid coordinates.

### 2. Drawing Lines with "Lerp"

To visualize a continuous function, we can't just plot dots. We implement a linear interpolation algorithm in `draw_line` to fill the gaps between data points with middle-dot characters (`·`).

```rust
fn draw_line(grid: &mut Vec<Vec<String>>, x0: usize, y0: usize, x1: usize, y1: usize, color: &str) {
    let steps = (x1 as i32 - x0 as i32).abs().max((y1 as i32 - y0 as i32).abs());
    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        let x = (x0 as f32 + (x1 as i32 - x0 as i32) as f32 * t) as usize;
        let y = (y0 as f32 + (y1 as i32 - y0 as i32) as f32 * t) as usize;
        // ... grid boundary check and coloring ...
    }
}

```

### 3. UI Polish: Title and Spacing

To make the output readable during fast training loops, the plotter includes:

* **Buffer Gaps:** Two empty lines at the top to separate the plot from previous terminal output.
* **Centered Titles:** A bold, Cyan-colored title centered horizontally based on the plot width.

```rust
// Atomic Buffer Print with UI polish
buffer.push_str("\n\n"); // The Gap
let padding = (width - title.len()) / 2;
buffer.push_str(&format!("{}\x1b[1;36m{}\x1b[0m\n\n", " ".repeat(padding), title.to_uppercase()));

```

## Why This Matters

This isn't about making the terminal look pretty; it's about ownership. When you build the plotter yourself:

1. **You understand the coordinate system.** You aren't guessing how your data is scaled.
2. **Just pure dopamine**

## If you want to use it

Here is how I came up to it:

```rust
use std::f32;

pub struct Trace {
    pub name: String,
    pub x: Vec<f32>,
    pub y: Vec<f32>,
    pub color: PlotColor,
    pub is_line: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum PlotColor {
    Red,
    Blue,
    Green,
    Cyan,
    Magenta,
    Yellow,
    White,
    Reset,
}

impl PlotColor {
    pub fn to_ansi(&self) -> &'static str {
        match self {
            PlotColor::Red => "\x1b[31m",
            PlotColor::Blue => "\x1b[34m",
            PlotColor::Green => "\x1b[32m",
            PlotColor::Cyan => "\x1b[36m",
            PlotColor::Magenta => "\x1b[35m",
            PlotColor::Yellow => "\x1b[33m",
            PlotColor::White => "\x1b[37m",
            PlotColor::Reset => "\x1b[0m",
        }
    }
}

pub fn render_plot(
    traces: &[Trace],
    width: usize,
    height: usize,
    fixed_bounds: Option<(f32, f32, f32, f32)>,
    title: String,
) {
    let (min_x, max_x, min_y, max_y) = match fixed_bounds {
        Some(bounds) => bounds,
        None => get_bounds(traces),
    };

    let margin_l = 10;
    let margin_b = 2;
    let plot_w = width - margin_l - 2;
    let plot_h = height - margin_b - 2;

    let y_tick_count = 5;
    let x_tick_count = 4;

    let mut grid = vec![vec![" ".to_string(); width]; height];

    for i in 0..=y_tick_count {
        let t = i as f32 / y_tick_count as f32;
        let py = map_val(t, 0.0, 1.0, plot_h as f32, 0.0) as usize;
        let val = map_val(t, 0.0, 1.0, min_y, max_y);

        grid[py][margin_l] = "┼".to_string();

        let label = format!("{:>9.1}", val);
        for (idx, c) in label.chars().enumerate() {
            if idx < margin_l {
                grid[py][idx] = c.to_string();
            }
        }
    }

    for i in 0..=x_tick_count {
        let t = i as f32 / x_tick_count as f32;
        let px = map_val(t, 0.0, 1.0, 0.0, plot_w as f32) as usize + margin_l + 1;
        let val = map_val(t, 0.0, 1.0, min_x, max_x);

        if px < width {
            grid[plot_h][px] = "┴".to_string();

            let label = format!("{:.1}", val);
            for (idx, c) in label.chars().enumerate() {
                if px + idx < width {
                    grid[plot_h + 1][px + idx] = c.to_string();
                }
            }
        }
    }

    for y in 0..plot_h {
        if grid[y][margin_l] == " " {
            grid[y][margin_l] = "│".to_string();
        }
    }
    for x in margin_l + 1..width {
        if grid[plot_h][x] == " " {
            grid[plot_h][x] = "─".to_string();
        }
    }
    grid[plot_h][margin_l] = "└".to_string();

    for trace in traces {
        let color_code = trace.color.to_ansi();
        for i in 0..trace.x.len() {
            let px = map_val(trace.x[i], min_x, max_x, 0.0, plot_w as f32) as usize + margin_l + 1;
            let py = map_val(trace.y[i], min_y, max_y, plot_h as f32 - 1.0, 0.0) as usize;

            if py < plot_h && px > margin_l && px < width {
                if trace.is_line && i > 0 {
                    let prev_px = map_val(trace.x[i - 1], min_x, max_x, 0.0, plot_w as f32)
                        as usize
                        + margin_l
                        + 1;
                    let prev_py =
                        map_val(trace.y[i - 1], min_y, max_y, plot_h as f32 - 1.0, 0.0) as usize;
                    draw_line(&mut grid, prev_px, prev_py, px, py, color_code);
                }
                grid[py][px] = format!("{}●\x1b[0m", color_code);
            }
        }
    }

    let mut buffer = String::new();
    buffer.push_str("\x1b[2J\x1b[H\x1b[?25l");

    buffer.push_str("\n\n");
    let title_len = title.len();
    if title_len < width {
        let padding = (width - title_len) / 2;
        buffer.push_str(&" ".repeat(padding));
    }
    buffer.push_str(&format!("\x1b[1;36m{}\x1b[0m\n\n", title.to_uppercase()));

    for row in grid {
        buffer.push_str(&row.concat());
        buffer.push('\n');
    }

    buffer.push('\n');
    for t in traces {
        buffer.push_str(&format!(
            "{} {} {} \x1b[0m  ",
            t.color.to_ansi(),
            if t.is_line { "──" } else { "●" },
            t.name
        ));
    }
    print!("{}", buffer);
    println!("\x1b[?25h");
}

fn draw_line(grid: &mut Vec<Vec<String>>, x0: usize, y0: usize, x1: usize, y1: usize, color: &str) {
    let steps = (x1 as i32 - x0 as i32)
        .abs()
        .max((y1 as i32 - y0 as i32).abs());
    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        let x = (x0 as f32 + (x1 as i32 - x0 as i32) as f32 * t) as usize;
        let y = (y0 as f32 + (y1 as i32 - y0 as i32) as f32 * t) as usize;
        if y < grid.len() && x < grid[0].len() {
            grid[y][x] = format!("{}·\x1b[0m", color);
        }
    }
}

fn get_bounds(traces: &[Trace]) -> (f32, f32, f32, f32) {
    let all_x: Vec<f32> = traces.iter().flat_map(|t| t.x.iter()).cloned().collect();
    let all_y: Vec<f32> = traces.iter().flat_map(|t| t.y.iter()).cloned().collect();
    (
        *all_x
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap(),
        *all_x
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap(),
        *all_y
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap(),
        *all_y
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap(),
    )
}

fn map_val(val: f32, in_min: f32, in_max: f32, out_min: f32, out_max: f32) -> f32 {
    if (in_max - in_min).abs() < 1e-6 {
        return out_min;
    }
    (val - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
}

pub fn main() {
    let mut traces = vec![];

    let t = Trace {
        name: "Predict 1".to_string(),
        x: vec![-5.0, -4.0, -3.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        y: vec![25.0, 16.0, 9.0, 1.0, 0.0, 1.0, 4.0, 9.0, 16.0, 25.0],
        color: PlotColor::Cyan,
        is_line: true,
    };

    traces.push(t);

    render_plot(&traces, 70, 25, None, String::from("Parabola"));
}
```

It's not perfect, it's not highly optimized or anything but works out of box, you don't need to spend a whole day setting up things to just go through one chapter. You own it. If something bothers you, some `println!();` statements and you are good to go.

That's it for now, we'll soon meet when I build something else.


