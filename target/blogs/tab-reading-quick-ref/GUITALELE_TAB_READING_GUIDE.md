# 𝄞 Guitalele Tab Viewer: Musician's Practice Manual

Welcome to the **Guitalele Tab Viewer**! This interactive songbook is a practice tool made specifically for the guitalele—a travel-friendly 6-string instrument tuned like a classical guitar but downsized to the body of a ukulele.

This guide focuses strictly on what you need to know as a musician reading the music, tracking your place, and using the player to practice effectively.

---

## 1. Interface & Layout Overview

When you open a song, the viewer dynamically arranges itself to fit your screen. On a computer or large tablet, it displays up to **4 measures per line** so you can see long musical phrases at once. On a mobile phone, it automatically refolds down to **1 or 2 measures per line** so you never have to pinch or zoom to see your music.

The screen is split into three main regions:

* **The Top Control Bar**: Where you play, pause, stop, and change the speed of the song.
* **The Main Notation Canvas**: The interactive sheet music running down the center of the page.
* **The Floating Sidebar**: A context window on the right side of your screen that dynamically updates with information about whatever note you are playing or hovering over.

---

## 2. Reading the Interactive Sheet Music

Each musical line combines three different views stacked vertically. This layout helps you track your melody, accompaniment, string positions, and rhythms simultaneously.

```
+------------------------------------------------------------+
| [1] Standard Staves  --> Classic sheet music notation      |
|                                                            |
| [2] Guitalele TAB    --> String lines and fret numbers     |
|                                                            |
| [3] Rhythm Lane      --> Quick-reference timing symbols    |
+------------------------------------------------------------+

```

### [1] Standard Notation Staves (Top Rows)

To give you a complete picture of the music, the viewer splits notes across two traditional clefs:

* **Treble Clef Staff**: Displays your higher-pitched notes (usually your melody).
* **Bass Clef Staff**: Displays your lower-pitched notes (usually your basslines).
* **Stem Directions (Multi-Voice Coding)**: If a song has two independent parts playing at the same time—such as a thumb-picked bassline and a fingerpicked melody line—the notes for the melody will have their stems pointing **up**, while the bass notes will have their stems pointing **down**.

### [2] Guitalele Tablature / TAB (Middle Rows)

The 6 horizontal lines represent the 6 strings of your guitalele, viewed as if you were looking down at the fretboard while holding it.

* **The String Tuning**: Because a guitalele is tuned a perfect fourth higher than a regular guitar (**A - D - G - C - E - A**), the lines represent:
* **Top Line (String 1)**: The thinnest, highest-pitched string (**A4**).
* **Bottom Line (String 6)**: The thickest, lowest-pitched string (**A2**).


* **Fret Numbers**: Numbers placed on these lines tell you exactly where to press your finger. A `0` indicates an open string (pluck it without holding down any fret).

### [3] Rhythm Lane (Bottom Row)

At the base of the layout is a dedicated rhythm guide. It uses clean, text-based shorthand symbols underneath the notes so you can instantly judge note lengths without counting complex flags:

* **`o`** = Whole note (4 beats)
* **`.`** = Half note (2 beats)
* **`:`** = Quarter note (1 beat)
* **`+`** = Eighth note (1/2 beat)
* **`=`** = Sixteenth note (1/4 beat)
* *A dot right next to a symbol (e.g., `:.`) multiplies its baseline duration by 1.5.*
* *Any symbol starting with **`r`** (such as `r`, `r+`, `r=`) represents a **rest**—a period of explicit silence lasting for that exact length.*

---

## 3. Playback Controls & Practice Tools

The top dashboard houses tools to customize how you listen and play along with a piece.

* **Play (`▶`) & Pause (`||`)**: Click Play to kick off the audio engine, which synthesizes a nylon-string acoustic guitalele sound on the fly. Click Pause to freeze the song at an exact spot so you can adjust your fingers or practice an awkward chord shift.
* **Stop (`■`)**: Instantly silences all ringing notes and rewinds the tracking cursor back to the absolute beginning of the song.
* **BPM (Beats Per Minute) Slider**: Slowing down a piece is one of the best ways to practice. Drag the slider to the left to decelerate a blazing fast solo, or drag it to the right to pick up the tempo.
* *Note: To protect the audio timing loop from glitching, the speed slider locks while a song is playing. Simply hit Stop to unlock it, slide to your new speed, and restart.*


* **Automatic View Tracking**: As the song plays, a bright highlighter tracks the current notes. When the tracker moves to a new row, the screen automatically scrolls smoothly to position the active line at the top of your screen, keeping your hands free for your instrument.

---

## 4. Using the Interactive Sidebar

The sidebar on the right side of the screen functions as a live translator for your sheet music.

```
+------------------------------+
|     CURRENT NOTE DETAILS     |
|==============================|
| MEASURE: 4                   |
| VOICE: 1 (Melody)            |
| TYPE: QUARTER NOTE    |
|                              |
| FRETBOARD POSITIONS:         |
| • Note: E4  -> String 2, F0  |
| • Note: C4  -> String 3, F0  |
+------------------------------+

```

Whenever you pause the music and hover your mouse over or tap into a note cluster on the canvas, the sidebar decodes it into plain English. It tells you:

1. **The Measure Number** and which structural track layer (Voice 1 melody or Voice 2 bass) the note belongs to.
2. **The Exact Rhythmic Value** typed out cleanly (e.g., `"Eighth Note"` or `"Dotted Quarter Note"`).
3. **The Note Map**: A bulleted breakdown of every single note in that chord, showing the traditional letter name (like `C#` or `E`), which string to play, and which fret to press.

---

## 5. Identifying Song Errors (Red Measures)

The viewer is equipped with an automated time checker. If a song file has a rhythmic error (for instance, if a bar accidentally contains 4.5 beats instead of 4 beats), the viewer marks it right on your screen.

If a measure turns a **translucent red color** and displays a warning symbol ($\triangle$) near the measure number, that signifies a time imbalance.

* Look right below the TAB lines for a status stamp like `v1: 3.50 / 4.00`.
* This tells you that Voice 1 in this bar only adds up to 3.5 beats instead of the required 4.0.
* While the player will still attempt to play through the mistake, seeing this red block lets you know why a specific measure might sound rushed, skipped, or structurally uneven.