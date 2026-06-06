# Shorthand Tablature Syntax Guide

This guide details how to format shorthand musical score data so that the parser can accurately convert it into digital notation.

### 1. Document Structure
A shorthand text file can contain one or multiple scores. Each score block must begin with metadata headers followed by its chronological musical measures.

#### Global Headers & Metadata
Headers must appear at the top of a score block, one per line. They are case-sensitive and must be followed by a colon (`:`).

* `Score: [Title]` — **(Required)** Instantiates a new score block.
* `Instrument: [Name]` — Optional (e.g., Guitalele). **Only Guitalele is supported now**
* `Time Signature: [X/Y]` — Optional (e.g., 4/4, 3/4). **Default:** 4/4
* `Capo: [Number]` — Optional integer. If provided, fret numbers will automatically shift upward by this value (except for muted or open strings).
* `Description: [Text]` — Optional global notes.

> **Note:** Empty lines, structural separators starting with `==`, or lines starting with `GUITALELE TAB` are ignored, allowing you to visually partition your files cleanly.

---

### 2. Formatting Measures
Measures hold the rhythmic tokens of your music. You can declare them explicitly or implicitly:

* **Explicit:** Prefix the line with `Measure X:` or `M X:` (e.g., `M 1: 0f6sq`).
* **Implicit:** Any plain text line following the metadata headers without an explicit prefix is automatically processed as the next sequential measure.
* **Bar lines (`|`):** You can use vertical pipes to visually separate measures or tokens; the parser will safely strip or bypass them.

---

### 3. Note and Event Tokens
Tokens inside a measure are separated by spaces. A token represents a single note, a rest, or a chord, and can hold additional modifiers.

#### Quick-Reference Duration Table
The following characters are used to define note and rest durations:

| Flag | Duration Type | Value (Beats) |
| :--- | :--- | :--- |
| `s` | Sixteenth | 0.25 |
| `e` | Eighth | 0.50 |
| `e.` | Dotted Eighth | 0.75 |
| `q` | Quarter (Default) | 1.00 |
| `q.` | Dotted Quarter | 1.50 |
| `h` | Half | 2.00 |
| `h.` | Dotted Half | 3.00 |
| `w` | Whole | 4.00 |
| `w.` | Dotted Whole | 6.00 |

---

### 4. Token Syntax Flavors

#### A. Rests
* **Syntax:** `-`
* **Example:** `-@h` (A half-note rest).

#### B. Single Notes
You can write single notes using either **Compact** or **Legacy** syntax.

* **Compact Syntax (Recommended):** `[Fret]f[String]s[Duration]`
    * *Example:* `3f6sq` (Fret 3, String 6, Quarter note duration).
    * *Special Frets:* Use `O` or `o` for open strings (0). Use `X` or `x` for muted strings (null).
* **Legacy Syntax:** `[Fret]:[String]`
    * *Example:* `2:3` (Fret 2, String 3). Defaults to a quarter note unless an explicit duration flag is appended.

#### C. Chords
Chords wrap multiple notes inside square brackets `[...]`.
* **Syntax:** `[Note1 Note2 Note3][Duration]`
* **Internal notes** can use compact notation (`2f4s`) or legacy colon notation (`2:4`). Inside the brackets, notes are separated by spaces or pipes.
* **Example (Compact Chord):** `[0f6s 2f5s 2f4s]w` (E minor triad held for a whole note).
* **Example (Legacy Chord):** `[0:6 2:5 2:4]h`

---

### 5. Advanced Modifiers (Appended to Tokens)
Any token (single note, rest, or chord) can accept these flags. They should be appended to the token string:

* **Duration Modifier (`@`):** Explicitly overrides the token duration if you aren't using compact syntax.
    * *Example:* `3:6@h.` (Fret 3, String 6, held for 3 beats).
* **Tie Flag (`t`):** Marks the note or chord to tie into the next event.
    * *Example:* `2f3sqt` (A tied quarter note).
* **Voice Flag (`v[Number]`):** Assigns the token to a specific polyphonic voice layer.
    * *Example:* `0f6sqv1` (Assigned to Voice 1).
* **Inline Description (`d:[Text]`):** Adds text/lyric annotations to a specific token. **This must always be placed at the absolute end of the token.**
    * *Example:* `3f1sqtd:Hammer-on` (Fret 3, String 1, quarter note, tied, with a "Hammer-on" description tag).

---

## Example Document Blueprint

```text
================================================================================
Score: C Major Scale
Time Signature: 4/4
Instrument: Guitalele
Capo: 0
Description: C Major Scale
================================================================================

0f3sqd:C4 2f3sqd:D4 0f2sqd:E4 1f2sqd:F4
3f2sqd:G4  0f1sqd:A4  2f1sqd:B4 3f1sqd:C5
```