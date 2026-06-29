# The Snowball That Became a Compiler

When I started this project around 3 weeks ago, I never thought this one project would compel me to face so many challenges.

It started with a simple idea. I needed something to practice the Guitalele, an instrument with a very limited fan following that never reached the heights of the Guitar or Ukulele. Because of that, resources like tabs, scores, and specialized tuners are incredibly scarce.

So, I sat down to build a simple text area that would take my own musical shorthand and parse it specifically to Guitalele standards.

Using AI as my primary co-pilot, we started building. And then, the engineering snowball completely took off.

## 1. The Editor

The first version was laughably simple — a `<textarea>` and a "Parse" button. You paste some shorthand, hit the button, and hope something renders on the other side.

For about an hour, that was enough.

## 2. Bells and Whistles to the Editor

Then the cracks showed. I needed to manage multiple scores — not just one. I needed to edit old ones, not just create new ones. I needed parameters like time signature, instrument, capo position, and a description for each score. A raw textarea was not going to cut it.

So the editor grew. A dropdown to switch between existing scores — all stored locally for now. Input fields for metadata — name, time signature (4/4, 3/4, 6/8, 2/4, 2/2), instrument (Guitalele for now, but the door is open), capo position (0 to 12). A publish/draft toggle. A delete button with a confirmation dialog. Auto-resizing textarea that grows as you type. A manual modal referencing every shorthand rule I had invented so far. No cloud, no server — just React state and a dream.

Once the basic editor was ready, the next itch started. There is a classic dialogue in the movie 'Yeh Jawaani Hai Deewani' -

>>Yaadein mithai ke dibbe ki tarah hoti hain...
>>Ek baar khula toh sirf ek tukda nahi kha paoge.

Well, here is Baba Palashananda Version of it:

>>Features mithai ke dibbe ki tarah hoti hain...
>>Ek baar scope khula toh sirf ek editor pe nahi rukoge.
>>
>>Poora app hi banaoge...

## 3. The Parser

The shorthand notation was my own invention. Simple enough to type quickly while practicing, but structured enough to represent any Guitalele score.

```
3f1sq        Fret 3, string 1, quarter note
[0:6 | 2:5]@q  G major chord, quarter note
-@q          Quarter rest
M1: 3f3sq | 3f3sq | 4f3sq | 5f3sq
```

I wrote a `parseToken` function to handle individual notes, then a `parseShorthandText` function to handle the full score. The parser had to deal with:

- **The @ syntax**: `3:1@q` — fret:string@duration
- **The compact syntax**: `3f1sq` — fret + f + string + s + duration
- **Chords**: `[3:1 | 5:2]@q` — notes wrapped in brackets
- **Rests**: `-@q`, `-h` — silence with duration
- **Two voices**: `v1` for melody, `v2` for bass — independent polyphonic lines
- **Ties**: `t` after duration — hold the note into the next beat
- **Annotations**: `d:text` — notes to the reader, doesn't affect sound
- **Measure validation**: Each measure must total exactly the right number of beats for the time signature

The parser also validates. If a measure has 3.5 beats in a 4/4 time signature, it throws an error. If a string number is out of range (1-6 for Guitalele), it throws an error. If a bracket is missing its closing pair, it throws an error.

```js
export const parseShorthandText = shorthandText => {
    const lines = shorthandText.split("\n");
    const scores = [];
    const errors = [];
    let currentScore = null;

    for (let lineIdx = 0; lineIdx < lines.length; lineIdx++) {
        const line = lines[lineIdx].trim();
        if (!line || line.startsWith("==")) continue;

        if (line.startsWith("Score:")) {
            if (currentScore) scores.push(currentScore);
            currentScore = { id: "", title: "", instrument: "",
                timeSignature: "4/4", measures: [], capo: 0 };
            continue;
        }
        // Parse measures, tokens, validate durations...
    }
    return { scores, errors };
};
```

I never formally studied compiler design. But here I was — writing a lexer, a parser, a validator, and a semantic analyzer for a domain-specific language I had just invented over a weekend. The "magic" of compilers didn't feel so magical anymore. It felt like a long chain of `if` conditions and regex patterns.

## 4. A Simple Note Player

Once the parser could produce structured data, I needed to hear if the output was correct. So I built a note player — a bare-bones audio thing using the Web Audio API. Pluck a synthesized string, play back a measure, validate by ear.

I did not know at the time that this "simple note player" would eventually grow into a full SVG-based score viewer with tablature rendering, sheet music notation, voice polyphony, metronome, auto-scroll, and a dark-themed GUI. But that story is for the next part.

## 5. The AI Review

At this point, I had a local tool that could parse shorthand, play notes, and display results. I showed the prototype to Gemini.

Gemini suggested adding a "Create" button.

A Create button won't work without knowing who created it. That needs login. Login means authentication. Auth means a backend. A backend means Firestore. And if I am going through all the trouble of setting up Firestore, why not just save my scores there?

## 6. "What Would I Lose?"

I sat with the trade-off for a moment.

**What would I lose?** A few days of effort. Maybe a week.

**What would I gain?** A working, persistent score library accessible from anywhere. No more copy-pasting stubs into blog files. No more context switching between tools. And underneath it all, knowledge — about authentication flows, about Firestore security rules, about real-world CRUD patterns, about shipping a full-stack feature from start to finish.

Oh, and a few sleepless nights. Well, three weeks to be very precise.

I made the call. The snowball wasn't done rolling.

## 7. The Cloud Backend

Up went Firebase. Authentication with Google OAuth and email/password. A `profiles` collection for user data. A `scores` collection with a composite key: `username:instrument:slugified-title`.

```js
const makeScoreDocId = (uname, instr, title) => {
  const s = slugify(title);
  return `${uname}:${slugify(instr)}:${s}`;
};
```

The CRUD operations came next. Load all scores for a user on page load (`getDocs` with a `where` query). Save a new score (`setDoc`). Update an existing one (`updateDoc`). Delete (`deleteDoc`). Publish/unpublish toggle. Email verification check before allowing writes.

```js
const loadScores = useCallback(async () => {
  if (!user) return;
  const q = query(scoresRef, where("userId", "==", user.uid));
  const snap = await getDocs(q);
  // ... build list, sort by creation date
}, [user]);
```

The security rules had to be right too. Readable by all (published scores are public), but writable only by the owner with a verified email. No one should overwrite someone else's score.

## 8. Self Use and the First Publication

With the cloud backend in place, I used the tool myself for four days. Wrote a few scores. The first one I published was **Ode to Joy** — a chord melody arrangement for Guitalele. I even made a YouTube Shorts to go with it.

It was working. People could see it. I could share it.

But looking at the raw shorthand text in the editor started to hurt my eyes.

## 9. The Syntax Highlighting

Raw shorthand text in a plain textarea is hard to read. All those `3f1sq@qd:` tokens blend into a grey soup. After staring at it for days, I needed color — desperately.

So I wrote a custom syntax highlighting system from scratch. No libraries. Just regex and inline styles.

```js
const COLOR_SCHEME = {
  separator: '#334155',
  headerLabel: '#67e8f9',
  headerValue: '#e2e8f0',
  measureLabel: '#c084fc',
  pipe: '#475569',
  bracket: '#fbbf24',
  chordContent: '#fde68a',
  duration: '#4ade80',
  voice: '#22d3ee',
  tie: '#f87171',
  annotation: '#fb923c',
};
```

Every token type gets its own color. Fret numbers in slate, strings in gray, durations in green, voices in cyan, annotations in orange, ties in red. The textarea sits transparent over a `<pre>` layer that renders the highlighted version. You type in invisible text over a colorful display — a trick borrowed from code editors like VS Code.

```js
function highlightShorthand(text) {
  const escaped = text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  const lines = escaped.split('\n');
  const coloredLines = lines.map((line) => {
    if (/^={2,}|^---+/.test(line))
      return `<span style="color:${COLOR_SCHEME.separator}">${line}</span>`;
    if (/^(Score|Time Signature|Instrument|Capo|Description|ID):/.test(line))
      return line.replace(/^(Score|Time Signature|Instrument|Capo|Description|ID):(.*)$/,
        (_, label, val) => `<span style="color:${COLOR_SCHEME.headerLabel}">${label}</span><span style="color:${COLOR_SCHEME.headerValue}">:${val}</span>`
      );
    // ... more token colorization
  });
  return coloredLines.join('\n');
}
```

Two functions and about 80 lines of regex. But those 80 lines make the difference between "what did I just type?" and "oh, I see, the quarter note is missing on beat 3."

## The Inventory

Three weeks in, here's what I had built:

1. **An Editor** — 815 lines of React with score metadata management, load/save/delete/publish workflow
2. **A Parser** — Full shorthand-to-structured-data pipeline with validation, error reporting, multi-voice polyphony support
3. **A Note Player** — Web Audio API pluck synthesis that would later evolve into something much bigger
4. **A Cloud Backend** — Firebase Auth + Firestore CRUD with security rules, user profiles, and published score browsing
5. **A Syntax Highlighter** — Custom tokenizer coloring 15+ token types across the shorthand DSL

And none of this was planned. The editor was supposed to be a debugger. The parser was born from boredom. The note player was a validation tool. The cloud backend was an AI's suggestion I took as a dare. The highlighter came from eye strain after publishing my first real score.

All because I wanted to practice a niche instrument that nobody writes tabs for.

The snowball hasn't stopped rolling. But that's a story for another day.
