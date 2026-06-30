# The Snowball That Became a Compiler

When I started this project around 3 weeks ago, I never thought this one project would compel me to face so many challenges.

It started with a simple idea. I needed something to practice the Guitalele, an instrument with a very limited fan following that never reached the heights of the Guitar or Ukulele. Because of that, resources like tabs, scores, and specialized tuners are incredibly scarce.

So, I sat down to build a simple text area that would take my own musical shorthand and parse it specifically to Guitalele standards.

Using AI as my primary co-pilot, we started building. And then, the engineering snowball completely took off.

## 1. The Editor

The first version was laughably simple — a `<textarea>` and a `Parse` button. You paste some shorthand, hit the button, and hope something renders on the other side.

For about an hour, that was enough.

## 2. Bells and Whistles to the Editor

Then the cracks showed. I needed to manage multiple scores, not just one. I needed to edit old ones, not just create new ones. I needed parameters like time signature, instrument, capo position, and a description for each score. A raw textarea was not going to cut it.

So the editor grew. A dropdown to switch between existing scores — all stored locally for now. Input fields for metadata — name, time signature (4/4, 3/4, 6/8, 2/4, 2/2), instrument (Guitalele for now, but the door is open), capo position (0 to 12). A publish/draft toggle. A delete button with a confirmation dialog. Auto-resizing textarea that grows as you type. A manual modal referencing every shorthand rule I had invented so far. No cloud, no server — just React state and a dream.

Once the basic editor was ready, the next itch started. There is a classic dialogue in the movie 'Yeh Jawaani Hai Deewani' -

>>Yaadein mithai ke dibbe ki tarah hoti hain...
>>Ek baar khula toh sirf ek tukda nahi kha paoge.

(_*English Translation:*_ "Memories are like a box of sweets... Once it opens, you won't be able to stop at just one piece.")

Well, here is Baba Palashananda Version of it:

>>Features mithai ke dibbe ki tarah hoti hain...
>>Ek baar scope khula toh sirf ek editor pe nahi rukoge.
>>
>>Poora app hi banaoge...

(_*English Translation:*_ "Features are like a box of sweets... Once the scope opens up, you won't just stop at a single code editor. You'll end up building the whole damn app...")

## 3. The Parser

I know Pianists use staff notations, guitarists tabs, ukulelists also use some version of tabs. All the available materials were standardized for these systems. Guitalele is orphaned in this respect.

The shorthand notation was my own invention. I started with this `0:6@q`. You won't believe me how easy things look on paper, until you actually implement and use it.

It was very easy on paper to write `:`, `@` etc. on paper. 

I wrote a `parseToken` function to handle individual notes, then a `parseShorthandText` function to handle the full score. The parser had to deal with:

- **Measure #**: `Measure 1` — First measure
- **The @ syntax**: `3:1@q` — fret:string@duration
- **Chords**: `[3:1 | 5:2]@q` — notes wrapped in brackets
- **Rests**: `-@q` — silence with duration
- **Ties**: `t` after duration — hold the note into the next beat

The first version of the parser I implemented could decode the following:

```guitalele shorthand
Measure 1: [0:6 | 2:5]@h -@h
Measure 2: 3:3@q | 3:3@q | 4:3@q | 5:3@q
```

Excited me started typing his very first tab on a system he implemented, a 'King' feel.

### The bubble burst

It was not very long, I understood, the system I implemented is utterly problematic to type and utterly confusing as well.

I also missed two major features:

- **Two voices**: `v1` for melody, `v2` for bass — independent polyphonic lines, one for chord and another for melody
- **Measure validation**: Each measure must total exactly the right number of beats for the time signature

Along with those two features, I have to come up with another idea of shorthand. Already 6 hours passed by that time still I am unable to even write a single full score. Even 'Twinkle Twinkle Little Star' felt like a chore to transcribe on my system.

Time for a break. I straight went to terrace and started looking at the clouds. After almost 30 minutes or so when I came back to my desk, the solution was quite clear, why not simply use fret, string notation, exactly how I show it on my channel?

And thus the first appearance of:

- **The compact syntax**: `3f1sq` — fret + f + string + s + duration

Almost every parsing was taken care, just had a task of character replacement. I finally did not replace the existing `:` and `@` parsing, rather extended it with new syntax.

Along with this, also removed the mandate of `Measure #` notation thus making writing even easier. Now with text based parsing, it also felt easier to actually introduce description. However, as the syntax is space based, I can't put space in the description yet. May be fix for some later time.

- **Annotations**: `d:text` — notes to the reader, doesn't affect sound

After all the features completed, within another hour few more validations entered the scene.

If a measure has 3.5 beats in a 4/4 time signature, it throws an error. If a string number is out of range (1-6 for Guitalele), it throws an error. If a bracket is missing its closing pair, it throws an error.

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

Compiler design was not part of my syllabus any time. It always felt magical to me how `if`, `else`, `for` convert into a string of `1` and `0`.

But here I was — writing a lexer, a parser, a validator, and a semantic analyzer for a domain-specific language I had just invented over a weekend. The "magic" of compilers didn't feel so magical anymore. It felt like a long chain of `if` conditions and regex patterns.

## 4. A Simple Note Player

Once the parser could produce structured data, I needed to hear if the output was correct. So I built a note player, a bare-bones audio thing using the Web Audio API.

At that time, it felt like a toy I got when I was 9. But I did not know at the time that this "simple toy note player" would eventually grow into a full SVG-based score viewer with tablature rendering, sheet music notation, voice polyphony, metronome, auto-scroll, and a dark-themed GUI.

But that story is for the next part.

## 5. The AI Review

At that point, I had a local tool that could parse shorthand, play notes, and display results. I showed the prototype to Gemini.

Gemini suggested adding a "Create" button.

A Create button won't work without knowing who created it. That needs login. Login means authentication. Auth means a backend. A backend means Firestore. And if I am going through all the trouble of setting up Firestore, why not just save my scores there?

## 6. "What Would I Lose?"

Gemini struck a string in me which resonated far longer than I expected. Trade off decision took quite longer than I expected it to be.

**What would I lose?** A few days of effort. Maybe a week.

**What would I gain?**

- A working, persistent score library accessible from anywhere
- No more copy-pasting stubs into blog files
- No more context switching between tools
- And underneath it all, knowledge of designing and developing system from scratch

Oh, and a few sleepless nights. Well, three weeks to be very precise.

I made the call. The snowball won't stop rolling.

## 7. The Cloud Backend

I just don't know why I chose Firebase. Maybe because, I spend significant office hours working in AWS, so an alternative for home.

1. Up went Firebase.
2. Authentication with Google OAuth and email/password.
3. A `profiles` collection for user data.
4. A `scores` collection with a composite key: `username:instrument:slugified-title`.

```js
const makeScoreDocId = (uname, instr, title) => {
  const s = slugify(title);
  return `${uname}:${slugify(instr)}:${s}`;
};
```

The CRUD operations came next. Firestore made the experience very smooth. No backend needed at all. Simple DB calls, that too direct from client browser. I once thought about it, how nice would it be to directly call DB from UI. Well, turns out Google gave the option, which I was not aware of.

After successfully launching the web app I can realistically claim, I have seen the practical limitations of a No-Backend architecture. Trust me, beyond few easy peasy work you absolutely need a strong backend layer.

Here are few things that I leveraged:

1. Load all scores for a user on page load (`getDocs` with a `where` query).
2. Save a new score (`setDoc`).
3. Update an existing one (`updateDoc`).
4. Delete (`deleteDoc`).
5. Publish/unpublish toggle.
6. Email verification check before allowing writes.

```js
const loadScores = useCallback(async () => {
  if (!user) return;
  const q = query(scoresRef, where("userId", "==", user.uid));
  const snap = await getDocs(q);
  // ... build list, sort by creation date
}, [user]);
```

The security rules had to be right too. Readable by all (published scores are public), but writable only by the owner with a verified email. No one should overwrite someone else's score.

If you are not coming from GCP or Firebase background, Security Rules are simple validations which  you can use to limit access to collections and actions on them.

## 8. Self Use and the First Publication

With the cloud backend in place, I used the tool myself for four days. Wrote a few scores. The first one I published was **Ode to Joy** — a chord melody arrangement for Guitalele. I even made a YouTube Shorts to go with it.

It was working. People could see it. I could share it.

But looking at the raw shorthand text in the editor started to hurt my eyes.

## 9. The Syntax Highlighting

Raw shorthand text in a plain textarea is hard to read. All those `3f1sqd:F4_Note` silver white tokens blend into a grey soup. After staring at them for days, I needed color. Desperately.

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

Every token type gets its own color. Fret numbers in slate, strings in gray, durations in green, voices in cyan, annotations in orange, ties in red. The textarea sits transparent over a `<pre>` layer that renders the highlighted version. You type in invisible text over a colorful display, a trick borrowed from code editors like VS Code.

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

Although I am the only one using it and may be, I would always be the only one using it. Still it feels to put a logical end to anything I start.

## The Inventory

A week in, here's what I had built:

1. **An Editor** — 815 lines of React with score metadata management, load/save/delete/publish workflow
2. **A Parser** — Full shorthand-to-structured-data pipeline with validation, error reporting, multi-voice polyphony support
3. **A Note Player** — Web Audio API that would later evolve into a much bigger part of the whole system
4. **A Cloud Backend** — Firebase Auth + Firestore CRUD with security rules, user profiles, and published score browsing
5. **A Syntax Highlighter** — Custom tokenizer coloring 15+ token types across the shorthand DSL

And none of this was planned. The editor was supposed to be a debugger. The parser was born from boredom. The note player was a validation tool. The cloud backend was an AI's suggestion I took as a dare. The highlighter came from eye strain after publishing my first real score.

All because I wanted to practice a niche instrument that nobody writes tabs for.

The snowball hasn't stopped rolling. But that's a story for another day.
