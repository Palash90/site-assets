# The SVG Engine and the Crackle War

If Part 1 was about the snowball starting to roll, Part 2 is about the snowball turning into an avalanche.

After the editor, parser, cloud backend, and syntax highlighter were done, I had a tool that could save and display scores. But "display" was just highlighted text in an editor. What I really wanted was a visual score — tablature lines, staff notation, something that looks like actual sheet music.

That's where the trouble really began.

## 1. The First SVG Renderer: "Grand Stave notations"

The very first commit of `GuitaleleViewer.js` was called "Grand Stave notations." It was born on May 29 at 1:33 AM — already a sign of what was coming.

The concept was simple: take the parsed score data and draw it as SVG. Tablature lines for guitarists, staff notation for musicians, both synchronized to the same timeline.

I chose SVG over Canvas for one reason: **interactivity**. Canvas is faster for animation but every pixel is dead. In SVG, every note is a DOM element. You can hover over it, click it, see its properties. A musician wants to know "what note is this?" — they shouldn't need to count frets. SVG gives me that for free.

I started with a minimal renderer. A few `<line>` elements for the strings, some `<text>` nodes for fret numbers, a `<rect>` for the background. Dark themed, because my eyes were already tired from the editor.

```jsx
<svg width="..." height="...">
  <!-- 6 horizontal lines for strings -->
  <line x1="130" y1="100" x2="800" y2="100" stroke="#6a80b8" />
  <!-- Fret numbers -->
  <text x="150" y="118" fill="#e2e8f0">3</text>
</svg>
```

It was ugly. But it rendered.

Six commits followed in the same night: "Almost there", "new looks", "Almost fixed", "Almost working". Each one fixing some layout quirk, some alignment issue, some color that didn't pop against the dark background.

By 3:37 AM, I hit "Done basic Guitalele Viewer." It worked. Barely.

## 2. The Two-Voice System: When One Melody Isn't Enough

The next day, I realized a fundamental problem. Guitalele music often has two independent lines running simultaneously — a melody (Voice 1) and a bass/drone (Voice 2). Standard tab notation doesn't handle this well.

I needed polyphony.

"Two Voiced System is pending" — that commit was me acknowledging the scope.

An hour later: "much better shaped with two voiced system running."

Two hours later: "Mostly working with two voiced setup."

The implementation was tricky. Each voice has its own rhythm lane, its own color (cyan for V1, orange for V2), its own stem direction. When both voices play on the same beat, they need to merge into one visual slot without overlapping. When one voice rests and the other plays, the rest symbol shouldn't suppress the note.

```js
const computeStaffStemData = (pitches, midLineMidi) => {
    if (pitches.length === 0) return null;
    const lowestY = Math.max(...pitches.map(p => p.staffY));
    const highestY = Math.min(...pitches.map(p => p.staffY));
    const avgMidi = pitches.reduce((sum, p) => sum + p.midi, 0) / pitches.length;

    let stemDown = avgMidi >= midLineMidi;
    if (isPolyphonicMeasure) stemDown = ev.voice === 2;

    return { lowestY, highestY, stemDown };
};
```

V1 stems go up, V2 stems go down. Always. That's the convention. Getting it right took three separate commit cycles.

## 3. Building a Guitalele from Oscillators

Once the visual renderer was working, I needed audio. The parser could produce structured data. The viewer could display it. But I couldn't tell if a score was correct until I heard it.

The Web Audio API was my only option. No samples, no MIDI library, no external synth. Just me, a sine wave generator, and a dream of sounding like a nylon-string Guitalele.

The first version was terrible. A raw oscillator with a gain envelope. It sounded like a beep from a 1980s digital watch. Not a warm, woody Guitalele.

```
May 31, 18:53 - "Working with audio completed"   (129 lines changed)
May 31, 18:55 - "Much audible"
May 31, 19:14 - "Added percussion for note mute"
```

I added layers:

- **Waveshaping distortion** to simulate the acoustic saturation of nylon strings
- **A noise burst at attack** to recreate the finger-on-string transient (the critical "stab" that makes a pluck sound like a pluck, not a sine wave)
- **String stretch simulation** — a pitch bend of ~12 cents at the very start of each note, mimicking the string being displaced by a pick and snapping back
- **Body resonance** — a secondary oscillator with a low-frequency drone to simulate the soundbox
- **A dynamics compressor** to catch multi-note peaks and prevent clipping
- **Low-shelf and peaking filters** for warmth and woody character

```js
export const playHumanizedGuitaleleNote = (ctx, midiOrChain, startTime, duration, velocity = 1.0, noteVoice = 1) => {
    // Acquire a pre-allocated voice from the pool — zero node creation
    const voice = acquireVoice(ctx, startTime);

    // String stretch: pitch bends up ~12 cents at attack then settles
    voice.osc.frequency.setValueAtTime(initialFundamental * 1.012, startTime);
    voice.osc.frequency.exponentialRampToValueAtTime(initialFundamental, startTime + 0.025);

    // Finger-on-string attack noise
    const noiseSrc = ctx.createBufferSource();
    noiseSrc.buffer = getNoiseBuffer(ctx);
    // ...
};
```

For muted strings, I created a separate percussive thump — a triangle wave at 95 Hz mixed with a sawtooth at 1400 Hz through a bandpass filter. It sounds like someone tapping the strings with their finger. Not perfect, but recognizable.

V1 (melody) got a natural decay curve. V2 (bass) got a slow, drone-like decay — it rings through the measure and fades gently, exactly how a bass note behaves on a real Guitalele.

## 4. The Scheduler: Timing is Everything

The hardest part wasn't synthesis. It was synchronization.

The audio and the visual had to stay in lockstep. The highlighted note on the SVG must match the sound coming out of the speakers. If they drift apart by even 100ms, the whole thing feels broken.

I built a lookahead scheduler. It runs a `scheduleTimelineChunk` function at a fixed interval, checking which notes need to be played in the next `scheduleAheadTime` window. Audio events are dispatched to the Web Audio API timeline instantly (it handles timing from there). Visual events are queued into a `visualQueue` and consumed by a `requestAnimationFrame` loop.

```js
const scheduleTimelineChunk = () => {
    const targetHorizonTime = absoluteCurrentPlaybackTime + scheduleAheadTime;

    while (nextBeatIndexRef.current < timelineLength) {
        const beatSlice = currentTimelineBeatsRef.current[nextBeatIndexRef.current];
        const eventAbsoluteSec = (beatSlice.startBeat - offsetBeat) * beatDurationSeconds;

        if (eventAbsoluteSec >= targetHorizonTime) break;

        // 1. Dispatch audio to Web Audio timeline
        beatSlice.notes.forEach(note => {
            playHumanizedGuitaleleNote(ctx, runtimeSegments, finalPluckTime, ...);
        });

        // 2. Queue visual update
        queueVisualUpdate(visualAudioTime, beatSlice.globalIndex);
    }

    // Drift-compensated lookahead
    nextTickMs += lookaheadInterval;
    lookaheadTimerRef.current = setTimeout(
        scheduleTimelineChunk,
        Math.max(0, nextTickMs - performance.now())
    );
};
```

The drift compensation was critical. `setTimeout` is not precise — it can drift by 4-15ms per call. By tracking `performance.now()` and adjusting the next timeout delay, I kept the scheduler stable across minutes of playback.

## 5. The Great Refactoring

By June 7, the viewer was a single file of chaos. SVG rendering, audio scheduling, score layout, event handling — all dumped into `GuitaleleViewer.js`. It worked, but it was unmaintainable.

I spent a full day extracting modules:

```
"Take out constants"           → guitaleleViewerUtils.js
"Take out audio playback"      → audio.js
"Take out scheduler loop"      → audio.js (scheduler)
"Take out SVG Builder"         → svgUtils.js
"Take score builder out"       → scoreBuilder.js
```

Also realized I needed `useMemo`. Every time the component re-rendered, the entire SVG was rebuilt from scratch. That meant 900 lines of DOM diffing for every single state change. Slugging the browser.

"Use of the memoized version" — a single commit that wrapped the layout computation in `useMemo`. The performance went from "stuttering" to "smooth."

## 6. The Metronome and the Polish

By June 9, the core engine was solid. Time for features.

- **Metronome clicks** — a woodblock sound (triangle wave at 950 Hz for upbeats, 1400 Hz for downbeats) with a high-frequency snap layer. Count-in before playback starts.
- **Three view modes** — Tab only, Sheet music only, Both. Toggle during playback disabled for obvious reasons.
- **Auto-scroll** — When playback reaches a new row, the container scrolls to keep it in view. Tracks the previous row index to avoid re-scrolling on every beat.
- **Progress bar** — Beat-based progress calculation, not index-based (handles gaps and rests correctly).
- **Fullscreen mode** — Because watching a 200px score viewer is useless.
- **Description panel** — Click any note to see its pitch, string, fret, and any annotations.

```js
const activeDescription = useMemo(() => {
    // Renders V1/V2 columns with note names, fret numbers, and descriptions
    // Color-coded by voice, formatted for readability
}, [displayEvents]);
```

## 7. The Crackle War

This deserves its own section because it was the single hardest bug I have ever fixed in my career.

The audio worked. Mostly. But during sustained playback, crackles would appear. Random pops and clicks. Sometimes after 10 seconds, sometimes after a minute. Inconsistent, non-deterministic, impossible to reproduce reliably.

The initial audio engine created a new oscillator and gain node for every note, and disconnected them when done. That meant the garbage collector was running constantly, collecting hundreds of short-lived audio nodes per minute. Each GC pause caused a drop in the audio thread — a crackle.

I tried everything:
- "Fix many audio issues" — no
- "Minor improvement on sound profile" — no
- "Fixed some major issues" — no
- "Fix many things. Good for regular play" — getting closer

The fix came at 1:06 AM on June 24. A 212-line rewrite of audio.js.

```js
const VOICE_COUNT = 12;
let voicePool = null;

function acquireVoice(ctx, voiceStartTime) {
    if (!voicePool) {
        voicePool = [];
        for (let i = 0; i < VOICE_COUNT; i++) {
            // Pre-allocate oscillators, gains, filters — all of them
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            const filter = ctx.createBiquadFilter();
            osc.connect(gain);
            gain.connect(filter);
            filter.connect(masterTarget);
            osc.start();
            voicePool.push({ osc, gain, filter, releaseTime: 0 });
        }
    }
    // Find an available voice or steal the soonest-free
    let best = voicePool.find(v => v.releaseTime <= scheduleBase);
    if (!best) best = voicePool.reduce((a, b) => a.releaseTime < b.releaseTime ? a : b);
    return best;
}
```

**12 voices. Pre-allocated. Zero GC during playback.**

No new nodes are created per note. No old nodes are collected. The pool manages 12 pre-built oscillator-gain-filter chains, reuses them, resets their parameters, and never touches the garbage collector. The crackles vanished.

## The Inventory

Two weeks into the viewer, here's what the engine looked like:

| Module | Lines | Purpose |
|---|---|---|
| `GuitaleleViewer.js` | 903 | Main component, controls, UI, state management |
| `audio.js` | 873 | Audio synthesis, voice pool, scheduler, metronome |
| `svgUtils.js` | 984 | SVG rendering for tab + staff notation |
| `scoreBuilder.js` | 396 | Layout engine: measure positions, slot widths |
| `guitaleleViewerUtils.js` | 182 | Constants, pitch calculations, duration labels |

**Total: ~3,338 lines of audio-visual engine.**

And none of this was planned. The SVG started as five lines. The audio started as one oscillator. The scheduler was an afterthought. The voice pool was a panicked fix at 1 AM.

All because I wanted to hear if my Guitalele tab was correct.

The snowball is still rolling. There's more to the story — the audio timeline pre-compilation, the tied-note chain resolver, the responsive layout system, the performance optimizations. But those are for another day.
