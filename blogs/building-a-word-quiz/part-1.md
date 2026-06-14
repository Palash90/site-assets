# Building a Real-Time Voice-Driven Word Quiz from Scratch

It's satisfying to see your child use your work to improve their practice sessions. On a fine Saturday evening, my younger kid started using my simple math-quiz application ([Math Study](http://palashkantikundu.in/component/study)). She was completely hooked on collecting the yellow stars the app throws your way when you get an answer right, and yanks one away when you mess up.

Dopamine at play, she tirelessly ground out simple addition problems until she hit 100 stars. And yes, to honor the scoreboard, I had to promise her two Cadbury Gems packets.

Watching her happy and locked into that simple feedback loop kicked off another idea. If that gamified engine could motivate her to solve math problems just to watch a numeric vector go up, why not use the exact same psychological trigger to help her learn alphabets and words?

Next morning, I rolled up my sleeves and underestimated the scope of the project. I learnt it the hard way...

My idea was simple, I will simply add a new array of all the words and roll up a textbox and integrate the rating system.

Oh boy...

I was completely wrong. It opened up another rabbit hole.

## Phase 1: Disintegration

I spent the first hour tearing apart the monolithic state coupling. The scoring panel and individual score vectors were technically separate UI components, but they were deeply intertwined with the arithmetic core. They relied on the same active handlers and state dispatches that calculated mathematical equations.

I tore out the system states and started lifting states.

## Phase 2: Designing the New Quiz

Once the states were completely disintegrated, I laid out the design of the word quiz component. 

It has to be a collection of simple tasks:

1. Have a set of checkboxes for each language (almost copy paste from the Math App)
   ![Word Study Design Layout](${word-quiz-study-layout})
2. Collect alphabets and words from a static context (similar to how I drove the equation handling)
3. Show them on UI
4. Add a textbox for collecting answer and submit...
   ![Word Quiz Flawed Design](${word-quiz-flawed-design})

Hold on, I simply can't use a textbox there...

A textbox that essentially takes input of another provided text with zero processing. Completely useless.

The kid has to do something for the input. The missing link was, the kid had to recognize the given letter or word. That's the processing step I completely overlooked.

## Phase 3: Nanah Sipahi at Rescue

There comes the power of feedback from actual users. I routed my dilemma to my elder daughter and asked what would be a good way to handle the situation. Well, a very simple thought occured to her which I completely failed to recognize - "Why don't you simply ask her to read the word?"

It was simple enough to suggest. But that was an uncharted terrain for me. I took some time to do my due research and redesigned the app to take voice input instead and now the system looks like the following:
![Word Quiz Current Design](${word-quiz-current-design})

That's how I solved my first problem.

## Phase 4: Implementation

The idea of voice input was simple in theory but when I started looking at it from a practical angle, there were a lot more to it.

With some research and Gemini consultation, it came to my notice that browsers can do Speech Recognition out of box. It's just a matter of calling the `window` api.

Happy with the knowledge, I threw up some code at the browser:

```javascript
  useEffect(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

    if (!SpeechRecognition) {
      setError('Speech Recognition not supported in this browser');
      return;
    }

    recognitionRef.current = new SpeechRecognition();

    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.abort();
      }
    };
  }, []);
```

To my surprise, I got to know Firefox does not support it. Thinking about the scale and scope of my application (primary user is my kid), I completely ignored it. As I am aware that the devices currently at home can all have either Chrome or Edge.

## Phase 5: The Unignorable Problem and the Pivot

I ignored the first problem but could not escape the next one. The Speech Recognition api can't deal with single syllable voices. So, `a`, `b`, `क`, `ম`, `ಅ` are pretty difficult to manage. I read on internet about handling these stuffs with some tips and tricks.

I tried a few, but none worked reliably. So, I completely resorted to another solution. Instead of trying to recognize a letter, what if I try to recognize words.

And so I did.

## Phase 6: The Data Overload

I was happy I could actually use some words for the app to work with. I tested for all the supported languages - Bengali, Hindi, Kannada and English. Except for a few, the system started working pretty smooth.

Over excited me populated a list of 1000 common words for each language and re-ran the application.

Nothing broke, everything worked fine. The app loads, the recognitions work smoothly and everything.

Then I opened the network tab in the Inspect window. I saw megabytes of data transfer.

If you are actually following my articles, you might already know that I am using `github pages` for hosting my static website and there is absolutely no server running for backend. Github although very generous in its usage limits for free tier, I still felt I should stop this data transfer. Data transfer rate was definitely one reason. The other reasons are classic ‘me’.:

1. I wanted to use lazy loading
2. I was a little afraid of slowing down my website as it already handles a very big script file. If interested, [read here](https://palashkantikundu.in/content/setting-up-a-site-3)

So, I used lazy loading of components:

```javascript
const StudyApp = lazy(() => import("./study/Study"));
```

A single line now saves every page load from transferring megabytes of data.

## Phase 7: The Launch

After I was satisfied with all my technical dopamine, it was time to publish the website to the actual users (UAT Mode, Beta Testing). I repeated my cycle of manual testing for each component I touched and some sanity testing on others too.

I called my sipahis again to give the app a thorough testing.

The moment they saw it, they had this question, "But how would I know which letter is this???"

I received the feedback, "The app is only useful if I already know the alphabet, but the first step is missing".

>> No matter how hard you try, a developer can only be a tester to an extent.
>> No matter how hard you try, a tester can only be a user to an extent
>> No matter how hard you try, you will always have scope for improvement

## Conclusion

A direct UAT bug on UX of my app, I had to solve it. I again started laying out design and again started coding. The work is in progress. Story for another day.

If you are still with me, thank you for reading till the end. Here is the source code for you: [](https://github.com/Palash90/site/tree/main/src/components/study)
