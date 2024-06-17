# Harvard CS50

Teaches computational thinking and programming fundamentals, not a language.
Languages used include C, python, SQL, JavaScript, HTML, and CSS.

## Week 0: Scratch

Leaning computer science is like drinking from a fire hose. You're not going to
be comfortable. Get comfortable being uncomfortable.

Computational thinking. Think like a computer.

Base 2 (binary). Binary is 1 or 0. Binary digit = bit. Each transistor (bit) can
be on / off.

1 bit = 2^1 = 2 combinations (0-1)
2 bit = 2^2 = 4 comabinations (0-3)
3 bit = 2^3 = 8 combinations (0-7)
4 bit = 2^4 = 16 combinations (0-15)
5 bit = 2^5 = 32 combinations (0-31)
6 bit = 2^6 = 64 combinations (0-63)
7 bit = 2^7 == 128 combinations (0-127)
8 bit = 2^8 == 256 combinations (0-255)

byte = 8 bit

Decimal = base 10: 123 == 100 + 20 + 3

### Characters

How can we represent the letter `A`? Give each letter a number (character
encoding).

* ASCII: English letters. 0-255 is more than enough for english.

How do we represent emoji / foreign language characters?

Unicode. A superset of ASCII. A random character encoding for up to 4 bytes per
character. Up to 4B characters. Unicode's mission is to represent all characters
ever recorded in all language.

😂 (U+1F602: base 16 / unicode code point) is the most popular character (face
with tears of joy). They have different "fonts" U+1F602. How do we change the
skin tone or alter the unicode? To modify it, we use the same pattern, but modify it

👍🏻 = U+1F44D U+1F3FD (thumbs up + skin tone). Many emojis are joined together.

### Colors

RGB = (255 x 255 x 255)

Each pixel (picture element) is 24 bit(s).

### Audio / Video

Represent notes by a number. One number is the note, another is the duration, another is strength.

A video file is a series of pictures (motion pictures), played back at a speed.


### Algorithms

Code is the implementation of an algorithm. For example, binary searching is
faster than an iterative algorithm.

Computational complexity is important. As arguments grow toward infinity, how
does runtime or space requirements change? Expoential will slow down
exponentially, linear will grow linearlly with the inputs, logarithmic will grow
a small amount as the input grows (log(2)n requires only 1 step for each
doubling). An algorithm's speed and time requirements is called big-O notation.

### Pseudocode

Human readable version of an algorithm. Remember to look for edge cases when

### Artificial Intelligence

LLMs: trained on all human text. Predicts how to answer a question based on
patterns found in all human text.

### Preparing to Code

A lot of programming is setting up an environment, importing functions, libraries, editors, etc.

### Functions as units of abstraction

A character moves around on an X|Y plane and performs functions with parameters and return values.
Programming is about composing functions together to apply an algorithm to an input to produce an output.

A function is an abstraction. Abstraction allows you to reduce complexity by
grouping operations into a larger, reusable unit. You don't need to know how the
internals work, you only need to know its inputs and outputs.

Build programs by composing functions into increasingly complex programs.

## Week 1: C

Evaluating code:

* Correctness. Does it run as intended?
* Design. How well is the code designed?
* Style. How asthetically pleasing and consistent is the code?

Abstraction: the art of simplifying code such that it deals with smaller and
smaller problems.

Types: the possible data stored in a variable.

Variables: a name or value that can change. (A pointer to memory).
