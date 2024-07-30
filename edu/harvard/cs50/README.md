# Harvard CS50

Teaches computational thinking and programming fundamentals, not a language.
Languages used include C, python, SQL, JavaScript, HTML, and CSS.

## Codespaces

1. Link github account: https://submit.cs50.io/
2. Login to cs50.dev: https://cs50.dev/
3. Run `update50`

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

## Week 3: Algorithms

Learn how to think algorithmically.

### Complexity

Time / complexity (big O notation)

* O(1) = Constant
* O(log n) = Logarithmic (binary == log(2)n)
* O(n) = Linear
* O(n log n) = Logarithmic (merge sort)
* O(n^2) = Quadratic

Big O == upper bound
Theta == lower bound

### Linear Search

Linear time. Iterate the list until the number is found.

```c

#include <stdio.h>
#include <string.h>

int find(int lst[], int len, int goal) {
    for (int i = 0; i < len; i++) {
        // use strcmp(s1, s2) == 0 for string equality
        // asciibetical comparison (ascii character comparison)
        if lst[i] == goal {
            return 1;
        }
    }
    return -1;
}

int numbers[] = {1, 2, 3, 4, 5};
int len = sizeof(numbers) / sizeof(numbers[0]);
printf("found == %s", find(numbers, len, 4) ? "true" : "false");
```

### Binary Search

log(2)n - Cutting the input in 1/2 with every iteration.

With sorted input.

* Repeat until found.
     * Divide the input in half, looking at the middle number.
     * If we've found what we are looking for, we are done.
     * If the middle number is greater than what we are looking for, repeat with
       the left half. Otherwise, repeat with the right half.

### Selection Sort

Selecting the lowest element from the list with each iteration. Very slow, but
low space requirements since only a single extra element needs to be stored with
each iteration.

O(n^2) == n(n-1)/2

For i from o to n-1:  (n)
    * Iterate all numbers from i, remembering the lowest.  (n)
    * Swap the lowest into i

### Bubble Sort

Continuing to swap elements, putting the lowest on the left (ascending) or right
(descending).

The slowest on random data as it's n^2 plus a lot of swaps.

O(n^2) == (n - 1)*(n - 1)

Repeat n times
    for i from 0 to n-2
        if numbers[i] and numbers[i + 1] out of order
            swap them
    if no swaps
        quit

### Recursion

A function that calls itself with a smaller input and ultimately hitting a base
case.

```c
// Recursion
void draw(int n) {
    // base case
    if (n <= 0) {
        return;
    }
    draw(n - 1)
    for (int i = 0; i < n; i++>) {
        print("#");
    }
    print("\n");
}
```

### Merge Sort

O(n log n)


if only one number
    quit
else
    sort left half
    sort right half
    merge halves
        for i = 0 to n - 1
            comapre left and right at n, take lowest, then other

```c
// n log n
// log n height, n width ==
int mergesort(int[] numbers) {

}
```

---

## Week 4: Memory

### Hexidecimal

RGB contains 3 255 (2^8 - 1) hex (base 16) values to represent color.

```
Base 16
----------------

16        1
F         F
16 * 15   1 * 15

240 + 15 == 255
```

Hex is convenient becausse it requires 4 bits at a time. 8 bits == byte.

Hex is represented as `0xFF`. The `0x` is a convention that says: the following
number is hexidecimal.
* The maximum value of a bit == `1`
* The maximum value of 4 bits == `0xF` == `15`
* The maximum value of a byte (8 bits) == `0xFF` == `255`
* The maximum value of a 4 byte integer == `0xFFFFFFFF`
    * 16^7 + 16^6 + 16^5 + 16^4 + 16^3 + 16^2 + 16^1 + 16^0 = `4294967295`


A pointer is a memory address.

A 32 bit OS and chip architecture can only access 3^32 (2 GB) memory addresses.

* 2 GB == 32 bit (4 byte)
* 16 exabytes == 64 bit (8 bytes)

Pointer arithmetic: doing math on addresses.

`NULL` is the address 0.

### Stack and Heap

When calling functions, each function call adds a frame to the call stack. Each
frame of the stack includes copies of argument values which are *not* shared
between frames. Thus, C is a `copy by value` language.

The heap is a sharable memory space which you obtain pointers to by allocating
memory using `malloc`.

The stack and heap can overflow (buffer overflow).


## Week 5: Data Structures

* Queue: FIFO / enqueue / dequeue
* Stack: LIFO / push / pop
* Linked list: non-contiguous memory

## Week 6: Python

* All languages have the same basic building blocks: variables, functions,
  iteration, errors, abstractions.
* High level, intrepreted languages like Python are easier but slower. Start w/
  high level languages, use lower languages (Rust or Go) where you need to.
* Objects: data and functions


## Week 6.5: AI

What is ethical use of AI assuming AI is better than humans?

System prompt "engineering": providing context or rules to "tune" the AI model
to respond with particular intention or limits.

How does AI change the job market? How is AI the same or different than other
transformative technologies like the printing press, farm equipment, or
internet?

Without ego, without judgement.

Why did ChatGPT break the AI industry into the forefront of technology?

* Generative
* Universal

minimax: maximizing or minimizing probability of a favorable outcome.

### Machine learning
Machines learn without being explicity programmed.

#### Reinforcement learning
Reward for success / punish for failure. Reinforces success. Requires labeled data.
* Example: robot learning

Explore vs. exploit
* Expliot the knowledge you know
* But explore a % time (randomness) to improve

#### Deep learning

Neuron based network designed after the human mind.

#### Generative AI

Attention: the relationship between words determines their probability of words
to group together. Transformer architecture. The "T" in ChatGPT. It takes
context into account.

> Attention is all you need - Seminal Google paper.

#### Hallucinations

When a chatbot confidently gives a plausible but incorrect answer. Will
obviously become less frequent as models improve.


## Week 7: SQL

Indexes improve speed at the expense of higher storage cost.
