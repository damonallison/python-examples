def grade_level(text: str) -> tuple[int, int, int]:
    """Computes the estimated grade level of `text`. using the Coleman-Liau
    index.

    grade level = 0.0588 * L - 0.296 * S - 15.8

    * L = average number of letters per 100 words
    * S = average number of sentences per 100 words

    Your program should count the number of letters, words, and sentences in the
    text.

    * A letter is any upper case or lower case.
    * Any sequence of characters separated by spaces should count as a word.
    * Any occurrence of a period, exclamation point, or question mark indicates
      the end of a sentence.
    """

    letters = 0
    words = 0
    sentences = 0

    for char in text:
        if char in [".", "?", "!"]:
            sentences += 1
        elif char.isspace():
            words += 1
        elif char.isalpha():
            letters += 1

    # hack: add one for the last word
    words += 1
    return (letters, words, sentences)


if __name__ == "__main__":
    letters, words, sentences = grade_level(input("Text: "))

    S = (sentences / words) * 100
    L = (letters / words) * 100

    cl_index = round((0.0588 * L) - (0.296 * S) - 15.8)

    if cl_index < 1:
        print("Before Grade 1")
    elif cl_index >= 16:
        print("Grade 16+")
    else:
        print(f"Grade {cl_index}")
