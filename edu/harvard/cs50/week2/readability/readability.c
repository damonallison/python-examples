#include <cs50.h>
#include <ctype.h>
#include <stdio.h>
#include <string.h>

// Coleman-Liau index - a formula for determining grade level
//
// index = 0.0588 * L - 0.296 * S - 15.8
//
// L = average number of letters per 100 words in the text.
// S = average number of sentences per 100 words of text.

typedef struct
{
    int letters;
    int words;
    int sentences;
} WordCounts;

WordCounts count(char *text)
{
    int sentcount = 0;
    int lettercount = 0;
    int wordcount = 0;
    int len = strlen(text);
    for (int i = 0; i < len; i++)
    {
        char cur = text[i];
        if ((cur == '.' || cur == '?' || cur == '!') &&
            (isspace(text[i + 1])))
        {
            sentcount++;
        }
        else if (isspace(cur))
        {
            wordcount++;
        }
        else
        {
            lettercount++;
        }
    }

    // complete hack - add one for the last word / sentence
    sentcount++;
    wordcount++;

    WordCounts result;
    result.letters = lettercount;
    result.sentences = sentcount;
    result.words = wordcount;
    return result;
}

int main(int argc, char *argv[])
{
    char *sent = get_string("Text: ");
    WordCounts counts = count(sent);

    float S = ((float)counts.sentences / (float)counts.words) * 100.0;
    float L = ((float)counts.letters / (float)counts.words) * 100.0;
    int index = (int)((0.0588 * L) - (0.296 * S) - 15.8);

    // printf("L = %f S = %f index=%i\n", L, S, index);

    if (index < 1)
    {
        printf("Before Grade 1\n");
    }
    else if (index >= 16)
    {
        printf("Grade 16+\n");
    }
    else
    {
        printf("Grade %i\n", index);
    }
}