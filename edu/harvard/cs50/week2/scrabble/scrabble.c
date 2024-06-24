#include <cs50.h>
#include <ctype.h>
#include <stdio.h>
#include <string.h>

int score(char c)
{
    static const int scores[] = {1, 3, 3, 2, 1, 4, 2, 4, 1, 8, 5, 1, 3,
                                 1, 1, 3, 10, 1, 1, 1, 1, 4, 4, 8, 4, 10};
    if (!isalpha(c))
    {
        return 0;
    }
    char upper = toupper(c);
    return scores[upper - 'A'];
}

int scrabble(char *word)
{
    int len = strlen(word);
    int total = 0;
    for (int i = 0; i < len; i++)
    {
        total += score(word[i]);
    }
    return total;
}

int main(int argc, char *argv[])
{
    char *p1 = get_string("Player 1: ");
    char *p2 = get_string("Player 2: ");

    int p1score = scrabble(p1);
    int p2score = scrabble(p2);

    if (p1score > p2score)
    {
        printf("Player 1 wins!\n");
    }
    else if (p1score < p2score)
    {
        printf("Player 2 wins!\n");
    }
    else
    {
        printf("Tie!\n");
    }
}
