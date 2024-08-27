#include <cs50.h>
#include <ctype.h> // is*
#include <stdio.h>
#include <stdlib.h>
#include <string.h> // str*

bool only_digits(string s)
{
    int len = strlen(s);
    for (int i = 0; i < len; i++)
    {
        if (!isdigit(s[i]))
        {
            return false;
        }
    }
    return true;
}

char rotate(char c, int key)
{
    if (!isalpha(c))
    {
        return c;
    }

    // subtract c from the start of the alphabet (preserving case) to start 'A' and 'a' at 0.
    //
    // this allows us to apply modulus and wrap back to the start of the alphabet.
    //
    // if we simply did: ((c + key) % 26) we'd end up with a non-ascii character.
    char sub;
    if (isupper(c))
    {
        sub = 'A';
    }
    else
    {
        sub = 'a';
    }

    char to_rotate = c - sub;

    int rotated = (to_rotate + key) % 26;

    return (char)rotated + sub;
}

int main(int argc, char *argv[])
{
    // for (int i = 0; i < argc; i++)
    // {
    //     printf("arg %d == %s", i, argv[i]);
    // }

    if (argc != 2)
    {
        printf("Usage: ./caesar KEY\n");
        return 1;
    }

    if (!only_digits(argv[1]))
    {
        printf("Usage: ./caesar KEY\n");
        return 1;
    }

    int key = atoi(argv[1]);
    string plaintext = get_string("plaintext:  ");
    int len = strlen(plaintext);

    printf("ciphertext:  ");
    for (int i = 0; i < len; i++)
    {
        printf("%c", rotate(plaintext[i], key));
    }
    printf("\n");
}