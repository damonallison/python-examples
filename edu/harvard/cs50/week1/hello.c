#include <stdio.h>
#include "cs50.h"

// int main(void)
// {
//     string s = get_string("what's your name? ");
//     printf("hello, %s\n", s);
// }

// int main(void)
// {
//     char c = get_char("do you agree? ");

//     // strings use "", chars use ''
//     if (c == 'Y' || c == 'y')
//     {
//         printf("agreed\n");
//     }
//     else if (c == 'N' || c == 'n')
//     {
//         printf("not agreed\n");
//     }
//     else
//     {
//         printf("unknown\n");
//     }
// }

void meow(int n)
{
    for (int i = 0; i < n; i++)
    {
        printf("meow\n");
    }
}

void mario(int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = n; j < n; j++)
        {
        }
    }
}
int change(int n)
{
    int coins = 0;
    while (n > 0)
    {
        if (n >= 25)
        {
            n -= 25;
        }
        else if (n >= 10)
        {
            n -= 10;
        }
        else if (n >= 5)
        {
            n -= 5;
        }
        else
        {
            n--;
        }
        coins++;
    }
    return coins;
}

int main(void)
{
    // meow(3);
    int coins = change(80);
    printf("you need %d coins", coins);
}