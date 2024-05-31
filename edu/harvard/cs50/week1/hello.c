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

int main(void)
{
    meow(3);
}