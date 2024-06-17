// Be sure to use -lcs50 when compiling.
//
// 	clang *.c -lcs50 -o mario
#include <cs50.h>
#include <stdio.h>

int main(void)
{
    int height = 0;
    do
    {
        height = get_int("how high? ");
    } while (height < 1 || height > 8);

    int total_size = (height * 2) + 2;
    for (int i = 1; i <= height; i++)
    {
        for (int j = 1; j <= height; j++)
        {
            if (j > height - i)
            {
                printf("#");
            }
            else
            {
                printf(" ");
            }
        }
        printf("  ");

        for (int j = 1; j <= i; j++)
        {
            printf("#");
        }
        printf("\n");
    }
}
