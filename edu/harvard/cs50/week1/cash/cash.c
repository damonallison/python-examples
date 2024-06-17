#include <cs50.h>
#include <stdio.h>

// Return the least amount of coins which total `n`
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
    int owed = 0;
    do
    {
        owed = get_int("Changed owed: ");
    }
    while (owed <= 0);

    int coins = change(owed);
    printf("%d\n", coins);
}
