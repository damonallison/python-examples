#include <cs50.h>
#include <stdio.h>
#include <string.h>

typedef struct
{
    string name;
    string number;
} person;

int main(void)
{
    person people[] = {
        {
            "Damon",
            "612-123-4567",
        },
        {
            "Sam",
            "763-987-6543",
        }};

    string name = get_string("Name: ");
    int size = sizeof(people) / sizeof(people[0]);

    for (int i = 0; i < size; i++)
    {
        if (strcmp(people[i].name, name) == 0)
        {
            printf("found %s\n", people[i].number);
            return 0;
        }
    }
    printf("not found\n");
    return 1;
}
