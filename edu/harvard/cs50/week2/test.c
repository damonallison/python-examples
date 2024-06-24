#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

char *upstr(const char *str)
{
    size_t len = strlen(str);
    char *newstr = (char *)malloc(len);
    for (int i = 0; i < len; i++)
    {
        newstr[i] = toupper(str[i]);
    }
    return newstr;
}

int main(int argc, char *argv[])
{
    printf("Arguments: %i\n", argc);
    for (int i = 0; i < argc; i++)
    {
        printf("Arg: %i = %s\n", i, argv[i]);
    }

    printf("Hello ");
    if (argc > 1)
    {
        char *upcase = upstr(argv[1]);
        printf("%s", upcase);
        free(upcase);
    }
    else
    {
        printf("world");
    }
    printf("\n");

    return 0;

    // char *s = "hello ";
    // for (int i = 0; i < strlen(s); i++)
    // {
    //     if (islower(s[i]))
    //     {
    //         printf("%c", toupper(s[i]));
    //     }
    //     else
    //     {
    //         printf("%c", s[i]);
    //     }
    // }
    // printf("\n");
}