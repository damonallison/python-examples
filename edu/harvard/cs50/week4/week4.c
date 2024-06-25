#include <cs50.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>

typedef uint8_t BYTE;

// How do floats work?
// how do you find the address of the location?

bool compare(char *s, char *t)
{
    // address comparison
    if (s == t)
    {
        return true;
    }
    return strcmp(s, t) == 0;
}

void comparison()
{
    char *ss = "damon";
    char *st = "damon";

    printf("identity equal == %i\n", compare(ss, ss));
    printf("value equal == %i\n", compare(ss, st));
    // strcmp: returns the mangnitue of the first value compared to the second
    // value.
    //
    // If the first value is less than the second, strcmp returns negative.
    //
    // If the first value is greater than the second, strcmp returns positive.
    printf("damon1 < damon2 == %i\n", strcmp("damon1", "damon2") < 0);
    printf("damon2 > damon1 == %i\n", strcmp("damon2", "damon1") > 0);
}

int copy()
{
    char *s = "damon";

    char *t = malloc(strlen(s) + 1); // includes the \0 char
    if (t == NULL)
    {
        // unable to allocate memory
        return 1;
    }

    for (int i = 0; i <= strlen(s); i++) // includes the \9 char
    {
        t[i] = s[i];
    }

    //
    // C is not type safe, so we can run over arrays (buffer overflow).
    //
    // This will cause a segmentation fault - accessing memory you don't own.
    //
    // t[100] = 'a';

    printf("equal == %i\n", compare(s, t));
    free(t);

    printf("%p\n", NULL);
    return 0;
}

int get()
{
    // getting an integer from the terminal
    int n;
    printf("n: \n");
    scanf("%i", &n);
    printf("n: %i\n", n);

    // Getting a string from the terminal is tricky!
    //
    // You have to dynamically allocate bytes as the string grows.
    //
    // char *s;
    // printf("s: \n");
    // scanf("%i", &n);
    // printf("n: %i\n", n);

    return 0;
}

int fileio()
{
    FILE *file = fopen("file.txt", "a");
    if (file == NULL)
    {
        return 1;
    }

    char *name = get_string("Name: ");
    char *number = get_string("Number: ");

    fprintf(file, "%s,%s\n", name, number);
    fclose(file);
    return 0;
}

int filecopy(char *src, char *dest)
{
    FILE *fsrc = fopen(src, "rb");
    FILE *fdst = fopen(dest, "wb");

    BYTE b;

    // Manually copy a file byte by byte. Note you'd typically copy in chunks.
    while (fread(&b, sizeof(b), 1, fsrc) != 0)
    {
        fwrite(&b, sizeof(b), 1, fdst);
    }

    fclose(fdst);
    fclose(fsrc);
    return 0;
}

int main(void)
{
    int x;
    long y;
    unsigned int xx;
    unsigned long yy;

    printf("sizeof(int) == %lu\n", sizeof(x));
    printf("sizeof(long) == %lu\n", sizeof(y));
    printf("sizeof(unisnged int) == %lu\n", sizeof(xx));
    printf("sizeof(unsigned long) == %lu\n", sizeof(yy));

    float f;
    double g;
    printf("sizeof(float) == %lu\n", sizeof(f));
    printf("sizeof(double) == %lu\n", sizeof(g));

    // Pointers
    // & == address of
    // * == value of (pointer dereference)
    int *xp = &x;

    // %p will format the address in a hex value
    printf("sizeof(int *)%lu\n", sizeof(xp));
    printf("address of x == %p\n", xp);

    // Strings
    char c = 'c';
    printf("sizeof(char)%lu\n", sizeof(c));

    char *s = "damon"; // 6 bytes (including \0)
    printf("sizeof(char *)%lu\n", sizeof(s));
    printf("strlen(s) == %lu\n", strlen(s));
    printf("address of s == %p\n", s);
    printf("address of s[0] == %p\n", &s[0]);
    printf("address of s[1] == %p\n", &s[1]);

    // pointer arithmetic
    for (int i = 0; i < strlen(s); i++)
    {
        // advancing the pointer one byte at a time
        printf("%c", *(s + i));
    }
    printf("\n");

    comparison();
    copy();
    filecopy("file.txt", "file2.txt");
}