// Implements a dictionary's functionality

#include <assert.h>
#include <ctype.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#include "dictionary.h"

// Represents a node in a hash table
typedef struct node
{
    char word[LENGTH + 1];
    struct node *next;
} node;

const unsigned int N = 1000;

// Hash table
node *table[N];

// Total size, set after load
int tableSize = 0;

// Returns true if word is in dictionary, else false
bool check(const char *word)
{
    int h = hash(word);
    node *n = table[h];

    while (n != NULL)
    {
        if (strcasecmp(word, n->word) == 0)
        {
            return true;
        }
        n = n->next;
    }
    return false;
}

// Hashes word to a number
unsigned int hash(const char *word)
{
    // sum of ascii values
    int sum = 0;
    int curr = 0;
    while (word[curr] != '\0')
    {
        sum += tolower(word[curr]);
        curr++;
    }
    return sum % 1000;
}

void append(node *head, node *n)
{
    assert(head != NULL);

    node *tmp = head;
    while (tmp->next != NULL)
    {
        tmp = tmp->next;
    }
    tmp->next = n;
}

int nodesize(node *n)
{
    if (n == NULL)
    {
        return 0;
    }
    if (n->next == NULL)
    {
        return 1;
    }
    return 1 + nodesize(n->next);
}

void freenode(node *n)
{
    if (n == NULL)
    {
        return;
    }

    if (n->next != NULL)
    {
        freenode(n->next);
    }
    free(n);
}

// Loads dictionary into memory, returning true if successful, else false
bool load(const char *dictionary)
{
    FILE *source = fopen(dictionary, "r");
    if (source == NULL)
    {
        printf("Could not open %s.\n", dictionary);
        return false;
    }

    // read word / create node
    char buffer[LENGTH + 2];

    int count = 0;
    while (fgets(buffer, sizeof(buffer), source) != NULL)
    {
        if (ferror(source))
        {
            printf("Error reading file");
            return false;
        }

        char *pos;
        if ((pos = strchr(buffer, '\n')) != NULL)
        {
            *pos = '\0';
        }

        node *n = malloc(sizeof(node));
        strcpy(n->word, buffer);
        n->next = NULL;

        int h = hash(n->word);
        // printf("word == %s - hash == %d\n", n->word, h);
        if (table[h] == NULL)
        {
            table[h] = n;
        }
        else
        {
            append(table[h], n);
        }
        count++;
    }

    tableSize = count;
    fclose(source);
    return true;
}

// Returns number of words in dictionary if loaded, else 0 if not yet loaded
unsigned int size(void)
{
    return tableSize;
}

// Unloads dictionary from memory, returning true if successful, else false
bool unload(void)
{
    for (int i = 0; i < N; i++)
    {
        freenode(table[i]);
    }
    return true;
}
