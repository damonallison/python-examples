#include <stdlib.h>
#include <stdio.h>

typedef struct Node
{
    int number;
    struct Node *prev;
    struct Node *next;
} Node;

void append(Node *head, Node *val)
{
    Node *tmp = head;
    while (tmp->next != NULL)
    {
        tmp = tmp->next;
    }
    tmp->next = val;
    val->prev = tmp;
}

int main()
{
    // int x = 4;
    // int *y = NULL;
    // int *z = NULL;

    // printf("y(addr) = %p\n", y);
    // printf("z(addr) = %p\n", z);

    // y = &x;
    // z = &x;

    // printf("x = %d\n", x);
    // printf("y(addr) = %p\n", y);
    // printf("x(addr) == %p\n", &x);

    // printf("y = %d\n", (*y));
    // printf("z = %d\n", (*z));

    // x = 10;
    // printf("x = %d\n", x);
    // printf("y(addr) = %p\n", y);
    // printf("x(addr) == %p\n", &x);

    // printf("y = %d\n", (*y));
    // printf("z = %d\n", (*z));

    Node *head = NULL;

    for (int i = 0; i < 10; i++)
    {
        Node *n = malloc(sizeof(Node));
        n->prev = NULL;
        n->number = i;
        n->next = NULL;

        if (head == NULL)
        {
            head = n;
        }
        else
        {
            append(head, n);
        }
    }

    Node *tmp = head;
    while (tmp != NULL)
    {
        printf("%d\n", tmp->number);
        tmp = tmp->next;
    }
    return 0;
}