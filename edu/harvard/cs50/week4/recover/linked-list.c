#include <stdio.h>
#include <stdlib.h>

// Define the structure for a doubly linked list node
struct Node
{
    int data;
    struct Node *prev;
    struct Node *next;
};

// Function to create a new node with given data
struct Node *createNode(int data)
{
    struct Node *newNode = (struct Node *)malloc(sizeof(struct Node));
    newNode->data = data;
    newNode->prev = NULL;
    newNode->next = NULL;
    return newNode;
}

// Function to insert a new node at the beginning
void insertAtBeginning(struct Node **head, int data)
{
    struct Node *newNode = createNode(data);
    if (*head != NULL)
    {
        (*head)->prev = newNode;
        newNode->next = *head;
    }
    *head = newNode;
}

// Function to insert a new node at the end
void insertAtEnd(struct Node **head, int data)
{
    struct Node *newNode = createNode(data);
    if (*head == NULL)
    {
        *head = newNode;
        return;
    }
    struct Node *temp = *head;
    while (temp->next != NULL)
    {
        temp = temp->next;
    }
    temp->next = newNode;
    newNode->prev = temp;
}

// Function to delete a node with a given key
void deleteNode(struct Node **head, int key)
{
    struct Node *temp = *head;

    // Search for the node to be deleted
    while (temp != NULL && temp->data != key)
    {
        temp = temp->next;
    }

    // If the node was not present in the list
    if (temp == NULL)
        return;

    // If the node to be deleted is the head node
    if (temp == *head)
    {
        *head = temp->next;
    }

    // Change next only if the node to be deleted is NOT the last node
    if (temp->next != NULL)
    {
        temp->next->prev = temp->prev;
    }

    // Change prev only if the node to be deleted is NOT the first node
    if (temp->prev != NULL)
    {
        temp->prev->next = temp->next;
    }

    // Free the memory occupied by temp
    free(temp);
}

// Function to print the doubly linked list from the beginning to the end
void printList(struct Node *head)
{
    struct Node *temp = head;
    while (temp != NULL)
    {
        printf("%d -> ", temp->data);
        temp = temp->next;
    }
    printf("NULL\n");
}

int main()
{
    struct Node *head = NULL;

    insertAtEnd(&head, 1);
    insertAtEnd(&head, 2);
    insertAtEnd(&head, 3);
    insertAtBeginning(&head, 0);

    printf("Doubly linked list: ");
    printList(head);

    deleteNode(&head, 2);
    printf("Doubly linked list after deletion of 2: ");
    printList(head);

    return 0;
}