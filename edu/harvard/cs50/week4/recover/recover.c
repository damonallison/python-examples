#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        printf("Usage: ./recover FILE\n");
        return 1;
    }

    // Open file in read mode
    FILE *diskImg = fopen(argv[1], "r");

    // FILE *diskImg = fopen("card.raw", "r");
    if (diskImg == NULL)
    {
        printf("Cannot open file\n");
        return 1;
    }

    // Create a buffer
    unsigned char buffer[512];

    // Create a counter for the # of the JPG file
    int imgCount = 0;

    // Declaring img
    FILE *img = NULL;

    // File name
    char filename[8];

    // While there are sill JPGs to read, continue reading thru the file
    while (fread(buffer, 512, 1, diskImg) == 1)
    {
        // If there are four bytes in a row that resemble a JPG, then...
        if (buffer[0] == 0xff && buffer[1] == 0xd8 && buffer[2] == 0xff && (buffer[3] & 0xe0) == 0xe0)
        {
            if (img != NULL)
            {
                fclose(img);
            }
            sprintf(filename, "%03i.jpg", imgCount);
            img = fopen(filename, "w");
            if (img == NULL)
            {
                printf("Cannot open file\n");
                fclose(diskImg);
                return 1;
            }
            ++imgCount;
        }
        if (img != NULL)
        {
            fwrite(buffer, 512, 1, img);
        }
    }
    if (img != NULL)
    {
        fclose(img);
    }
    fclose(diskImg);
}
