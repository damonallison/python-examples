// Modifies the volume of an audio file

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// Number of bytes in .wav header
const int HEADER_SIZE = 44;

int main(int argc, char *argv[])
{
    // for (int i = 0; i < argc; i++)
    // {
    //     printf("argv[%d] == %s\n", i, argv[i]);
    // }

    // Check command-line arguments
    if (argc != 4)
    {
        printf("Usage: ./volume input.wav output.wav factor\n");
        return 1;
    }

    // Open files and determine scaling factor
    FILE *input = fopen(argv[1], "r");
    if (input == NULL)
    {
        printf("Could not open file.\n");
        return 1;
    }

    FILE *output = fopen(argv[2], "w");
    if (output == NULL)
    {
        printf("Could not open file.\n");
        return 1;
    }

    float factor = atof(argv[3]);

    // Copy header from input file to output file
    char buffer[HEADER_SIZE];
    size_t bytesRead = fread(buffer, sizeof(char), HEADER_SIZE, input);

    if (bytesRead < HEADER_SIZE && !feof(input))
    {
        perror("Error reading input file");
        fclose(input);
        fclose(output);
        return EXIT_FAILURE;
    }
    if (fwrite(buffer, sizeof(char), HEADER_SIZE, output) != HEADER_SIZE)
    {
        perror("Error writing to output file");
        fclose(input);
        fclose(output);
        return EXIT_FAILURE;
    }
    int16_t sample;
    while (fread(&sample, sizeof(int16_t), 1, input) == 1)
    {
        if (ferror(input))
        {
            perror("error reading file");
            fclose(input);
            fclose(output);
            return EXIT_FAILURE;
        }
        int16_t adjusted = (int16_t)(sample * factor);
        // printf("%d\n", adjusted);

        if (fwrite(&adjusted, sizeof(int16_t), 1, output) != 1)
        {
            if (ferror(output))
            {
                perror("error writing file");
                fclose(input);
                fclose(output);
                return EXIT_FAILURE;
            }
        }
    }
    if (fclose(input) != 0)
    {
        perror("error closing input file");
    }
    if (fclose(output) != 0)
    {
        perror("error closing output file");
    }
    return 0;
}