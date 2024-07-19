#include <assert.h>
#include "helpers.h"
#include <math.h>

BYTE tobyte(double d)
{
    assert(d >= 0);
    return (BYTE)round(fmin(d, 255));
}

int mini(int x, int y)
{
    return (x < y) ? x : y;
}

int maxi(int x, int y)
{
    return (x > y) ? x : y;
}

// Convert image to grayscale
void grayscale(int height, int width, RGBTRIPLE image[height][width])
{
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            RGBTRIPLE curr = image[i][j];
            BYTE avg = tobyte((curr.rgbtBlue + curr.rgbtGreen + curr.rgbtRed) / 3.0);
            RGBTRIPLE gray;
            gray.rgbtBlue = avg;
            gray.rgbtGreen = avg;
            gray.rgbtRed = avg;
            image[i][j] = gray;
        }
    }
    return;
}

// Convert image to sepia
void sepia(int height, int width, RGBTRIPLE image[height][width])
{
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            RGBTRIPLE curr = image[i][j];

            BYTE sepiaRed = tobyte(.393 * curr.rgbtRed + .769 * curr.rgbtGreen + .189 * curr.rgbtBlue);
            BYTE sepiaGreen = tobyte(.349 * curr.rgbtRed + .686 * curr.rgbtGreen + .168 * curr.rgbtBlue);
            BYTE sepiaBlue = tobyte(.272 * curr.rgbtRed + .534 * curr.rgbtGreen + .131 * curr.rgbtBlue);

            RGBTRIPLE sepia;
            sepia.rgbtRed = sepiaRed;
            sepia.rgbtGreen = sepiaGreen;
            sepia.rgbtBlue = sepiaBlue;
            image[i][j] = sepia;
        }
    }
    return;
}

// Reflect image horizontally
void reflect(int height, int width, RGBTRIPLE image[height][width])
{
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width / 2; j++)
        {
            RGBTRIPLE temp = image[i][j];
            image[i][j] = image[i][width - 1 - j];
            image[i][width - 1 - j] = temp;
        }
    }
    return;
}

RGBTRIPLE average(int height, int width, int row, int col, RGBTRIPLE image[height][width])
{
    // constrain borders

    int startRow = maxi(row - 1, 0);
    int endRow = mini(row + 1, height - 1);
    int startCol = maxi(col - 1, 0);
    int endCol = mini(col + 1, width - 1);

    int count = 0;
    int totalRed = 0;
    int totalGreen = 0;
    int totalBlue = 0;

    // sum active region
    for (int i = startRow; i <= endRow; i++)
    {
        for (int j = startCol; j <= endCol; j++)
        {
            totalRed += image[i][j].rgbtRed;
            totalGreen += image[i][j].rgbtGreen;
            totalBlue += image[i][j].rgbtBlue;
            count++;
        }
    }

    double fcount = (double)count;
    RGBTRIPLE avg;
    avg.rgbtRed = tobyte(totalRed / fcount);
    avg.rgbtGreen = tobyte(totalGreen / fcount);
    avg.rgbtBlue = tobyte(totalBlue / fcount);
    return avg;
}

// Blur image
void blur(int height, int width, RGBTRIPLE image[height][width])
{
    RGBTRIPLE copy[height][width];
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            BYTE startRow = 0;

            copy[i][j] = image[i][j];
        }
    }

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            image[i][j] = average(height, width, i, j, copy);
        }
    }
}
