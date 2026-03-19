/**
 * @file RosenblattCard.cpp
 * @brief Implementation of the RosenblattCard class.
 */

#include <iostream>
#include "RosenblattCard.h"

/// Initializes a blank 20x20 card — 400 pixels all set to 0.
RosenblattCard::RosenblattCard()
{
    for(int row = 0; row < 20;row++)
    {
        for(int col = 0;col < 20;col++)
        {
            pixelData.push_back(0);
        }
    }
    
    sideLabel = "Even Split";
}
//--
/// Places a rectangle on the card. Determines the label by comparing how
/// many columns of the shape fall on the left (cols 0-9) vs right (cols 10-19).
void RosenblattCard::addRectangle(int topLeftX, int topLeftY, int width, int height)
{
    int numOnLeft = 10 - topLeftX;
    int numOnRight = (topLeftX + width) - 10;
    if(numOnLeft > numOnRight)
    {
        sideLabel = "left";
    }
    else if(numOnRight > numOnLeft)
    {
        sideLabel = "right";
    }
    //else- no clear left/right side
    
    for(int row = topLeftY;row < topLeftY + height;row++)
    {
        for(int col = topLeftX;col < topLeftX + width;col++)
        {
            if(row >= 0 && row < 20 && col >= 0 && col < 20)
            {
                int rowColIndex = convertRowColToIndex(row, col);
                pixelData[rowColIndex] = 1;
            }
        }
    }
}
//--
vector < int > RosenblattCard::get1DPixelData()
{
    return pixelData;
}
//--
string RosenblattCard::getSideLabel()
{
    return sideLabel;
}
//--
/// Displays the card as a 20x20 grid with '.' for active pixels and ' ' for
/// inactive ones. A center divider '||' separates left and right halves.
void RosenblattCard::print()
{
    cout<<"+---------||---------+"<<endl;
    for(int row = 0; row < 20;row++)
    {
        cout<<"|";
        for(int col = 0;col < 20;col++)
        {
            int rowColIndex = convertRowColToIndex(row, col);
            if(pixelData[rowColIndex] == 1)
            {
                cout<<".";
            }
            else
            {
                cout<<" ";
            }
        }
        cout<<"|"<<endl;
    }
    cout<<"+---------||---------+"<<endl;
    cout<<"Side: "<<sideLabel<<endl;
}
//--
/// Converts a 2D (row, col) coordinate to a flat array index.
/// Formula: index = row * 20 + col  (row-major, 20 columns per row).
int RosenblattCard::convertRowColToIndex(int row, int col)
{
    int retVal = (row * 20) + col;
    return retVal;
}
