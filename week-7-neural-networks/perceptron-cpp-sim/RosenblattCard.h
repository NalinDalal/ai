/**
 * @file RosenblattCard.h
 * @brief Represents one of Rosenblatt's stimulus cards — a 20x20 pixel image
 *        containing a rectangular shape on the left or right side.
 *
 * Each card stores 400 binary pixel values (1 = shape, 0 = background) in a
 * flat vector. The ground-truth label ("left" / "right") is determined by
 * which side of the center line (column 10) holds the majority of the shape.
 */

#pragma once
#include <vector>
#include <string>

using namespace std;

class RosenblattCard
{
public:
    /// Constructs a blank 20x20 card (all pixels set to 0).
    RosenblattCard();

    /// Places a rectangle on the card and auto-labels it "left" or "right".
    /// @param topLeftX  Column of the rectangle's top-left corner (0-19).
    /// @param topLeftY  Row of the rectangle's top-left corner (0-19).
    /// @param width     Width of the rectangle in pixels.
    /// @param height    Height of the rectangle in pixels.
    void addRectangle(int topLeftX, int topLeftY, int width, int height);

    /// Returns the 400-element flat pixel array (row-major order).
    vector < int > get1DPixelData();

    /// Returns the ground-truth label: "left", "right", or "Even Split".
    string getSideLabel();

    /// Prints the card to stdout with a center divider showing left/right halves.
    void print();

private:
    vector < int > pixelData;   ///< 400 binary pixels (row-major, 20 cols per row)
    string sideLabel;           ///< Ground-truth: "left", "right", or "Even Split"

    /// Converts (row, col) to a flat index: row * 20 + col.
    int convertRowColToIndex(int row, int col);
};
