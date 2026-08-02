/**
 * @file SideDetectorPerceptron.cpp
 * @brief Implementation of the SideDetectorPerceptron class.
 */

#include "SideDetectorPerceptron.h"
#include <iostream>
#include <iomanip>
#include <vector>

using namespace std;

/// Constructor: initializes 400 weights to random values in [-1.0, +1.0],
/// truncated to one decimal place. Random init avoids the "zeroing problem"
/// where weights stuck at 0.0 never get updated for pixels that are always
/// correctly classified.
SideDetectorPerceptron::SideDetectorPerceptron()
{
    for(int i = 0;i < (20 * 20);i++)
    {
        double randomWeight = -1.0 + ((double(rand()) / double(INT_MAX)) * 2.0);
        randomWeight = floor(randomWeight * 10.0) / 10.0;
        weights.push_back(randomWeight);
    }
}
//--
/// Perceptron learning rule:
///   For each card, predict the side. If wrong:
///     - Predicted "left" but actually "right" → delta = +0.1  (nudge positive)
///     - Predicted "right" but actually "left" → delta = -0.1  (nudge negative)
///   Only weights at active pixels (pixel == 1) are updated.
///   This incremental approach (vs full-blast ±1.0) lets the perceptron
///   accumulate evidence across many examples.
void SideDetectorPerceptron::train(vector < RosenblattCard >& cards)
{
    cout<<"Before training"<<endl;
    printWeights();

    for(RosenblattCard card : cards)
    {
        string prediction = predict(card);
        if(prediction != card.getSideLabel())
        {
            double delta;
            if(prediction == "left" && card.getSideLabel() == "right")
            {
                delta = .1;
            }
            else
            {
                delta = -.1;
            }
            vector < int > pixelData = card.get1DPixelData();
            for(int i = 0;i < pixelData.size();i++)
            {
                if(pixelData[i] == 1)
                {
                    weights[i] = weights[i] + delta;
                }
            }
        }
    }
    cout<<"After training"<<endl;
    printWeights();
}
//--
/// Computes: sum = Σ (pixel[i] * weight[i]) for i = 0..399
/// Returns "left" if sum < 0, "right" if sum >= 0.
/// Intuition: left-side weights should converge to negative values and
/// right-side weights to positive values after training.
string SideDetectorPerceptron::predict(RosenblattCard& card)
{
    string retVal;
    vector< int > pixelData = card.get1DPixelData();
    double sum = 0.0;
    
    for(int i = 0;i < pixelData.size();i++)
    {
        double pixelValue = double(pixelData[i]);
        sum = sum + (pixelValue * weights[i]);
    }
    
    if(sum < 0)
    {
        retVal = "left";
    }
    else
    {
        retVal = "right";
    }
    
    return retVal;
}
//--
/// Displays the 400 weights in a 20-row × 20-col grid.
/// After training, you should see negative values on the left half and
/// positive values on the right half, with noisier values near the center.
void SideDetectorPerceptron::printWeights()
{
    cout<<"400 weights displayed in corresponding rows and cols"<<endl;
    cout<<"   Col:";
    for(int i = 0;i < 20;i++)
    {
        cout<<setw(5)<<i;
    }
    cout<<endl<<"-----------------------------------------------------------------------------------------------------------\n";

    for(int row = 0;row < 20;row++)
    {
        cout<<"Row "<<setw(2)<<row<<"| ";
        for(int col = 0;col < 20;col++)
        {
            int rowColIndex = (row * 20) + col;
            cout<<fixed<<setprecision(1)<<setw(4)<<weights[rowColIndex]<<" ";
        }
        cout<<endl;
    }
}