/**
 * @file main.cpp
 * @brief Rosenblatt Perceptron simulation — trains on 50,000 randomly
 *        generated cards, then tests on 500 unseen cards.
 *
 * Pipeline:
 *   1. Generate 50k labeled training cards (random rectangles on left/right)
 *   2. Create a SideDetectorPerceptron and train it
 *   3. Generate 500 test cards and evaluate accuracy
 *   4. Print misclassified cards for inspection
 */

#include <iostream>
#include <vector>
#include <cassert>
#include <ctime>
#include "RosenblattCard.h"
#include "SideDetectorPerceptron.h"

using namespace std;

/// Generates random cards with odd-width rectangles to avoid even splits.
void loadData(vector < RosenblattCard >& labelledData, int requestedNumCards);

/// Non-AI baseline: counts active pixels on each side to classify.
string findSide(RosenblattCard& card);

int main()
{
    srand(time(0));

    vector < RosenblattCard > labelledData;
    loadData(labelledData, 50000);
    
    SideDetectorPerceptron p;
    p.train(labelledData);

    vector < RosenblattCard > testData;
    loadData(testData, 500);

    int countCorrect = 0;
    for(RosenblattCard card : testData)
    {
        string prediction = p.predict(card);
        
        if(prediction == card.getSideLabel())
        {
            countCorrect++;
        }
        else
        {
            card.print();
            cout<<"Prediction: "<<prediction<<endl;
            cout<<"Actual: "<<card.getSideLabel()<<endl<<endl;
        }
    }

    cout<<"Correct Predictions: "<<countCorrect<<"/"<<testData.size()<<endl;

    return 0;
}
//--
/// Generates `requestedNumCards` random cards. Each card gets a rectangle with:
///   - Position: x ∈ [0,17], y ∈ [0,17]
///   - Width: odd value in {3,5,7,9} (odd avoids exact center splits)
///   - Height: value in [2,8]
void loadData(vector < RosenblattCard >& labelledData, int requestedNumCards)
{
    for(int i = 0;i < requestedNumCards;i++)
    {
        int randX = rand() % 18; //0-17
        int randY = rand() % 18; //0-17
        int randWidth = (rand() % 7) + 3; //3-8
        //to avoid even splits down the middle, make the width odd
        if(randWidth % 2 == 0)
        {
            randWidth++;
        }
        int randHeight = (rand() % 7) + 2;
        
        RosenblattCard card;
        card.addRectangle(randX, randY, randWidth, randHeight);
        labelledData.push_back(card);
    }
}
//--
/// Non-AI baseline: iterates all 400 pixels, counts active pixels in
/// columns 0-9 (left) vs 10-19 (right), returns the side with more.
/// Achieves 100% accuracy — but only works because the problem is trivial.
string findSide(RosenblattCard& card)
{
    string retVal;
    vector < int > pixelData = card.get1DPixelData();
    int leftSideCount = 0;
    int rightSideCount = 0;
    
    for(int row = 0;row < 20;row++)
    {
        for(int col = 0;col < 20;col++)
        {
            int rowColIndex = (row * 20) + col;
            if(pixelData[rowColIndex] == 1)
            {
                if(col < 10) //column 0-9
                {
                    leftSideCount++;
                }
                else //column 10-19
                {
                    rightSideCount++;
                }
            }
        }
    }
    
    if(leftSideCount > rightSideCount)
    {
        retVal = "left";
    }
    else
    {
        retVal = "right";
    }
    
    return retVal;
}