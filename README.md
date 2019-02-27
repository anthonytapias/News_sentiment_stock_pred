# Predict Short Term Stock Price Movement from the News


## Introduction
In this project, I aim to see if a Machine Learning model can predict whether a stock's closing price will be higher than the opening price based on the number of positive or negative news of the company released by 12pm ET. An NLP model was built to distinguish if a particular article about a company was negative, positive or neutral. The NLP model ran each article of a particular company, acquired before 12pm, in order to get the sentiment rating for each article. Based on the number of positive articles, a Machine Learning model was trained to see if it can accuratly predict if the closing price will be higher or lower than the opening price of that day. 

## Building Natural Language Processing Model

### Data aquisition, processing and cleaning
I started off by looking online for labeled financial news articles. Luckily I found two datasets that in total that had about 18,000 labeled financial news articles from 1-9, 1 being negative, 9 being positive and 5 being neutral. I rescaled the sentiment scaling from 1-5 with 1 being negative, 2 weakly negative, 3 is neutral, 4 is weakly positive and 5 is positive. I did this because my model performed better.

After I rescaled the sentiment labels, I also dropped articles that were not revelent to the economy or as regarded financial news based on a revelency score below 33%. This also helped my model perform better. I manually labeled a couple of positive and negative articles written about Microsoft Corp. This was the company I aimed to test the model on. 

### Feature Engineer & Selection





