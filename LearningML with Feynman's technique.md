#Learning Machine Learning by Feynman's technique

## Introduction to AI
Artificial Intelligence (AI) is revolutionizing the way we interact with things and with our lives. Computers can innovate much faster and consistently than we can, and at the current rate of development, who's not willing to learn ML is going to lag behind.

The distinguished companies  below, backed by the biggest tech companies, are the ones to follow:
- DeepMind;
- OpenAI.

Machine Learning is one of many subfields of AI that issues the ability of computers "to learn without ever being programmed" (Arthur Samuel, 1959). ML is divided in:
- Supervised learning:
   • Linear regression;
   • Classification.
- Unsupervised learning:
   • Clustering;
   • Dimensionality reduction;
   • Reccomendation.
- Reinforcement learning:
   • Reward maximization.

According to multiple sources, machine learning is the core of the journey to AGI (Artificial General Intelligence), which is predicted to be the most important event of all human race (this represents that computers will be far more intelligent than humans and capable of developing themselves).

## Supervised Learning
Supervised Learning is a type of ML that uses datasets with labeled training examples.  It's the optimization of the function:

Y = f(x) + e
Y = Target variable;
x = Discrete variable(s);
e = error (or the fixed value).

By optimizing this, we're trying to find the best possible Y. It's possible to assign labels (classification) or predict continuous value (linear regression).

### Linear Regression
In linear regression we're trying to predict continuous values for our target variable. We'll use training datasets with as many data as possible. The number of lines generally is the ammount of observations and the number of columns is the ammount of features we have.

Problems like this are solved by supervised learning algorithms, like ordinary mean squares. This example methos is used for linear relationships between Y and x. The goal is to learn the model parameters in which our model is the most accurate possible.

To learn these parameters, we have to define a loss function (that measures how inacurate our model is) and to minimize this loss function.
