#                                                   Cloud Computing
Book Recommendation System
Team Members:
Venkata Sai Meghana Setty (801074198)
Vineetha Kagitha (801084149)

Overview:
This project, Book Recommendation system recommends books to users based on two approaches, one is User-item interactions and the other one is User-based similarity. Recommendation by User-item interactions is implemented by Alternating Least Square algorithm and User-User similarity recommendation is implemented by KNN cosine similarity between users.

We performed our recommendation system on an Explicit Data where there is data of ratings given to a book by the user.

Alternating Least Square :
In this approach, we are trying to recommend books to users by predicting the user rating for all the books and this predicted rating is obtained for all the users to each and every book in the dataset through Matrix factorization. If we consider a matrix between User and Book, where each cell in the matrix is a rating given by the user to the book, this is a very sparse matrix. ALS is an optimization approach of finding the two Matrix factors (User matrix and Book matrix) for the sparse matrix where multiplying the matrix factors will result in predicted ratings of each user to each book. 

Here in the picture, U is a user matrix of size (Number of users X Number of latent factors) and V is an item matrix of size (Number of latent factors  X Number of items)
The goal here is to find U,V such that R = U X V by solving the below equation.

Steps involved for implementing in PySpark:
Generating ratings RDD in the form (UserID, BookID, Rating)
Initializing user and book matrix of 15 latent features with random values
Compute User matrix by considering book matrix as constant. Below is the equation to compute.
 
Compute Book matrix by considering user matrix as constant. Below is the equation to compute.

Repeat the steps 3 and 4 for 60 iterations
Compute predicted ratings for all the books and users by multiplying the user and books matrices.
Output the top 15 books (which are not rated before) with highest predicted ratings per user. 

KNN Cosine Similarity :
In this approach, we are trying to recommend books to a user based on other similar users. We will be finding the cosine similarity between the users based on the common books they read and find k nearest neighbors for the particular user. We predict the rating of the books for the user based on the books we recommended.
Cosine Similarity is the measure of calculating the angle between two vectors.

where, Ai and Bi are components of vector A and B respectively.

Based on the k nearest neighbors, we have used weighted-sum to get the top 15 book recommendation of the user.

Steps involved for implementing in PySpark:
We have the training set as (userid,bookid,rating)
We then group the dataset based on the userid. (userid,(bookid1,rating1),(bookid2,rating2),(bookid3,rating3).....)
We find the cartesian product of the dataset by userid and find the cosine similarity between different users.
For a user, we find the k nearest neighbors in the decreasing cosine similarity.
For each similar user, we get the books and find the top 15 books with the highest score.
Motivation:
	We are building a recommendation system for suggesting the books they might be interested in. We were interested in this project because that is what we see anywhere are go in kindle, google books or when we register to any book websites. We found it quite fascinating when the book we never knew about came up in suggestions and turned out to be one of the good books. This obviously increases the profits of the company but reading new books is always one kind of fun.
Dataset:
The Dataset used for this Project is found in this link https://github.com/zygmuntz/goodbooks-10k
The dataset used are
Books.csv , This contains information of each book along with its id, Title, Author, etc.
Ratings.csv , This contains ratings given by user to book in this format
 (User-ID, Book-ID,Rating)
Due to the large corpus and training time constraint, we considered around 455000 ratings.
Framework:
Technologies : Apache Spark, Python, dsba-hadoop Cluster
 Libraries Used: Pandas, Math
Results:
Results for ALS Implementation is :

Recommended Books for UserID 1 are : 
The America's Test Kitchen Family Cookbook
The Fannie Farmer Cookbook: Anniversary
Ancestors of Avalon (Avalon, #5)
Daniel Deronda
Tales from Earthsea (Earthsea Cycle, #5)
Nocturnes: Five Stories of Music and Nightfall
The Women of Brewster Place
An Artist of the Floating World
Fledgling
Anne Frank : The Biography
The Highlander's Touch (Highlander, #3)
Hedda Gabler
I Love You Through and Through
Touching Spirit Bear (Spirit Bear, #1)
Fancy Nancy
Big Trouble



Results for KNN implementation is :

For user 1 books -

The Book of Disquiet
Story of a Girl
Blessings
A Bend in the River
The Art of Racing in the Rain
The Shadow of the Wind (The Cemetery of Forgotten Books
How to Win Friends and Influence People
An Abundance of Katherines
I Know This Much Is True
The House of God
The Lorax
Harry Potter Boxset (Harry Potter, #1-7)
The Prophet
Twenty Wishes (Blossom Street, #5)
2001: A Space Odyssey (Space Odyssey, #1)



Rest of the results can be found in:
ALSOutput.txt
KNNOutput.txt
Performance Evaluation:
We determined Root Mean Square Error-values for both the approaches User-based(KNN) and
User-Item(ALS) to evaluate the performance of the approaches.

KNN:
('RMSE error:', 0.48284271247461896)
Final Product:
As mentioned in the proposal, We have achieved extracting the recommendations based on user choices by two approaches, User-based similarity and User-Item interactions. We did not implement the recommendations based on Items like we mentioned that we are likely going to achieve. 
Work Division:

Venkata Sai Meghana Setty
Implementing user-based similarity recommendation system by KNN approach
Vineetha
Implementing user-item interactions based recommendation system by ALS approach

References:
http://stanford.edu/~rezab/classes/cme323/S15/notes/lec14.pdf
https://medium.com/radon-dev/als-implicit-collaborative-filtering-5ed653ba39fe








