# Cloud Computing
# Book Recommendation System
## Team Members:
Venkata Sai Meghana Setty (801074198)<br>
Vineetha Kagitha (801084149)<br>

## Overview:
This project, Book Recommendation system recommends books to users based on two approaches, one is User-item interactions and the other one is User-based similarity. Recommendation by User-item interactions is implemented by Alternating Least Square algorithm and User-User similarity recommendation is implemented by KNN cosine similarity between users.
<br>
We performed our recommendation system on an Explicit Data where there is data of ratings given to a book by the user.
<br>
### Alternating Least Square :
In this approach, we are trying to recommend books to users by predicting the user rating for all the books and this predicted rating is obtained for all the users to each and every book in the dataset through Matrix factorization. If we consider a matrix between User and Book, where each cell in the matrix is a rating given by the user to the book, this is a very sparse matrix. ALS is an optimization approach of finding the two Matrix factors (User matrix and Book matrix) for the sparse matrix where multiplying the matrix factors will result in predicted ratings of each user to each book. 
<br>
![ALSmat](https://user-images.githubusercontent.com/15541211/70202236-1abbe780-16e7-11ea-97fc-7d5e1882981d.PNG)
Here in the picture, U is a user matrix of size (Number of users X Number of latent factors) and V is an item matrix of size (Number of latent factors  X Number of items)
<br>The goal here is to find U,V such that R = U X V by solving the below equation.
![equation](https://user-images.githubusercontent.com/15541211/70202318-522a9400-16e7-11ea-93c9-6d7cc3fe4a08.PNG)

<em><u>Steps involved for implementing in PySpark:</u></em><br>
1. Generating ratings RDD in the form (UserID, BookID, Rating)
2. Initializing user and book matrix of 15 latent features with random values
3. Compute User matrix by considering book matrix as constant. Below is the equation to compute.
 ![usersequ](https://user-images.githubusercontent.com/15541211/70202339-6078b000-16e7-11ea-9412-0ed1366720fd.PNG)
4. Compute Book matrix by considering user matrix as constant. Below is the equation to compute.
![itemsequ](https://user-images.githubusercontent.com/15541211/70202350-666e9100-16e7-11ea-808c-6c3fa6ebddb8.PNG)
5. Repeat the steps 3 and 4 for 60 iterations
6. Compute predicted ratings for all the books and users by multiplying the user and books matrices.
7. Output the top 15 books (which are not rated before) with highest predicted ratings per user. 

### KNN Cosine Similarity :
In this approach, we are trying to recommend books to a user based on other similar users. We will be finding the cosine similarity between the users based on the common books they read and find k nearest neighbors for the particular user. We predict the rating of the books for the user based on the books we recommended.
<br>Cosine Similarity is the measure of calculating the angle between two vectors.

<br>where, Ai and Bi are components of vector A and B respectively.

<br>Based on the k nearest neighbors, we have used weighted-sum to get the top 15 book recommendation of the user.

<em><u>Steps involved for implementing in PySpark:</u></em><br>
1. We have the training set as (userid,bookid,rating)
2. We then group the dataset based on the userid. (userid,(bookid1,rating1),(bookid2,rating2),(bookid3,rating3).....)
3. We find the cartesian product of the dataset by userid and find the cosine similarity between different users.
4. For a user, we find the k nearest neighbors in the decreasing cosine similarity.
5. For each similar user, we get the books and find the top 15 books with the highest score.

## Motivation:
We are building a recommendation system for suggesting the books they might be interested in. We were interested in this project because that is what we see anywhere are go in kindle, google books or when we register to any book websites. We found it quite fascinating when the book we never knew about came up in suggestions and turned out to be one of the good books. This obviously increases the profits of the company but reading new books is always one kind of fun.

## Dataset:
The Dataset used for this Project is found in this link https://github.com/zygmuntz/goodbooks-10k
<br>The dataset used are
<br><strong>Books.csv</strong> , This contains information of each book along with its id, Title, Author, etc.
<br><strong>Ratings.csv</strong> , This contains ratings given by user to book in this format
 (User-ID, Book-ID,Rating)
<br>Due to the large corpus and training time constraint, we considered around 455000 ratings.

## Framework:
<br><strong>Technologies :</strong> Apache Spark, Python, dsba-hadoop Cluster
<br><strong>Libraries Used:</strong> Pandas, Math

## Results:
### Results for ALS Implementation is :

<em>Recommended Books for UserID 1 are : </em>
<br>The America's Test Kitchen Family Cookbook
<br>The Fannie Farmer Cookbook: Anniversary
<br>Ancestors of Avalon (Avalon, #5)
<br>Daniel Deronda
<br>Tales from Earthsea (Earthsea Cycle, #5)
<br>Nocturnes: Five Stories of Music and Nightfall
<br>The Women of Brewster Place
<br>An Artist of the Floating World
<br>Fledgling
<br>Anne Frank : The Biography
<br>The Highlander's Touch (Highlander, #3)
<br>Hedda Gabler
<br>I Love You Through and Through
<br>Touching Spirit Bear (Spirit Bear, #1)
<br>Fancy Nancy
<br>Big Trouble



### Results for KNN implementation is :

<em>For user 1 books -</em>

<br>The Book of Disquiet
<br>Story of a Girl
<br>Blessings
<br>A Bend in the River
<br>The Art of Racing in the Rain
<br>The Shadow of the Wind (The Cemetery of Forgotten Books
<br>How to Win Friends and Influence People
<br>An Abundance of Katherines
<br>I Know This Much Is True
<br>The House of God
<br>The Lorax
<br>Harry Potter Boxset (Harry Potter, #1-7)
<br>The Prophet
<br>Twenty Wishes (Blossom Street, #5)
<br>2001: A Space Odyssey (Space Odyssey, #1)



<strong>Rest of the results can be found in:
<br>ALSOutput.txt
<br>KNNOutput.txt
</strong>

## Performance Evaluation:
We determined Root Mean Square Error-values for both the approaches User-based(KNN) and
User-Item(ALS) to evaluate the performance of the approaches.
### ALS:
Overall Average RMSE is **3.91751520938**
### KNN:
('RMSE error:', <strong>0.48284271247461896</strong>)

## Final Product:
As mentioned in the proposal, We have achieved extracting the recommendations based on user choices by two approaches, User-based similarity and User-Item interactions. We did not implement the recommendations based on Items like we mentioned that we are likely going to achieve. 


## Work Division:

#### Venkata Sai Meghana Setty
Implementing user-based similarity recommendation system by KNN approach
#### Vineetha Kagitha
Implementing user-item interactions based recommendation system by ALS approach

## References:
http://stanford.edu/~rezab/classes/cme323/S15/notes/lec14.pdf

https://medium.com/radon-dev/als-implicit-collaborative-filtering-5ed653ba39fe

https://towardsdatascience.com/how-did-we-build-book-recommender-systems-in-an-hour-part-2-k-nearest-neighbors-and-matrix-c04b3c2ef55c

https://en.wikipedia.org/wiki/Cosine_similarity






