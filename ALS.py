#vineetha kagitha
#vkagitha@uncc.edu

import pyspark
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
import pandas as pd
import numpy as np
import sys
import math 

def multiplyTrans(value):
    return np.dot(value,np.transpose(value))

def multiply(val1,val2):
    return np.dot(val1,val2)

def getPredictedRating(X,Y):
    val =  np.dot(np.transpose(X),Y)
    return val[0][0]

def compute(id,mat):
    val = mat.value[id]
    return multiplyTrans(val)

def mulRating(r,id,mat):
    val = mat.value[id]
    return r*val;
  
def getRMSError(rating,pred_rating,num_ratings):
    diff = rating - pred_rating
    diff_sq = (diff**2)/num_ratings
    return diff_sq


if __name__ == "__main__":
    
    conf = (SparkConf().setAppName("ALS Recommendation System")
    #.setMaster("local")
    .set("spark.executor.memory","4g")
    .set("spark.driver.memory","4g")
    .set("spark.cores.max","4")
    .set("spark.network.timeout", "600s")
    .set('spark.kryoserializer.buffer.max', '512'))
    sc = SparkContext(conf=conf)
    sql_sc = SQLContext(sc)
    
    # reading ratings.csv file into a dataframe
    ratings_file = sys.argv[1]+'/new_ratings.csv'
    ratings_df = pd.read_csv(ratings_file)
    ratings_df = sql_sc.createDataFrame(ratings_df)
    rating_RDD = ratings_df.rdd
    
    # ratings_RDD is of form (user_id,book_id,rating)
    ratings_RDD = rating_RDD.map(lambda record : (str(record['user_id']),str(record['book_id']),int(record['rating']))).cache()
    ratings_RDD.collect()
    no_of_ratings = sc.broadcast(ratings_RDD.count())
    #ratings_RDD.collect()
    
    # Extracting book_id and its title from the books.csv file
    usecolumns = [
    'book_id',
    'title',
    ]
    books_file = sys.argv[1]+'/books.csv'
    books_df = pd.read_csv(books_file,usecols = usecolumns)
    books_df = sql_sc.createDataFrame(books_df)
    books_RDD = books_df.rdd
    
    # books_title is a dictionary which maps from book_id to title
    books_RDD = books_RDD.map(lambda book:(str(book['book_id']),book['title'])).cache()
    books_title = books_RDD.collectAsMap()
    
    no_of_books = books_RDD.count()
    
    # users_RDD contains the user_id's of users who have rated atleast one book.
    users_RDD = ratings_RDD.map(lambda record:str(record[0])).distinct().cache()
    no_of_users = users_RDD.count()
    
    # 15 latent factors are considered for this computation. 
    no_of_features = 15 
    
    # We randomly assign values into users matrix and books matrix, here both of them are column matrices
    users_mat = users_RDD.map(lambda user:(user,np.random.rand(no_of_features,1))).cache()
    users_broad = sc.broadcast(users_mat.collectAsMap())
    books_mat = books_RDD.map(lambda book:(book[0],np.random.rand(no_of_features,1))).cache()
    books_broad = sc.broadcast(books_mat.collectAsMap())

    # applying L2 regularizer with regularization value as 0.001
    regularization_param = 0.001
    IK = regularization_param*np.identity(no_of_features)

    num_of_iter = 60
    RMSE = np.zeros(num_of_iter)
    
    """ The ALS algorithm of fixing each of user matrix and books matrix constant once a time 
    and computing the other in the form of a quadratic equation is implemented for 15 iterations (assuming it converges for 15 iterations)"""
    for i in range(num_of_iter):
        # Fixing books matrix and computing users matrix
        YYT = ratings_RDD.map(lambda uir : (uir[0],compute(uir[1],books_broad))).reduceByKey(lambda a,b:np.add(a,b)).map(lambda yyt: (yyt[0],np.linalg.inv(np.add(yyt[1],IK))))
        YYT.collect()
    
        RY = ratings_RDD.map(lambda uir : (uir[0],mulRating(uir[2],uir[1],books_broad)))
        RY.collect()
        
        users_mat = YYT.join(RY).map(lambda tup:(tup[0],multiply(tup[1][0],tup[1][1]))).cache()
        users_broad = sc.broadcast(users_mat.collectAsMap())
        
        # Fixing users matrix and computing books matrix
        
        XXT = ratings_RDD.map(lambda uir : (uir[1],compute(uir[0],users_broad))).reduceByKey(lambda a,b:np.add(a,b)).map(lambda xxt: (xxt[0],np.linalg.inv(np.add(xxt[1],IK))))
        XXT.collect()
        
        RX = ratings_RDD.map(lambda uir : (uir[1],mulRating(uir[2],uir[0],users_broad)))
        RX.collect()
    
        books_mat = XXT.join(RX).map(lambda tup:(tup[0],multiply(tup[1][0],tup[1][1]))).cache()
        books_broad = sc.broadcast(books_mat.collectAsMap())
        
        rms = ratings_RDD.map(lambda r : ("RMSE",getRMSError(r[2],getPredictedRating(users_broad.value[r[0]],books_broad.value[r[1]]),no_of_ratings.value))).reduceByKey(lambda a,b:np.add(a,b)).collect()
        RMSE[i] = math.sqrt(rms[0][1])

    
    useridlist = ['1','6422','1756','6281','6165','376','467','2262','6146','6432','6434','2046','233','1689','6443','6381']
    #u_id = str(sys.argv[2])
    for u_id in useridlist:
        user_rated_books = ratings_RDD.filter(lambda uir : uir[0] == u_id).map(lambda val:val[1]).collect()
        uval = users_broad.value[u_id]
        predicted_ratings = {book:getPredictedRating(uval,bval) for book,bval in books_broad.value.items()}
        recommended_books = sorted(predicted_ratings.items(), key=lambda x: x[1],reverse = True)
        print("-----------------------------------------------------------------------------")
        print("Recommended Books for UserID {0} are : ".format(u_id))
        i = 0
        for book,r in recommended_books:
	    if i<16 and book not in user_rated_books:
                i+=1
                print(books_title[book].encode('utf-8'))
    
    print("\nPerformance Evaluation : ")   
    #print("RMSE values for each iteration: {0}".format(RMSE))
    print("Overall Average RMSE is {0}".format(np.mean(RMSE)))
            
    ratings_RDD.unpersist()
    books_RDD.unpersist()
    users_RDD.unpersist()
    users_mat.unpersist()
    books_mat.unpersist()

