# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 19:59:52 2019

@author: Venkata Sai Meghana Setty
email: msetty@uncc.edu
"""
import sys
from pyspark import SparkContext,SparkConf
from pyspark.sql import SQLContext
import pandas as pd
import math

def Recommendations(t,userBookRatingList,recombooks = 15):
    weightedsum ={}
    cosinesum = {}
    user = t[0]
    neighbors = t[1]
    for(neighbor,cosinesimilarity,numcommonrat) in neighbors:
        ubookslist = userBookRatingList.get(neighbor)
        if ubookslist:
            for index in range(0,len(ubookslist[0])):
                bookid = ubookslist[0][index]
                if bookid in cosinesum.keys():    
                    cosinesum[bookid] += cosinesimilarity
                else:
                    cosinesum[bookid] = cosinesimilarity
                if bookid in weightedsum.keys(): 
                    weightedsum[bookid] += cosinesimilarity*ubookslist[1][index]
                else:
                    weightedsum[bookid] = cosinesimilarity*ubookslist[1][index]
    sugbooks = []
    for bookid,csum in cosinesum.items():
        if csum == 0:
            continue
        sugbooks.append((bookid,1.0*weightedsum[bookid]/csum))
    sugbooks.sort(key=lambda x: x[1], reverse=True)
    return (user,sugbooks[:recombooks])

def UserBookRatings(userRatingGroup):   
    UserID = userRatingGroup[0]   
    lbooks = [item[0] for item in userRatingGroup[1]]   
    lrating = [item[1] for item in userRatingGroup[1]]  
    return (UserID, (lbooks, lrating))

def UserBookRatings_broadcast(sc, TrainRDD):
    luserBook = TrainRDD.map(lambda x: UserBookRatings(x)).collect()
    print('got the user book ratings')
    duserBook = {}
    for (user, l) in luserBook:
        duserBook[user] = l
    return (sc.broadcast(duserBook))

def GroupByUserID(t):
    return [(t[0][0],(t[0][1],t[1][0],t[1][1])),(t[0][1],(t[0][0],t[1][0],t[1][1]))]

def HigherCosineUsers(t,k):
    sortedneighborslist = sorted(t[1],key=lambda t:t[1],reverse=True) #decreasing order
    return (t[0],sortedneighborslist[:k])
    
def cosineSimilarityBetweenUsers(userpairinfo):
    ratingpairs = userpairinfo[1]
    dotproduct = 0.0
    sq1 = 0.0
    sq2 = 0.0
    totalpairs = 0
    for (a,b) in ratingpairs:
        dotproduct += a*b
        sq1 += a*a
        sq2 += b*b
        totalpairs +=1
    sq1 = math.sqrt(sq1)
    sq2 = math.sqrt(sq2)
    if sq1 == 0 or sq2 == 0:
        return (userpairinfo[0],(0,totalpairs))
    else:
        return (userpairinfo[0],(dotproduct/(sq1*sq2),totalpairs))

def BroadcastBookNamesDict(sc,bookRDD):
    books = bookRDD.collect()
    dbooks = {}
    for (bookid, authorname, title) in books:
        dbooks[bookid] = (authorname,title)
    return (sc.broadcast(dbooks))


def BookNames(user,bookrecords,bookdict):
    CompletebookInfo = []
    for n in bookrecords:
        CompletebookInfo.append((bookdict[n[0]],math.floor(n[1])))
    return (user,CompletebookInfo)
        

def SameBooksRating(usertuple):
    user1=usertuple[0][0]
    user2=usertuple[1][0]
    
    commonratinglist = []
    sortbookratingtupUser1 = sorted(usertuple[0][1],key=lambda x: x[0])
    sortbookratingtupUser2 = sorted(usertuple[1][1],key=lambda x: x[0])
    
    i=0
    j=0
    while i<len(sortbookratingtupUser1) and j<len(sortbookratingtupUser2):
        if sortbookratingtupUser1[i][0] == sortbookratingtupUser2[j][0]:
            commonratinglist.append([sortbookratingtupUser1[i][1],sortbookratingtupUser2[j][1]])
            i+=1
            j+=1
        elif sortbookratingtupUser1[i][0] < sortbookratingtupUser2[j][0]:
            i+=1
        else:
            j+=1
    return ((user1,user2),commonratinglist)

def FindRMSEerror(list1,list2):
    sum = 0.0
    n = 0
    for (obook,orating) in list1:
        for (rbook,rrating) in list2:
            if (obook[1] == rbook[1]):
                sum += abs(orating-rrating)**2
                n+=1
    if n == 0:
        return 0.0
    else:
        return math.sqrt(sum/n)
#The recommender system is done based on user-user mapping
if __name__=='__main__':
    arguments = len(sys.argv)
    #if arguments are not in the format we return it
    if arguments != 1:
        print("Improper arguments knn.py")
    else:
        conf = (SparkConf()
        .setMaster('local')
        .setAppName('KNN')
        .set('spark.executor.memory','6g')
        .set('spark.driver.memory','6g')
        .set('spark.cores.max','6')
        .set('spark.driver.host','127.0.0.1'))
        sc = SparkContext.getOrCreate(conf = conf) # creating the spark context
        sql_sc = SQLContext(sc)
        
        
        
        usebookcolumns = [
                'book_id',
                'authors',
                'title'
                ]
        
        books_df = pd.read_csv('./books.csv',usecols = usebookcolumns)
        books_df = sql_sc.createDataFrame(books_df)
        booksRDD = books_df.rdd # use this RDD to keep track of all the list you need
        

        #rating count
        useratingcols = [
                'user_id',
                'book_id',
                'rating']
        rating_df = pd.read_csv('./smallratings.csv',usecols=useratingcols)
        rating_df = sql_sc.createDataFrame(rating_df)
        ratingRDD = rating_df.rdd
        N = ratingRDD.count()
        print(N)
        
        trainingRDD = ratingRDD
        trainingRDD = trainingRDD.map(lambda t:(t[0],(t[1],t[2]))).groupByKey().mapValues(list).cache() #we will use this RDD to train so it needs to be cached
        print(trainingRDD.count())
        print(trainingRDD.first())
        #As this is user-user KNN , we need to find the cartisian product of the traindata
        cartesianUserRDD = trainingRDD.cartesian(trainingRDD)
        UniqueUserPairsRDD = cartesianUserRDD.filter(lambda x: x[0][0] < x[1][0])
        #finding the cosine_similarity
        #for cosinesimilarity we need ratings of books which two users commonly rated.
        userSimilarPairRDD = UniqueUserPairsRDD.map(lambda t: SameBooksRating(t))
        #cosine_similarity
        userPairCosineSimilarityRDD = userSimilarPairRDD.map(lambda t:cosineSimilarityBetweenUsers(t))
        print(userPairCosineSimilarityRDD.first())
        #dividing the userpairs and grouping them by userid
        UserNeighborsRDD = userPairCosineSimilarityRDD.flatMap(lambda t: GroupByUserID(t)).groupByKey().map(lambda x : (x[0], list(x[1])))
        print(UserNeighborsRDD.first())
        UserHigherNeighborsRDD = UserNeighborsRDD.map(lambda t: HigherCosineUsers(t, 200))
        print(UserHigherNeighborsRDD.first())
        
        UserBookRatingList = UserBookRatings_broadcast(sc, trainingRDD)


        RecommendedBooks = UserHigherNeighborsRDD.map(lambda t: Recommendations(t, UserBookRatingList.value))
        DBookNames = BroadcastBookNamesDict(sc, booksRDD)
        BookRecommendationtoUser = RecommendedBooks.map(lambda t: BookNames(t[0], t[1], DBookNames.value))
        #userid = int(sys.argv[1])
	
        #print(RecommendedTracksForUser.collect())
        #tracks = BookRecommendationtoUser.filter(lambda t:t[0]==userid).collect()
        #print ('For user %s recommended books is \"%s\"' %(userid,tracks) )
        #calculatiing for 15 users
        useridlist = [1,2,4,6422,1756]#,6281,6165,376,467,2262,6146,6432,6434,2046,233,1689,6443,6381]
        booksrecommended = BookRecommendationtoUser.filter(lambda t: t[0] in useridlist).collect()
        booksrecommendeddict = {}
        for (userid,booklist) in booksrecommended:
            print ('For user %s books - "%s"' %(userid,booklist) )
            booksrecommendeddict[userid] = booklist
            
        originalbooksrating = trainingRDD.filter(lambda t:t[0] in useridlist)
        
        originalbooknamesrating = originalbooksrating.map(lambda t: BookNames(t[0],t[1],DBookNames.value))
        orgbooks = originalbooknamesrating.collect()
        orgbooksdict = {}
        for userid,booklist in orgbooks:
            orgbooksdict[userid] = booklist
            
        #we have two dicts compare them
        print()
        RMSE_error = 0.0
        for key in orgbooksdict.keys():
          RMSE_error +=FindRMSEerror(orgbooksdict[key],booksrecommendeddict[key])
        print('RMSE error:',RMSE_error/len(useridlist))
        #ratingRDD.unpersist()
        trainingRDD.unpersist()	        
        sc.stop()
