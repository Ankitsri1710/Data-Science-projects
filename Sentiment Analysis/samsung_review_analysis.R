#### Importing Important libraries #####
library(rvest)
library(XML)
library(magrittr)

aurl<- "https://www.amazon.in/Samsung-Galaxy-Storage-Additional-Exchange/product-reviews/B07X9YN2FJ/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews"

samsung_review<- NULL

for (i  in 1:40){
      murl <- read_html(as.character(paste(aurl,i,sep="=")))
      rev <- murl %>% html_nodes(".review-text") %>% html_text()
      samsung_review <- c(samsung_review,rev)
}
write.table(samsung_review,"samsung_review.txt")
getwd()
###############
######Sentiment analysis########
txt<- samsung_review
str(samsung_review)
length(samsung_review)
View(txt)

#install.packages("tm")
library(tm)
# Converting into corpus
y<-Corpus(VectorSource(txt))
inspect(y[2])

y <- tm_map(y, function(x) iconv(enc2utf8(x), sub='byte'))

y1 <- tm_map(y, tolower)
inspect(y1[2])
y1 <- tm_map(y1,removePunctuation)
inspect(y1[2])
y1 <- tm_map(y1,removeNumbers)
inspect(y1[2])
y1 <- tm_map(y1,removeWords,stopwords('english'))
inspect(y1[2])
y1 <- tm_map(y1,stripWhitespace)
inspect(y1[2])

#install.packages("wordcloud")
library(wordcloud)
windows()
wordcloud(y1,min.freq = 10,random.order =FALSE,colors = rainbow(30))
# converting unstructured data into structured data
tdm<-TermDocumentMatrix(y1)
tdm
dtm<- t(tdm)
dtm<- DocumentTermMatrix(y1)
dtm

corpus.dtm.frequent<-removeSparseTerms(tdm,0.99)
corpus.dtm.frequent
tdm<-as.matrix(tdm)
length(tdm)
str(tdm)
dim(tdm)

tdm[1:20,1:10]

q<-rowSums(tdm)
q
head(q)
min(q)
max(q)
w_sub<- subset(q,q>=60)
w_sub
barplot(w_sub,las=2,col = rainbow(40))
# Again mapping of corpus
y1<- tm_map(y1,removeWords,c('phone','will','even','however','mah','really','just',"won't",'dosent','users','way','use','one','either','always','played'))
y1<- tm_map(y1,stripWhitespace)
inspect(y1[8])
y2<-tm_map(y1,removeWords,c('works','dosent','got','thing','like','bcz','able','dont'))
y2<-tm_map(y2,stripWhitespace)
wordcloud(y2,min.freq = 10,random.order =FALSE,colors = rainbow(30))
tdm<-TermDocumentMatrix(y2)
tdm<- as.matrix(tdm)
tdm[1:40,10:100]
q<-rowSums(tdm)
q
w_sub<-subset(q,q>=60)
w_sub
barplot(w_sub,las=2,col = rainbow(40))
wordcloud(words = names(w_sub), freq = w_sub,col=rainbow(40))
w_sub1 <- sort(rowSums(tdm), decreasing = TRUE)
head(w_sub1)
windows()
wordcloud(words = names(w_sub1), freq = w_sub1, random.order=F
          , colors=rainbow(40), scale = c(2,0.5), rot.per = 0.4)

wordcloud(words = names(w_sub1), freq = w_sub1, random.order=F
          , colors= rainbow(30),scale=c(3,0.5),rot.per=0.3)
####################word cloud#############################
#install.packages("wordcloud2")
library(wordcloud2)

w1 <- data.frame(names(w_sub1), w_sub1)
colnames(w1) <- c('word', 'freq')

wordcloud2(w1, size=0.3, shape='circle')
?wordcloud2

wordcloud2(w1, size=0.3, shape = 'triangle')
wordcloud2(w1, size=0.3, shape = 'star')


#####################################
# lOADING Positive and Negative words  
pos.words <- readLines(file.choose())	# read-in positive-words.txt
neg.words <- readLines(file.choose()) 	# read-in negative-words.txt

stopwdrds <-  readLines(file.choose())

### Positive word cloud ###
pos.matches <- match(names(w_sub1), pos.words)
pos.matches <- !is.na(pos.matches)
freq_pos <- w_sub1[pos.matches]
names <- names(freq_pos)
windows()
wordcloud(names, freq_pos, scale=c(4,1), colors = brewer.pal(8,"Dark2"))


### Matching Negative words ###
neg.matches <- match(names(w_sub1), neg.words)
neg.matches <- !is.na(neg.matches)
freq_neg <- w_sub1[neg.matches]
names <- names(freq_neg)
windows()
wordcloud(names, freq_neg, scale=c(4,.5), colors = brewer.pal(8, "Dark2"))
#############SENTIMENT ANALYSIS################
#install.packages("syuzhet")
library(syuzhet)
my_data<-readLines("C:\\Users\\sriva\\OneDrive\\Documents\\samsung_review.txt")
s_v<-get_sentences(my_data)
class(s_v)
str(s_v)
head(s_v)
sentiment_vector<-get_sentiment(s_v,method = "bing")
sentiment_vector
affin_s_v<-get_sentiment(s_v,method = "afinn")


plot(sentiment_vector,type="l",main = "Plot Trajectory",xlab = "Narrative Time",ylab = "Emotional valence")
abline(h=0, col ="blue")
neg<- s_v[which.min(sentiment_vector)]     
neg
pos<- s_v[which.max(sentiment_vector)]
pos
# Sentiment Analysis using "nrc" dictionary.
#nrc_s_v<-get_sentiment(s_v, method = "nrc")
nrc_s_v<-get_nrc_sentiment(s_v)
nrc_s_v
barplot(sort(colSums(prop.table(nrc_s_v[,1:10]))),horiz = T,las=1,main = "Emotions",xlab = "percentage",col = 1:10,cex.names = 0.7)
windows()
anger<-which.max(nrc_s_v$anger>3)
anger
Anger_review<-s_v[235]
disgust<-which.max(nrc_s_v$surprise)
disgust
Disgust_review<-s_v[131]
Positive<-which.max(nrc_s_v$positive)
Positive
Positive_review<-s_v[131]
Negative<-which.max(nrc_s_v$negative)
Negative
Negative_review<-s_v[119]
trust<-which.max(nrc_s_v$trust)
trust
trust_review<-s_v[131]
anticipation<-which.max(nrc_s_v$anticipation)
anticipation
anticipation_review<-s_v[131]
joy<-which.max(nrc_s_v$joy)
joy
joy_review<-s_v[131]
fear<-which.max(nrc_s_v$fear)
fear
fear_review<-s_v[139]
surprise<-which.max(nrc_s_v$surprise)
surprise
surprise_review<-s_v[131]
sadness<-which.max(nrc_s_v$sadness)
sadness
sadness_review<-s_v[60]
Anger_review
Positive_review
Negative_review
sadness_review
fear_review
joy_review
anticipation_review
trust_review
surprise_review
