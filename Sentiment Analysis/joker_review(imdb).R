library(rvest)
url<-"https://www.imdb.com/title/tt7286456/reviews?ref_=tt_urv"

joker_review<-NULL
for (i  in 1:40){
  iurl <- read_html(as.character(paste(url,i,sep="=")))
  rev <- iurl %>% html_nodes(".imdb-user-review") %>% html_text()
  joker_review <- c(joker_review,rev)
}
write.table(joker_review,"joker_review.txt")
getwd()

###############
######Sentiment analysis########
txt<- joker_review
str(joker_review)
length(joker_review)
View(txt)

#install.packages("tm")
library(tm)

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
library(wordcloud)
windows()
wordcloud(y1,min.freq = 10,random.order =FALSE,colors = rainbow(30))
y2<-tm_map(y1,removeWords,c('movie','joker','review','found','just','film','october'
                            ,'helpful','every','far','sign','september','will','thats'
                            ,'comes','much','maybe','know','really','first','dosent','take','vote'))
y2<-tm_map(y2,stripWhitespace)
wordcloud(y2,min.freq = 10,random.order =FALSE,colors = rainbow(30))
# converting unstructured data into structured data
tdm<-TermDocumentMatrix(y2)
tdm
dtm<- t(tdm)
dtm<- DocumentTermMatrix(y2)
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
w_sub<- subset(q,q>=100)
w_sub
barplot(w_sub,las=2,col = rainbow(40))

tdm<-TermDocumentMatrix(y2)
tdm<- as.matrix(tdm)
tdm[1:40,10:100]
q<-rowSums(tdm)
q
w_sub1<-subset(q,q>=60)
w_sub1
barplot(w_sub1,las=2,col = rainbow(40))
y3<-tm_map(y2,removeWords,c('cant','yet','truly','just','get','instead','somewhat','just','joker','tell','didnt','dont','feels','movie'))
y3<-tm_map(y3,stripWhitespace)
tdm<-TermDocumentMatrix(y3)
tdm<-as.matrix(tdm)
q<-rowSums(tdm)
q
w_sub<-subset(q,q>=60)
w_sub
barplot(w_sub,las=2,col = rainbow(40))
wordcloud(words = names(w_sub), freq = w_sub,col=rainbow(40))
w_sub1 <- sort(rowSums(tdm), decreasing = TRUE)
head(w_sub1)
windows()
wordcloud(words = names(w_sub1), freq = w_sub1, random.order=F, colors=rainbow(40), scale = c(2,0.5), rot.per = 0.4)

wordcloud(words = names(w_sub), freq = w_sub1, random.order=F, colors= rainbow(30),scale=c(3,0.5),rot.per=0.3)
####################word cloud#############################
#install.packages("wordcloud2")
library(wordcloud2)

w1 <- data.frame(names(w_sub), w_sub)
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
my_data<-readLines("C:\\Users\\sriva\\OneDrive\\Documents\\joker_review.txt")
s_v<-get_sentences(my_data)
class(s_v)
str(s_v)

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
head(nrc_s_v)
barplot(sort(colSums(prop.table(nrc_s_v[,1:10]))),horiz = T,las=1,main = "Emotions",xlab = "percentage",col = 1:10)
anger<-which.max(nrc_s_v$anger>0)
anger
Anger_review<-s_v[27]
negative<-which.max(nrc_s_v$negative)
negative
negative_review<-s_v[1003]
disgust<-which.max(nrc_s_v$surprise)
disgust
Disgust_review<-s_v[332]
Positive<-which.max(nrc_s_v$positive)
Positive
Positive_review<-s_v[149]
trust<-which.max(nrc_s_v$trust)
trust
trust_review<-s_v[63]
anticipation<-which.max(nrc_s_v$anticipation)
anticipation
anticipation_review<-s_v[1003]
joy<-which.max(nrc_s_v$joy)
joy
joy_review<-s_v[63]
fear<-which.max(nrc_s_v$fear)
fear
fear_review<-s_v[475]
surprise<-which.max(nrc_s_v$surprise)
surprise
surprise_review<-s_v[332]
sadness<-which.max(nrc_s_v$sadness)
sadness
sadness_review<-s_v[827]
Anger_review
Positive_review
negative_review
sadness_review
fear_review
joy_review
anticipation_review
trust_review
surprise_review

