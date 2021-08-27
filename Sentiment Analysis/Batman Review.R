library(rvest)
library(XML)
library(magrittr)

url<-"https://www.imdb.com/title/tt0468569/reviews?ref_=tt_urv"
imdb_review<-NULL

for (i in 1:20) {
  iurl<-read_html(as.character(paste(url,i,sep = "=")))
  mrev<-iurl%>%html_nodes(".imdb-user-review")%>%html_text()
  imdb_review<-c(imdb_review,mrev)
}
write.table(imdb_review,"Batman_Review.txt")
getwd()
library(tm)
library(syuzhet)
txt<-imdb_review
str(imdb_review)
length(imdb_review)
View(txt)
y<-Corpus(VectorSource(txt))
y<-tm_map(y,function(x) iconv(enc2utf8(x),sub = 'byte'))
y<-tm_map(y,removePunctuation)
y<-tm_map(y,removeNumbers)
y<-tm_map(y,tolower)
y<-tm_map(y,removeWords,stopwords('english'))
y<-tm_map(y,stripWhitespace)
inspect(y[1])
Cleaned_data<-data.frame(text=sapply(y,as.character),stringsAsFactors = F)
write.table(Cleaned_data,"Batman Cleaned review.txt")
getwd()
ImdbReview<-read_lines("F:/My Assignment/nlp/Batman Cleaned review.txt")
sentiment<-get_sentences(ImdbReview)
class(sentiment)
str(sentiment)
head(sentiment)
sentiment_vector<-get_sentiment(sentiment,method = "bing")
sentiment_vector
affin_s_v<-get_sentiment(sentiment,method = "afinn")
plot(sentiment_vector,type="l",main = "Plot Trajectory",xlab = "Narrative Time",ylab = "Emotional valence")
abline(h=0, col ="blue")
plot(affin_s_v,type="l",main = "Plot Trajectory",xlab = "Narrative Time",ylab = "Emotional valence")
abline(h=0, col ="blue")
neg<- sentiment[which.min(sentiment_vector)]     
neg
pos<- sentiment[which.max(sentiment_vector)]
pos
# Sentiment Analysis using "nrc" dictionary.
#nrc_s_v<-get_sentiment(s_v, method = "nrc")
nrc_s_v<-get_nrc_sentiment(sentiment)
nrc_s_v
barplot(sort(colSums(prop.table(nrc_s_v[,1:10]))),horiz = T,las=1,main = "Emotions",xlab = "percentage",col = 1:10,cex.names = 0.7)
windows()
# Emotions
anger<-print(sentiment[which.max(nrc_s_v$anger)])
positive<-print(sentiment[which.max(nrc_s_v$positive)])
fear<-print(sentiment[which.max(nrc_s_v$fear)])
trust<-print(sentiment[which.max(nrc_s_v$trust)])
joy<-print(sentiment[which.max(nrc_s_v$joy)])
disgust<-print(sentiment[which.max(nrc_s_v$disgust)])
sadness<-print(sentiment[which.max(nrc_s_v$sadness)])
negative<-print(sentiment[which.max(nrc_s_v$negative)])
anticipation<-print(sentiment[which.max(nrc_s_v$anticipation)])

library(wordcloud2)
dtm<-DocumentTermMatrix(y)
tdm<-TermDocumentMatrix(y)
corpus.dtm.frequent<-removeSparseTerms(tdm,0.99)
corpus.dtm.frequent
tdm<-as.matrix(tdm)
length(tdm)
str(tdm)
dim(tdm)

tdm[1:20,1:10]

Rev<-rowSums(tdm)
Rev
head(Rev)
min(Rev)
max(Rev)
w_sub<- subset(Rev,Rev>=50)
w_sub
barplot(w_sub,las=2,col = rainbow(10))
# Transformation
y1<-tm_map(y,removeWords,c('ever','two','also','every','never','many','joker'
                           ,'dark','film','movie','sign','one'))
y1<-tm_map(y1,stripWhitespace)
tdm1<-TermDocumentMatrix(y1)
corpus.dtm.frequent<-removeSparseTerms(tdm1,0.99)
corpus.dtm.frequent
tdm1<-as.matrix(tdm1)
length(tdm1)
str(tdm1)
dim(tdm1)

tdm1[1:20,1:10]

Rev1<-rowSums(tdm)
Rev1
head(Rev1)
min(Rev1)
max(Rev)
w_sub1<- subset(Rev1,Rev1>=50)
w_sub1
barplot(w_sub1,las=2,col = rainbow(10))

y2<-tm_map(y1,removeWords,c('bit','may','now','makes','going','want','cant','batman','review'))
y2<-tm_map(y2,stripWhitespace)
tdm2<-TermDocumentMatrix(y2)
corpus.dtm.frequent<-removeSparseTerms(tdm2,0.99)
corpus.dtm.frequent
tdm2<-as.matrix(tdm2)
length(tdm2)
str(tdm2)
dim(tdm2)

tdm2[1:20,1:10]

Rev2<-rowSums(tdm2)
Rev2
head(Rev2)
min(Rev2)
max(Rev2)
w_sub2<- subset(Rev2,Rev2>=50)
w_sub2
barplot(w_sub2,las=2,col = rainbow(10))

# word cloud
windows()
wordcloud(y2,scale = c(4,0.5),min.freq = 30,colors = rainbow(20))
