# Loading important libraries
library(rvest)
library(XML)
library(magrittr)
library(tm)

# loading url
url<-"https://www.imdb.com/search/title/?title_type=feature&groups=top_100&sort=runtime,asc"
#reading HTML url from website
webpage<-read_html(url)
#scraping rank 
rank_data_imdb<- html_nodes(webpage, '.text-primary')
#Converting into text
ranks<- html_text(rank_data_imdb)
head(ranks)
# data preprocessing: converting text data into numeric 
ranks<-as.numeric(ranks)
head(ranks)
# Scraping titles
title_data<-html_nodes(webpage,'.lister-item-header a')
titles<-html_text(title_data)
head(titles)
#scraping description
description_data<-html_nodes(webpage,'.ratings-bar+.text-muted')
descriptions<-html_text(description_data)
head(descriptions)
descriptions<- gsub('\n',"",descriptions)
head(descriptions)
#Scarping runtime
run_time_data<-html_nodes(webpage,'.text-muted .runtime')
run_time<-html_text(run_time_data)
head(run_time)
run_time<-gsub('min','',run_time)
run_time<-as.numeric(run_time)
head(run_time)
genre_data<-html_nodes(webpage,'.genre')
genre<-html_text(genre_data)
genre<-gsub('\n','',genre)
genre<-gsub(" ","",genre)
#taking only first genre for each movie
genre<-gsub(",.*","",genre)
# Converting each genre into factor.
genre<-as.factor(genre)
head(genre)
rating_data<-html_nodes(webpage,'.ratings-imdb-rating strong')
ratings<-html_text(rating_data)
ratings<-as.numeric(ratings)
head(ratings)
# Scraping votes
votes_data<-html_nodes(webpage,'.sort-num_votes-visible span:nth-child(2)')
votes<-html_text(votes_data)
votes<-gsub(",","",votes)
votes<-as.numeric(votes)
head(votes)
# scraping directors name
directors_data<-html_nodes(webpage,'.text-muted+ p a:nth-child(1)')
directors<-html_text(directors_data)
directors<-as.factor(directors)
head(directors)
#scraping starcast name
star_cast_data<-html_nodes(webpage,'.lister-item-content .ghost+ a')
star_cast<-html_text(star_cast_data)
star_cast<-as.factor(star_cast)
head(star_cast)
#scraping metacsore
metascores<-html_nodes(webpage,'.metascore')
metascores<-html_text(metascores)
metascores<-as.numeric(metascores)
length(metascores)
head(metascores)
for (i in c(6,20,27,31,43)){
  
  a<-metascores[1:(i-1)]
  
  b<-metascores[i:length(metascores)]
  
  metascores<-append(a,list("NA"))
  
  metascores<-append(metascores,b)
}
metascores<-as.numeric(metascores)
length(metascores)
summary(metascores)
# scraping gross revenue
gross_data <- html_nodes(webpage,'.ghost~ .text-muted+ span')

#Converting the gross revenue data to text
gross_revenue <- html_text(gross_data)
gross_revenue<-gsub("M","",gross_revenue)
gross_revenue<-substring(gross_revenue,2,6)#  removing '$' sign. 
length(gross_revenue)
head(gross_revenue)
for (i in c(4,7,20,49)){
  a<-gross_revenue[1:(i-1)]
  b<-gross_revenue[i:length(gross_revenue)]
  gross_revenue<-append(a,list("NA"))
  gross_revenue<-append(gross_revenue,b)
}
gross_revenue<-as.numeric(gross_revenue)
length(gross_revenue)
summary(gross_revenue)
head(gross_revenue)
movie_df<- data.frame(Rank=ranks,Metascores=metascores,Ratings=ratings,Titles=titles,Genre=genre,Votes=votes,Descriptions=descriptions,Runtime=run_time,Directors=directors,Starcast=star_cast,GrossRevenue=gross_revenue)
attach(movie_df)
#install.packages("ggplot2")
library(ggplot2)
qplot(data = movie_df,Runtime,fill=Genre,bins=30)
windows()
ggplot(movie_df,aes(x=Runtime,y=Ratings))+geom_point(aes(size=Votes,col=Genre))
ggplot(movie_df,aes(x=Runtime,y=GrossRevenue))+geom_point(aes(size=Votes,col=Genre))
                                                    
