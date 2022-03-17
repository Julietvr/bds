require(tidyverse)
meta <- read.csv('metadata.csv')
music <- read.csv('music_data.csv')
music_sc <- scale(music)
y1 <-meta$track_listens
y2 <- as.numeric(substring(meta$album_date_released,1,4))

# 4 minuten
# belangrijk om te schalen
matsvm <- svd(music_sc)
plot(matsvm$d^2/sum(matsvm$d^2))

plot(matsvm$d[1:50]^2/sum(matsvm$d^2))

# zijn ze gecorreleerd
qr(matsvm$v[,1:12])$rank
cor1 <- lapply(1:12,function(k) sum(matsvm$u[,k]*y1)/sum(y1^2))
cor2 <- lapply(1:12,function(k) sum(matsvm$u[,k]*y2)/sum(y2^2))
sum(matsvm$d[1:100]^2)/sum(matsvm$d^2)


#na
vna <- sapply(music, function(col) length(which(is.na(col))))
#concentrations
conc <- lapply(music, function(col){
  tbcol = table(col)
  tb <- tibble(
  value= as.numeric(names(which.max(tbcol))),
  freq = max(tbcol))})
tbconc <- bind_rows(conc)%>%
  mutate(name= names(music))%>%
  filter(freq!=1)
hist(tbconc$freq)
table(tbconc$freq)

#t ziet er naar uit dat missing values al vervangen zijn door gemiddeldes
#op id na heeft elke variable minstens 90 dubbel
#-> duplicaten verwijderen op records?

# we werken best op subsamples, dat kunnen we ook verdelen
# in een regressie komt dat uit op een nest betas, 
# aggregatie: kiezen tussen gemiddelde of modus (op zijn bayes)

# wat kunnen we doen afgezien van regressie? 
# - regressieboom (adaboost)
# - kernel (polynomen tot de derde graad / smoothers)
# - pcr in combinatie met forward step
# - knn gemiddelde
# - 
##https://examples.dask.org/machine-learning/svd.html
