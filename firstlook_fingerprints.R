#pct <- c(63,13,24)


docpad <- '/Users/fhaot/Documents/Big Data/Project big data/NISTDB4-F.csv'
doc <- read.csv(docpad, header = 0)
names(doc)
length(names(doc))
nr_na_percol <- sapply(doc, function(col) length(which(is.na(col))))
plot(seq_along(nr_na_percol), nr_na_percol)
dplyr::n_distinct(nr_na_percol)
hist(nr_na_percol)
# eerste kolommen braaf dan heel blok vol na's

nr_na_perrow <- apply(doc,1, function(col) length(which(is.na(col))))
plot(seq_along(nr_na_perrow), nr_na_perrow)
dplyr::n_distinct(nr_na_perrow)
hist(nr_na_perrow)
# modus rond de 100-300 links afgeknotte zeer platte gaussiaan

yval <- doc[, length(doc)]
table(yval)
# ALRTW, geen blankost meeste rond de 380, T slechts 123
# stratified - 