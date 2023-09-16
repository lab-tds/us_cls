library(Boruta)

us <- readRDS('./data/US_ret_pros_2022_DEC_01.rds')
colnames(us)
us_ia <- us[, c(1:10,18:22)]
head(us_ia)
df_f <- us_ia[complete.cases(us_ia), ]
dim(df_f)
head(df_f)
?Boruta()
head(df_f[,-c(4)])
df_f$study <- as.character(df_f$study)
df_f$study[df_f$study == 'prospective'] <- '0'
df_f$study[df_f$study == 'retrospective'] <- '1'
table(df_f$study)
boruta.us <- Boruta(result ~., data = df_f[,-c(4)], doTrace = 2)
print(boruta.us)
# saveRDS(boruta.us, file = './data/Boruta_analysis.rds')
boruta.us <- readRDS('./data/Boruta_analysis.rds')
colnames(boruta.us$ImpHistory)[5] <- 'ri'

png('boruta_fixed.png', width = 8, height = 8, units = 'in', res = 300)
par(mar = c(7.1, 4.1, 2.1, 2.1))
plot(boruta.us, xlab = "", xaxt = "n")
lz<-lapply(1:ncol(boruta.us$ImpHistory),function(i)
  boruta.us$ImpHistory[is.finite(boruta.us$ImpHistory[,i]),i])
names(lz) <- colnames(boruta.us$ImpHistory)
Labels <- sort(sapply(lz,median))
axis(side = 1,las=2,labels = names(Labels),
     at = 1:ncol(boruta.us$ImpHistory), cex.axis = 1.17)
dev.off()

getSelectedAttributes(boruta.us, withTentative = F)
boruta.us_df <- attStats(boruta.us)
print(boruta.us_df)
colnames(us)
us_f <- us[,-c(11:20)]
max(us_f$size, na.rm = TRUE)
us_f$size[us_f$size == 70] <- 7
