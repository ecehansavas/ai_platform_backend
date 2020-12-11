library(rJava)
library(proxy)
library(mclust)
library(streamMOA)
library(stream)


k=c(10)

xfname=c("/home/ecehan/workspace/ai_platform_backend/streams/stream1_X.txt")

lfname=c("/home/ecehan/workspace/ai_platform_backend/streams/stream1_labels.txt")

stream_names = c("Stream 1")

data_length = 50000

# datayi algoritmaya kacar kacar verecegi
part_size = 1000
part_start_indexes = seq(1, (data_length-part_size+1), by=part_size)

window_length = 100

Sys.time()

X = read.table(xfname, sep=",")
lbls = read.table(lfname)
lbls = t(lbls)
streammem = DSD_Memory(x=X, class=lbls, description="memo desc", loop=TRUE)
streamkm = DSC_StreamKM(sizeCoreset=10000, numClusters=k, length=data_length)
update(streamkm, streammem,5)



