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
part_size = 100
part_start_indexes = seq(1, (data_length-part_size+1), by=part_size)

window_length = 100

cat("========== ^ ==========\n")
cat("New run of CluStream algorithm PART by PART at :")
Sys.time()
cat("window length (horizon) : [", window_length, "]\n")
cat("---\n")

X = read.table(xfname, sep=",")
lbls = read.table(lfname)
lbls = t(lbls)
streammem = DSD_Memory(x=X, class=lbls, description="memo desc", loop=TRUE)
clustream = DSC_CluStream(m=100, horizon=window_length, k=k)

all_ass = c()

reset_stream(streammem, pos = 1)

#for(cnt in cnts)
total_time = 0
aris = c()

for(si in part_start_indexes)
{
    begin = Sys.time()
    update(clustream, streammem, part_size)
    ass = get_assignment(clustream, tail(head(X, si+part_size-1), part_size), type = "macro")
    ass[is.na(ass)] = -
    end = Sys.time()
    this_time = end - begin
    total_time = total_time + this_time
    cat("Found Labels: " , ass, "\n")
    cat("Real Labels : " , tail(head(t(lbls), si+part_size-1), part_size), "\n")
    ari = adjustedRandIndex(ass, tail(head(t(lbls), si+part_size-1), part_size))
    all_ass = c(all_ass, ass)
    aris = c(aris, ari)
    cat("Indexes : [", si, ":", si+part_size-1, " ] ari : [", ari, "] Execution Time : [", this_time, "] seconds.\n")
}

cat("Total Time of this stream : [", total_time, "] seconds, average ari : [", mean(na.omit(aris)), "]\n")
total_ari = adjustedRandIndex(all_ass, head(t(lbls), length(all_ass)))
cat("Total ari : [", total_ari, "]\n")
cat("---\n")

cat("========== V ==========\n")
