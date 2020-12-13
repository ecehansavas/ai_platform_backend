library(rJava)
library(proxy)
library(stream)
library(streamMOA)
library(mclust)

args<-commandArgs(TRUE)

xfname = args[1]
lfname = args[2]
k = as.numeric(args[3])
epsilon = as.numeric(args[4])
part_size =  as.numeric(args[5]) # data process size
data_length = as.numeric(args[6])

#cat("New run of DenStream algorithm PART by PART at :")
#cat("---\n")
Sys.time()



part_start_indexes = seq(1, (data_length-part_size+1), by=part_size)

X = read.table(xfname, sep=",")
lbls = read.table(lfname)
lbls = t(lbls)
streammem = DSD_Memory(x=X, class=lbls, description="memo desc", loop=TRUE)
denstream = DSC_DenStream(epsilon=epsilon, k=k)

all_ass = c()

reset_stream(streammem, pos = 1)

total_time = 0
aris = c()
for(si in part_start_indexes)
{
    begin = Sys.time()
    update(denstream, streammem, part_size)
    ass = get_assignment(denstream, tail(head(X, si+part_size-1), part_size), type = "macro")
    ass[is.na(ass)] = -1
    end = Sys.time()
    this_time = end - begin
    total_time = total_time + this_time
    #cat("Found Labels: " , ass, "\n")
    #cat("Real Labels : " , tail(head(t(lbls), si+part_size-1), part_size), "\n")
    ari = adjustedRandIndex(ass, tail(head(t(lbls), si+part_size-1), part_size))
    all_ass = c(all_ass, ass)
    aris = c(aris, ari)
    # cat("Indexes : [", si, ":", si+part_size-1, " ] ari : [", ari, "] Execution Time : [", this_time, "] seconds.\n")
    cat("<ACCURACY_START>",si, ":", si+part_size-1, "datalength:", data_length, "acc", ari, "meanacc", mean(na.omit(aris)), "time", total_time,"<ACCURACY_END>\n")
}

# cat("Total Time of this stream : [", total_time, "] seconds, average ari : [", mean(na.omit(aris)), "]\n")
# total_ari = adjustedRandIndex(all_ass, head(t(lbls), length(all_ass)))
# cat("Total ari : [", total_ari, "]\n")

