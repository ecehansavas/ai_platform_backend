library(rJava)
library(proxy)
library(stream)
library(streamMOA)
library(mclust)

ks=c(10,10,10,20,4,10,10,10)

x_filenames=c("/home/alaettin/streams/stream1.txt_X.txt",
              "/home/alaettin/streams/stream2.txt_X.txt",
              "/home/alaettin/streams/stream3.txt_X.txt",
              "/home/alaettin/streams/stream4.txt_X.txt",
              "/home/alaettin/streams/stream5.txt_X.txt",
              "/home/alaettin/streams/stream6.txt_X.txt",
              "/home/alaettin/streams/stream7.txt_X.txt",
              "/home/alaettin/streams/stream8.txt_X.txt" )

l_filenames=c("/home/alaettin/streams/stream1.txt_labels.txt",
              "/home/alaettin/streams/stream2.txt_labels.txt",
              "/home/alaettin/streams/stream3.txt_labels.txt",
              "/home/alaettin/streams/stream4.txt_labels.txt",
              "/home/alaettin/streams/stream5.txt_labels.txt",
              "/home/alaettin/streams/stream6.txt_labels.txt",
              "/home/alaettin/streams/stream7.txt_labels.txt",
              "/home/alaettin/streams/stream8.txt_labels.txt" )

stream_names = c("Stream 1",
                 "Stream 2",
                 "Stream 3",
                 "Stream 4",
                 "Stream 5",
                 "Stream 6",
                 "Stream 7",
                 "Stream 8"  )


cat("========== ^ ==========\n")
cat("New run of DenStream algorithm PART by PART at :")
Sys.time()
cat("---\n")

data_length = 50000

#part_size = 1000
part_size = 100
part_start_indexes = seq(1, (data_length-part_size+1), by=part_size)

for(i in 1:length(ks))
#for(i in c(3))
{
  k = ks[i]
  xfname = x_filenames[i]
  lfname = l_filenames[i]
  
  X = read.table(xfname, sep=",")
  lbls = read.table(lfname)
  lbls = t(lbls)
  streammem = DSD_Memory(x=X, class=lbls, description="memo desc", loop=TRUE)
  denstream = DSC_DenStream(epsilon=0.05, k=k)

  all_ass = c()
  
  reset_stream(streammem, pos = 1)
  
  #for(cnt in cnts)
  total_time = 0
  aris = c()
  for(si in part_start_indexes)
  {
    begin = Sys.time()
    #if(si == 1)
    #{
    #  update(denstream, streammem, part_size)
    #}
    update(denstream, streammem, part_size)
    ass = get_assignment(denstream, tail(head(X, si+part_size-1), part_size), type = "macro")
    ass[is.na(ass)] = -1
    end = Sys.time()
    this_time = end - begin
    total_time = total_time + this_time
    ari = adjustedRandIndex(ass, tail(head(t(lbls), si+part_size-1), part_size))
    all_ass = c(all_ass, ass)
    aris = c(aris, ari)
    cat("Stream : [", stream_names[i], "] Indexes : [", si, ":", si+part_size-1, " ] ari : [", ari, "] Execution Time : [", this_time, "] seconds.\n")
  }
  cat("Total Time of this stream : [", total_time, "] seconds, average ari : [", mean(na.omit(aris)), "]\n")
  total_ari = adjustedRandIndex(all_ass, head(t(lbls), length(all_ass)))
  cat("Total ari : [", total_ari, "]\n")
  cat("---\n")
}
cat("========== V ==========\n")
