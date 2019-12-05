from nltk.corpus import gutenberg

index = 0
sources = {}
story_cnt = {}
length_cnt = {}
for fileID in gutenberg.fileids():
    source = fileID.split("-")[0]
    if (source not in sources):
        sources[source] = index
        length_cnt[source] = 0
        story_cnt[source] = 1
        f = open("gutenberg/author_" + str(index) + ".txt", "w")
        f.write(source)
        index += 1
    else:
        f = open("gutenberg/author_" + str(sources[source]) + ".txt", "a")
        story_cnt[source] += 1

    text = gutenberg.raw(fileID)
    text.encode('ascii', errors='ignore').decode('ascii')
    text = text.strip()
    if (len(text) > 0):
        f.write(text)
        length_cnt[source] += len(text)
    
        
    f = open("gutenberg/overview.txt", "w")
    for name in story_cnt:
        f.write(name)
        f.write("\t")
        f.write(str(story_cnt[name]))
        f.write("\t")
        f.write(str(length_cnt[name]))
        f.write("\n")
        
    f.close()
