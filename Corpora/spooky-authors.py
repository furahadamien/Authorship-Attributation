
fr = open("spooky-author-identification/train.txt", "r", errors="ignore")
lines = fr.readlines()

length_cnt = {}
for line in lines:
    fields = line.split("\t")
    for i in range(len(fields)):
        fields[i] = fields[i].replace('"', '')
        fields[i] = fields[i].strip()

    print(fields)
    
    if fields[0] == "id":
        continue
    
    author = fields[2]
    text = fields[1]

    if author not in length_cnt:
        length_cnt[author] = 0
        open("spooky-authors/" + author + ".txt", "w").close()

    fw = open("spooky-authors/" + author + ".txt", "a")
    fw.write(text)
    fw.write("\n")
    length_cnt[author] += len(text)
    
fr.close()

f = open("spooky-authors/overview.txt", "w")
for author in length_cnt:
    f.write(author)
    f.write("\t")
    f.write(str(length_cnt[author]))
    f.write("\n")

f.close()
