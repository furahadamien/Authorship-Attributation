import os

directory = os.fsencode("blogs")
length_cnt = {}
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    
    source_file = open("blogs/" + filename, "r", errors='ignore')
    author = filename.split(".")[0] 
    output_file = open("blogs(parsed)/" + author + ".txt", "w")

    if author not in length_cnt:
        length_cnt[author] = 0
    
    raw = source_file.readlines()
    write_line = False
    for line in raw:
        try:
            line.encode('ascii', errors='ignore').decode('ascii')
            line = line.strip()

            if line == "<post>":
                write_line = True
                continue
            if line == "</post>":
                write_line = False
                continue

            if write_line and len(line) > 0:
                output_file.write(line)
                output_file.write("\n")
                length_cnt[filename.split(".")[0]] += len(line)
        except Exception as e:
            print(e)
            continue

    source_file.close()
    output_file.close()


f = open("blogs(parsed)/overview.txt", "w")
for name in length_cnt:
    f.write(name)
    f.write("\t")
    f.write(str(length_cnt[name]))
    f.write("\n")
f.close()
