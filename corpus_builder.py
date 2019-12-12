from os import listdir
from os import mkdir
from os import stat
from os.path import isfile
import random
import re
from nltk.tokenize import sent_tokenize

num_docs = 100
min_doc_len = 25

def get_n_authors(files, n):
    authors = set([])
    author_docs = {}
    print(len(files))
    while len(authors) < n:
        test_file = random.choice(files)
        print("trying: " + test_file)
        tf = open(test_file, "r")
        try:
            sentences = get_sentence_per_line("".join(tf.readlines()[1:]))
            if len(sentences) >= num_docs:
                print("getting n random sentences")
                author_docs[test_file] = get_n_lines(sentences, num_docs)
                authors.add(test_file)
                print("[" + str(len(authors)) + "] added " + test_file)
        except:
            print("decoding error?")
        tf.close()
        
    return list(authors), author_docs

def get_sentence_per_line(full_text):
    print("parsing sentences from text")
    full_text = full_text.replace("\n", " ")
    full_text = re.sub(r'([a-zA-Z])\.([a-zA-Z])', r'\1. \2', full_text)
    sentences = sent_tokenize(full_text)
    filtered = [ s for s in sentences if len(s) >= min_doc_len ]
    return filtered

def get_n_lines(lines, n):
    random_docs = []
    copy_of_lines = lines.copy()
    while len(random_docs) < num_docs:
        line = random.choice(copy_of_lines)
        random_docs.append(line)
        copy_of_lines.remove(line)
    return random_docs

def write_newlines(authors, author_docs, dir_path, L):
    print("chose " + str(L) + " authors randomly")
    idn = 0
    for author in authors:
        print("writing lines for author_" + str(idn))
        full_dir = "ISG/" + dir_path + "/" + str(L) + "/"
        try:
            stat(full_dir)
        except:
            mkdir(full_dir)
            
        f = open(full_dir + "author_" + str(idn) + ".txt", "w")
        for line in author_docs[author]:
            f.write(line)
            f.write("\n")
        f.close()
        
        idn += 1

    print("done writing author lines")

ff_files = [ "Corpora/fanfiction.net/WebScraping/hp_files/" + str(f) for f in listdir("Corpora/fanfiction.net/WebScraping/hp_files/") ]
ff_files + [ "Corpora/fanfiction.net/WebScraping/st_files" + str(f) for f in listdir("Corpora/fanfiction.net/WebScraping/st_files/") ]

wp_files = [ "Corpora/writingprompts/" + str(f)  for f in listdir("Corpora/writingprompts/") if isfile("Corpora/writingprompts/" + str(f)) ]
blog_files = [ "Corpora/blogs(parsed)/" + str(f)  for f in listdir("Corpora/blogs(parsed)/") ]
gb_files = [ "Corpora/gutenberg/" + str(f) for f in listdir("Corpora/gutenberg/") if f != "overview.txt" ]


for L in [3, 5, 10]: # [3, 5, 10]:
    gb_authors, gb_docs = get_n_authors(gb_files, L)
    print("got " + str(L) + " authors...")
    write_newlines(gb_authors, gb_docs, "corpus_gb", L)
    print("done gb author lines")

for L in [3, 5, 10]: #[3, 10, 50]:
    try:
        wp_authors, wp_docs = get_n_authors(wp_files, L)
        write_newlines(wp_authors, wp_docs, "corpus_wp", L)

        hp_authors, hp_docs = get_n_authors(ff_files, L)
        write_newlines(hp_authors, hp_docs, "corpus_hp", L)

        blog_authors, blog_docs = get_n_authors(blog_files, L)
        write_newlines(blog_authors, blog_docs, "corpus_blog", L)

    except Exception as e:
        print(e)
        print("error in loop " + str(L))
        
    print("done")
