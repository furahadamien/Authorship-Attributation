import praw
import re
from nltk.tokenize import sent_tokenize

reddit = praw.Reddit(client_id='JkBvxUR1U_PzJQ', client_secret='XJaFpaACXwZZUm5rA94l_l1I97I', user_agent='NLP-A2')

post_stream = reddit.subreddit('writingprompts').top('all')
count = 0
post_count = 0
index = 0
authors = {}
story_cnt = {}
length_cnt = {}

# grab submissions from the subreddit sorted by 'top of all time', up to a fixed limit on number of sentences
for post_id in post_stream:
    if len(authors) > 10000000:
        break

    # every 1000 authors, overwrite the overview.txt with updated info
    if (post_count % 10 == 0):
        f = open("writingprompts/overview.txt", "w")
        for name in authors:
            f.write(name)
            f.write("\t")
            f.write(str(story_cnt[name]))
            f.write("\t")
            f.write(str(length_cnt[name]))
            f.write("\n")
        f.close()

    
    submission = reddit.submission(id=post_id)
    
    # skip posts that are not tagged as [WP]
    if "[WP]" not in submission.title:
        continue
    
    # weird characters in post title or comment will cause an error - just skip those posts entirely
    try:
        print(submission.title)
        submission.comments.replace_more(limit=0)
        for top_level in submission.comments:
            # skip the auto-mod comment that's present on every WP post; skip comments without authors (user account deleted)
            if "Off-Topic Discussion" not in top_level.body and top_level.author is not None:
                if (top_level.author.name not in authors):
                    authors[top_level.author.name] = index
                    story_cnt[top_level.author.name] = 1
                    length_cnt[top_level.author.name] = 0
                    f = open("writingprompts/author_" + str(index) + ".txt", "w")
                    f.write(top_level.author.name)
                    f.write("\n") # first line in file
                    index += 1
                else:
                    f = open("writingprompts/author_" + str(authors[top_level.author.name]) + ".txt", "a")
                    f.write("\n##########\n")
                    story_cnt[top_level.author.name] += 1

                # 0. use nltk to tokenize the processed body into sentences
                sentences = sent_tokenize(top_level.body)
                        
                for s in sentences:
                    # 0. remove all non-ASCII characters
                    s = s.encode('ascii', errors='ignore').decode('ascii')
                    # 1. convert comment body to lowercase
                    # s = s.lower()
                    # 2. remove any hyperlinks (common for WP to link author subreddit)
                    s = re.sub('\[.+?\]\(http.+?\)', '', s)
                    # reddit automatically creates a hyperlink for linked subreddits so remove those, too
                    s = re.sub('[\s|\b]\/r\/.+?[\s|\b]', '', s)
                    # 3b. replace all characters not in symbol set with blank ''
                    # s = re.sub('[^a-z., ]', '', s)
                    # remove any strings of repeated periods, replace with a single period
                    # s = re.sub('[.]+', '.', s)
                    # 4. store in output file with one sentence per line
                    s = s.strip()
                    # if the line only contained white space, don't bother including it in sample
                    if len(s) > 0:
                        f.write(s)
                        length_cnt[top_level.author.name] += len(s)
                        
                count+=1 # only increment the count for valid comments (comments that are parsed and recorded successfully)
                f.close()
            
    except Exception as e:
        print("EXCEPTION:")
        print(e)
        pass

    post_count += 1

f = open("writingprompts/overview.txt", "w")
for name in story_cnt:
    f.write(name)
    f.write("\t")
    f.write(str(story_cnt[name]))
    f.write("\t")
    f.write(str(length_cnt[name]))
    f.write("\n")
    
f.close()
