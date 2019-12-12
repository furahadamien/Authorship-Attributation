import praw
import re
from nltk.tokenize import sent_tokenize

if len(sys.argv)== 1:
    print("Enter an argument to actually re-run the program.")
    exit(0)

reddit = praw.Reddit(client_id='JkBvxUR1U_PzJQ', client_secret='XJaFpaACXwZZUm5rA94l_l1I97I', user_agent='NLP-A2')

count = 0
post_count = 0
index = 0
authors = {}
story_cnt = {}
length_cnt = {}

subreddit_list = [ "fringly", "AliciaWrites", "XcessiveWriting", "psycho_alpaca", "Lilwa_Dexel",
                   "ScarecrowSid", "EvenAsIWrite", "nickofnight", "LeoDuhVinci", "StoryStar",
                   "FiresofFordregha", "AlannaWu", "PerilousPlatypus", "shoringupfragments",
                   "ThadsMind", "psalmsandstories", "Leebeewilly", "FoxFictions",
                   "LandOfMisfits", "MattWritinCollection", "Luna_Lovewell", "oppositewerewolf",
                   "Remyxed", "Ryter", "bluelizardK", "WeirdEmoKidStories", "penguin347",
                   "FatDragon", "Nazer_The_Lazer", "NordicNarrator", "MatiWrites" ]

authors_list = {}
authors_list["FoxFictions"] = "Cody_Fox23"
authors_list["XcessiveWriting"] = "XcseeiveSmash"
authors_list["EvenAsIWrite"] = "Shadowyugi"
authors_list["FiresofFordregha"] = "Fordregha"
authors_list["shoringupfragments"] = "ecstaticandinsatiate"
authors_list["LandOfMisfits"] = "LadyLuna21"
authors_list["Remyxed"] = "RemixPhoenix"
authors_list["Ryter"] = "Ryter99"
authors_list["psalmsandstories"] = "psalmoflament"
authors_list["WeirdEmoKidStories"] = "That2009WeirdEmoKid"
authors_list["NordicNarrator"] = "blacksponge"
authors_list["MatiWrites"] = "matig123"

for user in subreddit_list:

    post_stream = reddit.subreddit(user).top('all')

    # grab submissions from the subreddit sorted by 'top of all time', up to a fixed limit on number of sentences
    for post_id in post_stream:

        # every 10 posts, overwrite the overview.txt with updated info
        if (post_count % 10 == 0):
            f = open("writingprompts/authors/overview.txt", "w")
            for name in authors:
                f.write(name)
                f.write("\t")
                f.write(str(story_cnt[name]))
                f.write("\t")
                f.write(str(length_cnt[name]))
                f.write("\n")
            f.close()

        submission = reddit.submission(id=post_id)
        
        # weird characters in post title or comment will cause an error - just skip those posts entirely
        try:
            print(submission.title)
            if (submission.author.name not in authors):
                authors[submission.author.name] = index
                story_cnt[submission.author.name] = 1
                length_cnt[submission.author.name] = 0
                f = open("writingprompts/authors/author" + str(index) + ".txt", "w")
                f.write(submission.author.name)
                f.write("\n") # first line in file
                index += 1
            else:
                f = open("writingprompts/authors/author" + str(authors[submission.author.name]) + ".txt", "a")
                story_cnt[submission.author.name] += 1

            # 0. use nltk to tokenize the processed body into sentences
            s = submission.selftext
            
            # 0. remove all non-ASCII characters
            s.encode('ascii', errors='ignore').decode('ascii')
            # 1. remove any hyperlinks (common for WP to link author subreddit)
            s = re.sub('\[.+?\]\(http.+?\)', '', s)
            # reddit automatically creates a hyperlink for linked subreddits so remove those, too
            s = re.sub('[\s|\b]\/r\/.+?[\s|\b]', '', s)
            s = s.strip()
            # if the line only contained white space, don't bother including it in sample
            if len(s) > 0:
                f.write(s)
                length_cnt[submission.author.name] += len(s)
                    
            count+=1 # only increment the count for valid comments (comments that are parsed and recorded successfully)
            f.close()
                
        except Exception as e:
            print("EXCEPTION:")
            print(e)
            pass

        post_count += 1

    f = open("writingprompts/authors/overview.txt", "w")
    for name in story_cnt:
        f.write(name)
        f.write("\t")
        f.write(str(story_cnt[name]))
        f.write("\t")
        f.write(str(length_cnt[name]))
        f.write("\n")
        
    f.close()
