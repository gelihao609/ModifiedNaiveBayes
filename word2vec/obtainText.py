#'title' denotes the exact title of the article to be fetched
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

title = "Machine learning"
from wikipedia import page
wikipage = page(title)
print(wikipage.content)
text_file = open("Output.txt", "w")
with open("Output.txt", "w") as text_file:
    text_file.write(wikipage.content)