import nltk
import re
from nltk import *
from nltk.chunk import conlltags2tree, tree2conlltags
from pprint import pprint



Data= 'During inspection it was found that the POS1 ' \
      'strut is leaking and requires replacement Raise notification :' \
      ' 2 x fitters* Machine will be down*28.07.2018 11:06:18 UTC Damon Stacey (STACDA9) ' \
      'THIS JOB HAS BEEN RESCHEDULED 14.09.2018 03:24:15 UTC Scott Williams (WILLSO9) Advised by site that hub i' \
      's also leaking now Last time this corner was off a new strut was fitted but a second hand hub was all that was available at the time.' \
      ' I have added the material number for a complete as the hub and strut needs to be changed. Komatsu have been engaged ' \
      'for warranty on the strut. See attached email for photos from site. 16.10.2018 10:50:13 UTC Traven Hooper (HOOPT9)' \
      ' Job has been completed 16.10.2018 20:49:54 UTC Gavin Casten (CASTG91) Phone +6149805626 Work completed ahead of schedule'

#-----------------------------Part of Speech Tagging-----------------------------------
def preprocess(sent):
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)
    return sent
Tagged= preprocess(Data)
print(Tagged)

#-------------------------Pattern for Named Entity Recognition----------------------
pattern = 'NP: {<DT>?<JJ>*<NN>}'
chunkParse = nltk.RegexpParser(pattern)
Chunks = chunkParse.parse(Tagged)
#print(Chunks)
Chunks.draw()

#IOB tagging of chunks
iob_tagged = tree2conlltags(Chunks)
#pprint(iob_tagged)

NamedEntityRecognition= nltk.ne_chunk(Tagged)
NamedEntityRecognition.draw()


