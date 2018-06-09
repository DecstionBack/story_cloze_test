import csv
import re
import numpy as np
import string
import random

COMMA = ","
class Story:
	def __init__(self, context, ending):
		self.context = context
		self.ending  = ending
		self.fake_endings = []

	def add_fake_ending(self, fake_end):
		self.fake_endings.append(fake_end)

	def context_for_csv(self):
		split_context_pre = re.split(r"(\.|\!|\?)",self.context)
		if len(split_context_pre) % 2 == 1:
			split_context_pre = split_context_pre[:-1]

		split_context = []
		for i in range(4):
			split_context.append(split_context_pre[i*2] + split_context_pre[i*2 + 1])

		for i in range(len(split_context)):
			if "," in split_context[i]:
				split_context[i] = "\"" + split_context[i] + "\""
		return(COMMA.join(split_context)+",")

	def ending_for_csv(self):
		if "," in self.ending:
			return "\"" + self.ending + "\""
		return self.ending

	def fake_ending_for_csv(self, index):
		if "," in self.fake_endings[index]:
			return "\"" + self.fake_endings[index] + "\""
		return self.fake_endings[index]

	def __str__(self):
		ret = "Context: " + str(self.context) + "\nEnding: " +  str(self.ending)+ "\nFake Endings:"
		for i in range(len(self.fake_endings)):
			ret += "\n" + str(i) + ") " + self.fake_endings[i]
		return(ret)

all_names_to_gender = {}
def intake_names(names_file):
	all_names    = open(names_file,'r')
	for i in range(32469):
		name_stats = all_names.readline().split(",")
		if name_stats[0] in all_names_to_gender:
			if all_names_to_gender[name_stats[0]][1] < int(name_stats[2]): 
				all_names_to_gender[name_stats[0]] = (name_stats[1],int(name_stats[2]))
		else:
			all_names_to_gender[name_stats[0]] = (name_stats[1],int(name_stats[2]))

#replace names in the sentences with tags 
def replace_actors(sentence, gender_matters=True):
	names_seen   = []
	tokens = sentence.split(' ')
	for t in tokens:
		punc_process = ''
		for c in t:
			if c in string.punctuation:
				break
			punc_process = punc_process + c

		if punc_process in all_names_to_gender:
			names_seen.append(punc_process)

	s_test = sentence
	names_seen = sorted(names_seen,key=len,reverse=True)

	num_m = 0
	num_f = 0
	for i in range(len(names_seen)):
		act_gender = all_names_to_gender[names_seen[i]][0]
		if gender_matters:
			if act_gender == 'F':
				s_test = s_test.replace(names_seen[i],"<"+act_gender+str(num_f)+">")
				num_f += 1
			else:
				s_test = s_test.replace(names_seen[i],"<"+act_gender+str(num_m)+">")
				num_m += 1
		else: #if don't care about gender can just use this androgenous person tag
			s_test = s_test.replace(names_seen[i],"<A"+str(i)+">")

	return s_test

# search strings for finding gendered pronouns or the Gendered makers
male_re = re.compile("(^|\s)(he|his|him|<M[0-9]>)(\.|\s|\'|\?|\!)",re.IGNORECASE)
female_re = re.compile("(^|\s)(she|her|hers|<F[0-9]>)(\.|\s|\'|\?|\!)",re.IGNORECASE)
def pronoun_search(sentence):
	male_pronoun_found   = False if male_re.search(sentence) == None else True
	female_pronoun_found = False if female_re.search(sentence) == None else True
	if male_pronoun_found and female_pronoun_found:
		return 2 #return  2 if has both gendered pronouns 
	elif male_pronoun_found:
		return 1 #return  1 if just has male pronouns
	elif female_pronoun_found:
		return 0 #return  0 if just has female pronouns
	else:
		return 3 #return  3 if has no gendered pronouns 

j = " "
p = "."
stories = []
def process_stories(file_name):
	with open(file_name, 'r') as short:
		short_reader = csv.reader(short, delimiter=",")
		next(short_reader, None)

		for i in short_reader:
			s_act_replace = replace_actors(j.join(i[2:]))
			if s_act_replace.count(".") != 4:
				continue
			GT_sent = s_act_replace.split(p) 
			if len(GT_sent) > 5:
				GT_sent = GT_sent[:-1]
			context = p.join(GT_sent[:-1])+p
			GT_end  = GT_sent[-1]+p

			stories.append(Story(context, GT_end))

			Fake_end = replace_actors(i[-1])

			gendered_contexts[pronoun_search(context)].append(context)
			gendered_endings[pronoun_search(Fake_end)].append(Fake_end)

gendered_contexts = []
gendered_endings = []
for i in range(4):
	gendered_endings.append([])
	gendered_contexts.append([])
def gen_fake_endings():
	for story in stories:
		for i in range(5):
			gendered_end = pronoun_search(story.context) 
			story.add_fake_ending(random.choice(gendered_endings[gendered_end]))
	return

#1 is ground truth
#0 is fake ending
def write_fake_stories_to_file(file_to_write):
	story_output = open(file_to_write, "w")
	story_output.write("sentence1, sentence2, sentence3, sentence4, sentence5, is_real_ending\n")
	for story in stories:
		story_output.write(story.context_for_csv()+" ")
		story_output.write(story.ending_for_csv()+",1\n")
		for i in range(len(story.fake_endings)):
			story_output.write(story.context_for_csv()+" ")
			story_output.write(story.fake_ending_for_csv(i)+",0\n")
	print("Wrote stories to: " + file_to_write)

test = False
if not test:
	# this data came from: https://www.ssa.gov/oact/babynames/limits.html
	intake_names('../data/yob2017.txt')

	## these data sets came from: http://cs.rochester.edu/nlp/rocstories/
	print("Processing Spring 2016")
	process_stories('../data/ROCStories__spring2016 - ROCStories_spring2016.csv')
	print("Processing Winter 2017")
	process_stories('../data/ROCStories_winter2017 - ROCStories_winter2017.csv')
	print("Finished Pre-processing all data")

	gen_fake_endings()
	print("Generated Fake Sentences")

	write_fake_stories_to_file("../data/storydata_all.csv")

else:
	test_c = "Hi, my name is Jerry. I like to eat lemmons! Sally what do you think? I enjoy eating cherries.\n"
	test_e = "I think lemmons are gross."
	test_story = Story(test_c,test_e)
	print(test_story.context_for_csv())
