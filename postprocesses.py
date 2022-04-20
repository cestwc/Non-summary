import torch
import numpy as np

def tag(e, model):
	with torch.no_grad():
		e['tags'] = model(e['input_ids'].unsqueeze(0).to(model.device)).logits.argmax(2).cpu().tolist()[0]
	return e

def bagFromLabels(src, labels):
	bag = {}
	for x, y in zip(src, labels):
		if y > 0 and x != 1:
			if x not in bag:
				bag[x] = [y]
			else:
				bag[x].append(y)
	return {x:int(np.median(ys)) for x, ys in bag.items()}

def getLongestSpan(binary_sequence):
	assert set(binary_sequence) == {0, 1}
	starts = []
	ends = []
	running = False
	for i, v in enumerate(binary_sequence):
		if v == 1 and not running:
			starts.append(i)
			running = True
		elif v == 0 and running:
			ends.append(i - 1)
			running = False
	if running:
		ends.append(i)
		running = False

	spans = [e - s + 1 for s, e in zip(starts, ends)]

	segment = spans.index(max(spans))
	return (starts[segment], ends[segment])

def bagToSeq(src_sent, bag):
	labels = [0] * len(src_sent)
	src_sent_ = src_sent.copy()

	while len(bag) > 0:
		noticeable = list(map(lambda x: int(x in bag), src_sent_))
		if sum(noticeable) == 0:
			break

		seg_start, seg_end = getLongestSpan(noticeable)
		# if seg_start - seg_end > -2:
		#     break

		for i in range(seg_start, seg_end + 1):
			labels[i] = 1
			if src_sent_[i] in bag:
				bag[src_sent_[i]] -= 1
			src_sent_[i] = 1

		bag = {k:v for k, v in bag.items() if v > 0}

	sequence = []
	running = False
	for t, v in zip(src_sent, labels):
		if v > 0:
			if not running:
				sequence.append(1437)
				sequence.append(49193)
				sequence.append(1437)
				running = True
			sequence.append(t)
		else:
			running = False

	return sequence

import nltk
nltk.download('punkt')

from nltk.tokenize import wordpunct_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
word_detokenize = TreebankWordDetokenizer().detokenize

def removeWrongUnigrams(prediction, article):
	available = set(wordpunct_tokenize(article.lower()))
	prediction = prediction.replace('$$$$', 'SEPSEPSEP')
	prediction = [x for x in wordpunct_tokenize(prediction) if (x.lower() in available or x == 'SEPSEPSEP')]
	strings = []
	temp = []
	for i, x in enumerate(prediction):
		temp.append(x)
		if x == 'SEPSEPSEP':
			if len(temp) > 1:
				strings.extend(temp)
			temp = []
		elif i == len(prediction) - 1:
			if len(temp) > 0:
				strings.extend(temp)
			temp = []

	return word_detokenize(strings).replace('SEPSEPSEP', '\n')

def postprocess(e, tokenizer, source_key = 'article'):
	summary_ids = bagToSeq(e['input_ids'], bagFromLabels(e['input_ids'], e['tags']))

	prediction = tokenizer.decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
	prediction = removeWrongUnigrams(prediction, e[source_key])
	e['prediction'] = prediction
	return e
