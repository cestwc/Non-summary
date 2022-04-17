from datasets import load_metric

bertscore = load_metric("bertscore")

def bertscore_from_datum(e):
	e['bertscore'] = bertscore.compute(predictions=e['prediction'], references=e['reference'], lang="en", use_fast_tokenizer=True)['f1']
	if 'prediction_' in e:
		e['bertscore_'] = bertscore.compute(predictions=e['prediction_'], references=e['reference_'], lang="en", use_fast_tokenizer=True)['f1']
	return e

bs_single = lambda p, r: bertscore.compute(predictions = [p], references = [r], lang = 'en')['f1']
bs_batch = lambda p, r: bertscore.compute(predictions = p, references = r, lang = 'en')['f1']
bscore = lambda p, r: bs_single(p, r) if type(p) == str else bs_batch(p, r)

import string
table_ = str.maketrans(string.punctuation, ' '*len(string.punctuation))
remove_punct = lambda x: ' '.join(x.translate(table_).split())

# from rouge_score import rouge_scorer

# scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)
# rg_single = lambda p, r: scorer.score(r, p)
# rg_batch = lambda p, r: [scorer.score(r, p) for i in range(len(p))]
# rouge = lambda p, r: rg_single(p, r) if type(p) == str else rg_batch(p, r)
