# https://stackoverflow.com/questions/59759522/mobilebert-from-tensorflow-lite-in-python


import numpy as np
import tensorflow as tf
from bert_tokenization import FullTokenizer
import numpy as np
import math
import time

class MobileBERT:
	def __init__(self):
		self.max_length = 384
		self.interpreter = tf.lite.Interpreter(model_path="mobilebert_float_20191023.tflite")
		self.tokenizer = FullTokenizer("vocab.txt", True)
		self.interpreter.allocate_tensors()
		self.input_details = self.interpreter.get_input_details()
		self.output_details = self.interpreter.get_output_details()

	def get_summary(self):
		print("Inputs:",self.input_details,"\nOutputs:",self.output_details)

	def get_masks(self,tokens):
		if len(tokens)>self.max_length:
			raise IndexError("Token length more than max seq length!")
		return np.asarray([1]*len(tokens) + [0] * (self.max_length - len(tokens)))


	def get_segments(self,tokens):
		if len(tokens)>self.max_length:
			raise IndexError("Token length more than max seq length!")
		segments = []
		current_segment_id = 0
		for token in tokens:
			segments.append(current_segment_id)
			if token == "[SEP]":
				current_segment_id = 1
		return np.asarray(segments + [0] * (self.max_length - len(tokens)))


	def get_ids(self,tokens):
		token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
		input_ids = token_ids + [0] * (self.max_length-len(token_ids))
		return np.asarray(input_ids)

	def compile_text(self,text):
		text = text.lower().replace("-"," ")
		return ["[CLS]"] + self.tokenizer.tokenize(text) + ["[SEP]"]

	def run(self,query,context):
		stokens =  self.compile_text(query) + self.compile_text(context)
		if len(stokens)>self.max_length:
			raise IndexError("Token length more than max seq length!")
			print("Max exceeded")
		# input_ids = tf.dtypes.cast(self.get_ids(stokens),tf.int32)
		input_ids = self.get_ids(stokens).astype('int32')
		# input_masks = tf.dtypes.cast(self.get_masks(stokens),tf.int32)
		input_masks = self.get_masks(stokens).astype('int32')
		# input_segments = tf.dtypes.cast(self.get_segments(stokens),tf.int32)
		input_segments = self.get_segments(stokens).astype('int32')

		self.interpreter.set_tensor(self.input_details[0]['index'], [input_ids])
		self.interpreter.set_tensor(self.input_details[1]['index'], [input_masks])
		self.interpreter.set_tensor(self.input_details[2]['index'], [input_segments])
		sTime = time.time()
		with tf.device('/CPU:0'):
			self.interpreter.invoke()
		print(time.time()-sTime)
		end_logits = self.interpreter.get_tensor(self.output_details[0]['index'])
		start_logits = self.interpreter.get_tensor(self.output_details[1]['index'])

		# end = tf.argmax(end_logits,output_type=tf.dtypes.int32).numpy()[0]
		end = np.argmax(end_logits)
		# start = tf.argmax(start_logits,output_type=tf.dtypes.int32).numpy()[0]
		start = np.argmax(start_logits)

		print("start", start, "end", end)
		answers = " ".join(stokens[start:end+1]).replace("[CLS]","").replace("[SEP]","").replace(" ##","")
		return answers
		

	def square_rooted(self,x):
		return math.sqrt(sum([a*a for a in x]))


	def cosine_similarity(self,x,y):
		numerator = sum(a*b for a,b in zip(x,y))
		denominator = square_rooted(x)*square_rooted(y)
		return numerator/float(denominator)


if __name__ == "__main__":
	m = MobileBERT()
	m.get_summary()
	sTime = time.time()
	# last = m.run(
	# 		"what year was the declaration of independence signed",
	# 		"In fact, independence was formally declared on July 2, 1776, a date that John Adams believed would be “the most memorable epocha in the history of America.” On July 4, 1776, Congress approved the final text of the Declaration. It wasn't signed until August 2, 1776."
	# 	)
	last = m.run(
			"How many partially reusable launch systems were developed?",
			"A reusable launch system (RLS, or reusable launch vehicle, RLV) is a launch system which is capable of launching a payload into space more than once. This contrasts with expendable launch systems, where each launch vehicle is launched once and then discarded. No completely reusable orbital launch system has ever been created. Two partially reusable launch systems were developed, the Space Shuttle and Falcon 9. The Space Shuttle was partially reusable: the orbiter (which included the Space Shuttle main engines and the Orbital Maneuvering System engines), and the two solid rocket boosters were reused after several months of refitting work for each launch. The external tank was discarded after each flight."
		)
	print(time.time() - sTime," seconds")
	print("Answer", last)

