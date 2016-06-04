""" Subclasses the twitter-text-parser for more fine-grained preprocessing """
from ttp import ttp

class NLPParser(ttp.Parser):
	def format_tag(self, tag, text):
		'''Return formatted HTML for a hashtag.'''
		return ('#' + text).encode('utf-8')

	def format_username(self, at_char, user):
		'''Return formatted HTML for a username.'''
		return '<USER>'

	def format_list(self, at_char, user, list_name):
		'''Return formatted HTML for a list.'''
		return '<LIST>'

	def format_url(self, url, text):
		'''Return formatted HTML for a url.'''
		return '<URL>'