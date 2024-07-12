from functools import lru_cache
import asyncio
import glob
import json
import os
import time

import numpy as np

from .evaluators import Evaluator
from .providers import ModelProvider

from asyncio import Semaphore
from datetime import datetime, timezone


color_to_code = {
    'red': '\033[91m',
    'green': '\033[92m',
    'yellow': '\033[93m',
    'blue': '\033[94m',
    'purple': '\033[95m',
    'orange': '\033[38;5;208m',
    'cyan': '\033[96m',
    'white': '\033[97m',
    'black': '\033[98m',
    'lightgrey': '\033[37m',
    'bold': '\033[1m',
    'underline': '\033[4m',
    'end': '\033[0m'
}

def cprint(text, color='bold'):
    if not color in color_to_code:
        raise ValueError(f"Color {color} not supported.")
    print(f"{color_to_code[color]}{text}{color_to_code['end']}")
    
def get_timestamp():
    return datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

def clog(text, color='bold'):
    cprint(f"[{get_timestamp()}] {text}", color)


class LLMNeedleHaystackTester:
    """
    This class is used to test the LLM Needle Haystack.
    """
    def __init__(self,
                 model_to_test: ModelProvider = None,
                 evaluator: Evaluator = None,
                 needle = None,
                 haystack_dir = "PaulGrahamEssays",
                 retrieval_question = None,
                 results_version = 1,
                 context_lengths_min = 1000,
                 context_lengths_max = 16000,
                 context_lengths_num_intervals = 35,
                 context_lengths = None,
                 document_depth_percent_min = 0,
                 document_depth_percent_max = 100,
                 document_depth_percent_intervals = 35,
                 document_depth_percents = None,
                 document_depth_percent_interval_type = "linear",
                 num_concurrent_requests = 1,
                 save_results = True,
                 results_dir = 'results/',
                 save_contexts = True,
                 final_context_length_buffer = 200,
                 seconds_to_sleep_between_completions = None,
                 print_ongoing_status = True,
                 **kwargs):
        """
        :model_to_test: The model to test. Default is None.
        :evaluator: An evaluator to evaluate the model's response. Default is None.
        :param needle: The needle to be found in the haystack. Default is None.
        :param haystack_dir: The directory of text files to use as background context (or a haystack) in which the needle is to be found. Default is Paul Graham Essays.
        :param retrieval_question: The question which with to prompt the model to do the retrieval.
        :param results_version: In case you would like to try the same combination of model, context length, and depth % multiple times, change the results version other than 1
        :param num_concurrent_requests: Due to volume, this object is set up to run concurrent requests, default = 1. Be careful of rate limits.
        :param save_results: Whether or not you would like to save your contexts to file. Warning: These will get long! Default = True
        :param results_dir: The directory to save the results. Default is 'results/'.
        :param save_contexts: Whether or not you would like to save your contexts to file. Warning: These will get long! Default is True.
        :param final_context_length_buffer: The amount of cushion you'd like to leave off the input context to allow for the output context. Default 200 tokens
        :param context_lengths_min: The minimum length of the context. Default is 1000.
        :param context_lengths_max: The maximum length of the context. Default is 200000.
        :param context_lengths_num_intervals: The number of intervals for the context length. Default is 35.
        :param context_lengths: The lengths of the context. Default is None.
        :param document_depth_percent_min: The minimum depth percent of the document. Default is 0.
        :param document_depth_percent_max: The maximum depth percent of the document. Default is 100.
        :param document_depth_percent_intervals: The number of intervals for the document depth percent. Default is 35.
        :param document_depth_percents: The depth percentages of the document. Default is None.
        :param document_depth_percent_interval_type: The type of interval for the document depth percent. Must be either 'linear' or 'sigmoid'. Default is 'linear'.
        :param seconds_to_sleep_between_completions: The number of seconds to sleep between completions. Default is None.
        :param print_ongoing_status: Whether or not to print the ongoing status. Default is True.
        :param kwargs: Additional arguments.
        """
        if not model_to_test:
            raise ValueError("A language model must be provided to test.")
        if not needle or not haystack_dir or not retrieval_question:
            raise ValueError("Needle, haystack, and retrieval_question must be provided.")

        self.needle = needle
        self.haystack_dir = haystack_dir
        self.retrieval_question = retrieval_question
        self.results_version = results_version
        self.num_concurrent_requests = num_concurrent_requests
        self.save_results = save_results
        self.results_dir = results_dir
        self.final_context_length_buffer = final_context_length_buffer
        self.save_contexts = save_contexts
        self.seconds_to_sleep_between_completions = seconds_to_sleep_between_completions
        self.print_ongoing_status = print_ongoing_status
        self.testing_results = []

        clog(f"Results will be saved to {self.results_dir}...", "green")
        clog(f"num_concurrent_requests: {self.num_concurrent_requests}", "green")


        if context_lengths is None:
            if context_lengths_min is None or context_lengths_max is None or context_lengths_num_intervals is None:
                raise ValueError("Either context_lengths_min, context_lengths_max, context_lengths_intervals need to be filled out OR the context_lengths_list needs to be supplied.")
            else:
                self.context_lengths = np.round(np.linspace(context_lengths_min, context_lengths_max, num=context_lengths_num_intervals, endpoint=True)).astype(int)
        else:
            self.context_lengths = context_lengths

        if document_depth_percent_interval_type not in [None, "linear", "sigmoid"]:
            raise ValueError("document_depth_percent_interval_type must be either None, 'linear' or 'sigmoid'. If you'd like your own distribution give a list of ints in via document_depth_percent_intervals")

        if document_depth_percents is None:
            if document_depth_percent_min is None or document_depth_percent_max is None or document_depth_percent_intervals is None:
                raise ValueError("Either document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals need to be filled out OR the document_depth_percents needs to be supplied.")
            
            if document_depth_percent_interval_type == 'linear':
                self.document_depth_percents = np.round(np.linspace(document_depth_percent_min, document_depth_percent_max, num=document_depth_percent_intervals, endpoint=True)).astype(int)
            elif document_depth_percent_interval_type == 'sigmoid':
                self.document_depth_percents = [self.logistic(x) for x in np.linspace(document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals)]
            else:
                raise ValueError("document_depth_percent_interval_type must be either 'sigmoid' or 'linear' if document_depth_percents is None.")
        else:
            self.document_depth_percents = document_depth_percents
        
        self.model_to_test = model_to_test
        self.model_name = self.model_to_test.model_name
        
        self.evaluation_model = evaluator

    def logistic(self, x, L=100, x0=50, k=.1):
        if x in [0, 100]:
            return x
        x = -k * (x - x0)
        return np.round(L * self.sigmoid(x), 3)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    async def bound_evaluate_and_log(self, sem, context_length, depth_percent):
        async with sem:
            await self.evaluate_and_log(context_length, depth_percent)


    async def run_test(self):
        sem = Semaphore(self.num_concurrent_requests)

        # Run through each iteration of context_lengths and depths
        tasks = []
        for context_length in self.context_lengths:
            for depth_percent in self.document_depth_percents:
                task = asyncio.create_task(self.bound_evaluate_and_log(sem, context_length, depth_percent))
                tasks.append(task)

        # Wait for all tasks to complete
        await asyncio.gather(*tasks)

    async def evaluate_and_log(self, context_length, depth_percent):
        # Checks to see if you've already checked a length/percent/version.
        # This helps if the program stop running and you want to restart later
        if self.save_results and self.result_exists(context_length, depth_percent):
            return
        
        id = f"CTX {context_length} @ {depth_percent}%"
        clog(f"{id}: Starting", "red")

        # Go generate the required length context and place your needle statement in
        context = await self.generate_context(context_length, depth_percent)

        clog(f"{id}: context generated.", "lightgrey")

        # Prepare your message to send to the model you're going to evaluate
        prompt = self.model_to_test.generate_prompt(context, self.retrieval_question)

        clog(f"{id}: prompt generated.", "lightgrey")

        test_start_time = time.time()
        response = await self.model_to_test.evaluate_model(prompt)
        test_end_time = time.time()
        test_elapsed_time = test_end_time - test_start_time

        clog(f"{id}: response received.", "lightgrey")

        score = self.evaluation_model.evaluate_response(response)
        results = {
            # 'context' : context, # Uncomment this line if you'd like to save the context the model was asked to retrieve from. Warning: This will become very large.
            'model': self.model_name,
            'context_length': int(context_length),
            'depth_percent': float(depth_percent),
            'version': self.results_version,
            'needle': self.needle,
            'model_response': response,
            'score': score,
            'test_duration_seconds': test_elapsed_time,
            'test_timestamp_utc': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S%z')
        }

        self.testing_results.append(results)

        if self.print_ongoing_status:
            print(f"-- Test Summary --\nDuration: {test_elapsed_time:.1f} seconds\nContext: {context_length} tokens\nDepth: {depth_percent}%\nScore: {score}\nResponse: {response}\n")

        if self.save_contexts:
            context_filename = f'len_{context_length}_depth_{int(depth_percent * 100)}'
            context_file_location = os.path.join("contexts", self.haystack_dir, self.model_name.replace(".", "_"), context_filename)
            os.makedirs(os.path.dirname(context_file_location), exist_ok=True)
            results['file_name'] = context_file_location
            # Save the context to file for retesting
            with open(f'{context_file_location}_context.txt', 'w') as f:
                f.write(context)

        if self.save_results:
            # Save the context to file for retesting
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
            
            # Save the result to file for retesting
            results_file = f'{context_filename}_results.json'
            results_filepath = os.path.join(self.results_dir, results_file)
            with open(results_filepath, 'w') as f:
                json.dump(results, f)

        if self.seconds_to_sleep_between_completions:
            await asyncio.sleep(self.seconds_to_sleep_between_completions)

    @lru_cache
    def result_exists(self, context_length, depth_percent):
        """
        Checks to see if a result has already been evaluated or not
        """

        if not os.path.exists(self.results_dir):
            return False
        
        for filename in os.listdir(self.results_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.results_dir, filename)
                context_length_res, depth_percent_res, version_res, model_res = self.get_file_result(filepath)
                context_length_met = context_length_res == context_length
                depth_percent_met = depth_percent_res == depth_percent
                version_met = version_res == self.results_version
                model_met = model_res == self.model_name
                if context_length_met and depth_percent_met and version_met and model_met:
                    return True
        return False
    
    @lru_cache
    def get_file_result(self, filepath):
        try:
            with open(filepath, 'r') as f:
                result = json.load(f)
            return result['context_length'], result['depth_percent'], result.get('version', 1), result['model']
        except Exception as e:
            print(f"Error reading {filepath}")
            raise e

    async def generate_context(self, context_length, depth_percent):
        # Load up tiktoken so we navigate tokens more easily

        # Get your haystack dir files loaded into a string
        context = self.read_context_files()

        # Truncate the haystack dir essays to the context length you desire
        context = self.encode_and_trim(context, context_length)

        # Insert your random statement according to your depth percent
        context = self.insert_needle(context, depth_percent, context_length)

        return context
    
    def insert_needle(self, context, depth_percent, context_length):
        # ensure that no bos or eos tokens are added to needle
        tokens_needle = self.model_to_test.tokenizer.encode(self.needle, add_special_tokens=False)
        tokens_context = self.model_to_test.encode_text_to_tokens(context)

        # Reducing the context length by 150 buffer. This is to account for system message, the user question, and response.
        context_length -= self.final_context_length_buffer

        # If your context + needle are longer than the context length (which it will be), then reduce tokens from the context by the needle length
        if len(tokens_context) + len(tokens_needle) > context_length:
            tokens_context = tokens_context[:context_length - len(tokens_needle)]

        if depth_percent == 100:
            # If your depth percent is 100 (which means your needle is the last thing in the doc), throw it at the end
            tokens_new_context = tokens_context + tokens_needle
        else:
            # Go get the position (in terms of tokens) to insert your needle
            insertion_point = int(len(tokens_context) * (depth_percent / 100))

            # tokens_new_context represents the tokens before the needle
            tokens_new_context = tokens_context[:insertion_point]

            # We want to make sure that we place our needle at a sentence break so we first see what token a '.' is
            # period_tokens = self.model_to_test.encode_text_to_tokens('.')
            
            """NOTE: end-of-word periods are tokenized separately, so we need to account for them. o/w could end up at beginning"""
            single_period_token_id = self.model_to_test.tokenizer.encode(".", add_special_tokens=False)
            end_of_word_period_token_id = self.model_to_test.tokenizer.encode("word.", add_special_tokens=False)[-1]
            period_tokens = [single_period_token_id, end_of_word_period_token_id]
            
            # Then we iterate backwards until we find the first period
            while tokens_new_context and tokens_new_context[-1] not in period_tokens:
                insertion_point -= 1
                tokens_new_context = tokens_context[:insertion_point]

            # Once we get there, then add in your needle, and stick the rest of your context in on the other end.
            # Now we have a needle in a haystack
            tokens_new_context += tokens_needle + tokens_context[insertion_point:]

        # Convert back to a string and return it
        new_context = self.model_to_test.decode_tokens(tokens_new_context)
        return new_context

    def get_context_length_in_tokens(self, context):
        return len(self.model_to_test.encode_text_to_tokens(context))

    @lru_cache
    def read_context_files(self):
        context = ""
        max_context_length = max(self.context_lengths)
        base_dir = os.path.abspath(os.path.dirname(__file__))  # Package directory

        while self.get_context_length_in_tokens(context) < max_context_length:
            for file in glob.glob(os.path.join(base_dir, self.haystack_dir, "*.txt")):
                with open(file, 'r') as f:
                    context += f.read()
        return context

    @lru_cache
    def encode_and_trim(self, context, context_length):
        tokens = self.model_to_test.encode_text_to_tokens(context)
        
        # add the buffer to the context length
        context_length += self.final_context_length_buffer

        if len(tokens) > context_length:
            context = self.model_to_test.decode_tokens(tokens, context_length)
        return context
    
    def get_results(self):
        return self.testing_results
    
    def print_start_test_summary(self):
        print ("\n")
        print ("Starting Needle In A Haystack Testing...")
        print (f"- Model: {self.model_name}")
        print (f"- Context Lengths: {len(self.context_lengths)}, Min: {min(self.context_lengths)}, Max: {max(self.context_lengths)}")
        print (f"- Document Depths: {len(self.document_depth_percents)}, Min: {min(self.document_depth_percents)}%, Max: {max(self.document_depth_percents)}%")
        print (f"- Needle: {self.needle.strip()}")
        print ("\n\n")

    def start_test(self):
        if self.print_ongoing_status:
            self.print_start_test_summary()
        asyncio.run(self.run_test())