import wandb
import random
import numpy as np
import openai
from llm_utils import chat_completion_rl
import re
from torch.multiprocessing import Queue
import json
from copy import deepcopy
from llm_utils import get_llm_config
from llm_utils import setup_chat_rate_limiter, chat_completion_rl, get_llm_config, num_tokens_consumed_by_chat_request, get_model_max_tokens, embedding_rl, pretty_print_chat_messages
from utils.llm_tools import process_function_call_and_return_message, function_definition_list_factory, available_functions_factory, hash_messages, process_functions_into_function_names, clean_string, detect_cycles
from summarizer_and_modifier import add_line_numbers, write_files_from_dict, load_code_files_into_dict
from openai.error import InvalidRequestError
from pathlib import Path
from utils import prompts
# from log_evaluator import count_errors_from_file_dict, run_tests_from_file_dict
import pandas as pd
import traceback
import re
from utils.models import DyNODEModel, RNNModel, TransformerModel
import torch
import time
from torch import optim
import pysindy as ps
from tqdm import tqdm
from pysindy import SINDy
from pysindy.optimizers import STLSQ, SR3, SSR, FROLS
from pysindy.differentiation import FiniteDifference, SmoothedFiniteDifference, SpectralDerivative
from pysindy.feature_library import PolynomialLibrary
from utils import gp_method

torch.set_float32_matmul_precision('high')


def get_code_from_message(message):
    match = re.search(r'```python\n(.*?)\n```', message, re.DOTALL)
    if match:
        code = match.group(1)
    else:
        match = re.search(r'``` python\n(.*?)\n```', message, re.DOTALL)
        if match:
            code = match.group(1)
        else:
            match = re.search(r'```(.*?)```', message, re.DOTALL)
            if match:
                code = match.group(1)
            else:
                code = None
    return code


def extract_failed_tests(test_output):
    # Search for the failed tests count in the test output string
    match = re.search(r'(\d+) failed', test_output)
    if match:
        # Extract the number of failed tests
        failed_tests_count = int(match.group(1))
    else:
        # Default to 10 if no match is found
        failed_tests_count = 10

    return failed_tests_count

def initialize_agent(method_name, env, config, rate_limiter, wandb_project_name=None, logger=None):
    # Initialize Weights & Biases if a project name is provided
    if wandb_project_name:
        wandb.init(project=wandb_project_name, config=config)

    # Depending on the method name, initialize the agent
    if "NSDT" in method_name:
        agent = NSDT(env, config, logger, rate_limiter, method_name=method_name)
    elif method_name == "ZeroShot":
        agent = ZeroShot(env, config, logger, rate_limiter, optimize_params=False, name=method_name)
    elif method_name == "ZeroOptim":
        agent = ZeroShot(env, config, logger, rate_limiter, optimize_params=True, name=method_name)
    elif method_name == "DyNODE":
        agent = DyNODEAgent(env, config, logger, rate_limiter)
    elif method_name == "RNN":
        agent = RNNAgent(env, config, logger, rate_limiter)
    elif method_name == "Transformer":
        agent = TransformerAgent(env, config, logger, rate_limiter)
    elif method_name == "SINDY":
        agent = SINDYAgent(env, config, logger, rate_limiter)
    elif method_name == "GP":
        agent = GPAgent(env, config, logger, rate_limiter)
    else:
        raise ValueError(f"Unknown method name: {method_name}")

    return agent

class Agent:
    def __init__(self, env, config, logger, rate_limiter):
        self.env = env
        self.config = config
        self.seed_value = None
        self.logger = logger
        self.rate_limiter = rate_limiter

    def run(self, state):
        raise NotImplementedError("Agents must implement a run method.")

    def seed(self, seed_value):
        self.seed_value = seed_value
        random.seed(seed_value)
        np.random.seed(seed_value)
    
    def get_llm_config(self):
        return get_llm_config(self.config, self.logger, self.name, self.rate_limiter)

class NSDT(Agent):
    def __init__(self, env, config, logger, rate_limiter, method_name=''):
        super().__init__(env, config, logger, rate_limiter)
        self.name = method_name
        self.load_from_checkpoint = ''
        self.replay_llm_responses_path = ''
        # self.replay_llm_responses_path = 'logs/run-20240127-155404_NSDT-ZeroShot-ZeroOptim_Cancer_0_3-runs_log_MAIN_TABLE/Cancer/0/NSDT_llm_responses.json'
        self.replay_llm_responses_path_index = 0
        self.responses = []
        self.message_hash_same_increase_temperature = 0
        self.step_idx = 0
        self.max_re_tries = 30
        self.re_tries = 0
        self.max_iterations = 10
        if self.load_from_checkpoint:
            with open(self.load_from_checkpoint, 'r') as f:
                data = json.load(f)
            self.simulator_code_dict = data['simulator_code_dict']
            self.steps = data['steps']
            self.step = data['step']
            self.meta_messages = data['meta_messages']
            self.responses = data['responses']
            self.message_hash = data['message_hash']
        else:
            self.simulator_code_dict = {}
            self.steps = []
            self.step = None
            self.meta_messages = []
            self.messages = []
            self.responses = []
            self.message_hash = hash_messages([])
        
        self.folder_path = f"{self.config.run.log_path.split('.txt')[0]}/{self.env.env_name}/{self.env.seed}/"
        Path(self.folder_path).mkdir(parents=True, exist_ok=True)
        self.max_tokens = get_model_max_tokens(config)
        self.functions = function_definition_list_factory()
        self.system_message = {"role": "system", "content": prompts.system_prompt()}

    def print_dialog(self, messages, response_msg=False):
        num_tokens = num_tokens_consumed_by_chat_request(messages=messages, functions=self.functions)
        pretty_print_chat_messages(messages, num_tokens, self.max_tokens, logger=self.logger, response_msg=response_msg, step_idx=self.step_idx, total_steps=len(self.steps), max_re_tries=self.max_re_tries, re_tries=
self.re_tries)

    def save_agent_state(self, messages, beginning_step=''):
        data_to_save = {'messages': messages,
                        'simulator_code_dict': self.simulator_code_dict,
                        'steps': self.steps,
                        'step': self.step,
                        'meta_messages': self.meta_messages,
                        'messages': self.messages,
                        'message_hash': self.message_hash,
                        }
        if not beginning_step:
            path = f'{self.folder_path}current_{self.name}_state.json' 
        else:
            path = f'{self.folder_path}NeurosymbolicLLMAgent_state_beginning_step_{self.step_idx}.json'
        with open(path, 'w') as f:
            json.dump(data_to_save, f)

    def get_llm_response(self, messages, max_tokens=None, n=1, print_=True):
        if print_:
            self.print_dialog(messages)
        self.save_agent_state(messages)
        llm_config = self.get_llm_config()
        if max_tokens is not None:
            llm_config['max_tokens'] = max_tokens
        llm_config['messages'] = messages
        if n is not None:
            llm_config['n'] = n
        # Check if the messages have changed, if they have, then set temperature to zero, if still the same then set temperature to 0.1, as we are repeating ourselves.
        tmp_messages = [clean_string(str(msg)) for msg in messages]
        if detect_cycles(tmp_messages):
            self.message_hash_same_increase_temperature += 0.4
            if self.message_hash_same_increase_temperature >= 1:
                self.message_hash_same_increase_temperature = 1
            self.logger.info(f'[Increasing LLM temperature to {self.message_hash_same_increase_temperature}]')
        else:
            if self.message_hash_same_increase_temperature > 0:
                self.logger.info(f'[Annealing LLM temperature to {self.message_hash_same_increase_temperature}]')
                self.message_hash_same_increase_temperature -= 0.1
                if self.message_hash_same_increase_temperature <= 0:
                    self.message_hash_same_increase_temperature = 0
        llm_config['temperature'] = self.message_hash_same_increase_temperature
        # # message_hash = hash_messages(messages)
        # if message_hash == self.message_hash:
        #     self.message_hash_same_increase_temperature += 0.4
        #     if self.message_hash_same_increase_temperature >= 1:
        #         self.message_hash_same_increase_temperature = 1
        #     llm_config['temperature'] = self.message_hash_same_increase_temperature
        # else:
        #     self.message_hash_same_increase_temperature = 0
        # self.message_hash = message_hash
        llm_config['functions'] = self.functions
        if messages[-1].get('function_call'):
            llm_config['function_call'] = messages[-1]['function_call']
            del(messages[-1]['function_call'])
        if self.replay_llm_responses_path:
            with open(self.replay_llm_responses_path, 'r') as f:
                responses = json.load(f)
            response = responses[self.replay_llm_responses_path_index]
            self.replay_llm_responses_path_index += 1
            if 'error' in response:
                raise InvalidRequestError(response['error'], '')
        else:
            try:
                # Check number of tokens
                num_tokens = num_tokens_consumed_by_chat_request(messages=messages, functions=self.functions)
                if num_tokens > self.max_tokens:
                    raise InvalidRequestError('InvalidRequestError', 'SelfGeneratedErrorOverTokenLimit')
                response = chat_completion_rl(**llm_config)
                self.responses.append(response)
                with open(f'{self.folder_path}{self.name}_llm_responses.json', 'w') as f:
                    json.dump(self.responses, f)
            except openai.error.InvalidRequestError as e:
                self.responses.append({'error': 'InvalidRequestError'})
                self.logger.error('Error: InvalidRequestError')
                self.logger.error(traceback.format_exc())
                self.logger.info("Error:", e.__dict__)  # or use a logging framework
                raise e
        if len(response['choices']) > 1 or n is not None:
            message_responses = []
            for choice in response['choices']:
                message_response = choice["message"]
                if not message_response.get('content'):
                    message_response['content'] = None
                message_responses.append(message_response)
            # self.print_dialog([message_response], response_msg=True)
            if print_:
                self.print_dialog(message_responses, response_msg=True)
            return message_responses
        else:
            message_response = response["choices"][0]["message"]
            if not message_response.get('content'):
                message_response['content'] = None
            if print_:
                self.print_dialog([message_response], response_msg=True)
            return message_response

    def get_function_names_as_str(self):
        fns = process_functions_into_function_names(self.functions)
        return ', '.join([f'`{fn}`'for fn in fns])

    def run(self, state=''):
        # try:
        return self._run(state)
        # except Exception as e:
        #     self.logger.error('Error in LLMatic.run()')
        #     self.logger.error(e)
        #     self.logger.error(traceback.format_exc())
        #     self.save_agent_state(self.sub_messages)
        #     write_files_from_dict(self.file_dict, base_dir=f'{self.folder_path}/{self.name}')
        #     raise e


    def get_llm_response_with_retries(self, messages, n=None, print_=True):
            has_returned_successfully = False
            while not has_returned_successfully:
                try:
                    response_message = self.get_llm_response(messages, n=n, print_=print_)
                    has_returned_successfully = True
                except InvalidRequestError as e:
                    # Calculate exactly where the token limit overflowed, and undo messages till just before it overflowed.
                    while (self.max_tokens - num_tokens_consumed_by_chat_request(messages=messages, functions=self.functions)) < 700:
                        messages.pop(3)
                    self.re_tries += 1
                    if self.re_tries > self.max_re_tries:
                        self.logger.warning(f'[WARNING] Maximum re-tries reached: {self.re_tries}/{self.max_re_tries}, exiting run!')
                        raise ValueError(f'[ERROR] Maximum re-tries reached: {self.re_tries}/{self.max_re_tries}, stopping run.')
            return response_message

    def generate_reflection_competition_for_generation_dict(self, generation_dict):
        if self.env.env_name == 'Cancer' or self.env.env_name == 'Cancer-ood' or self.env.env_name == 'Cancer-iid' or 'Cancer-random' in self.env.env_name:
            val_loss_per_dim_str = f"(Where the val loss per dimension is tumor_volume val loss: {generation_dict['code_dict']['loss_per_dim_dict']['tumor_volume_val_loss']:.3g}, chemotherapy_drug_concentration val loss: {generation_dict['code_dict']['loss_per_dim_dict']['chemo_drug_concentration_val_loss']:.3g})"
        elif self.env.env_name == 'Cancer-untreated':
            val_loss_per_dim_str = f"(Where the val loss per dimension is tumor_volume val loss: {generation_dict['code_dict']['loss_per_dim_dict']['tumor_volume_val_loss']:.3g})"
        elif self.env.env_name == 'Cancer-chemo':
            val_loss_per_dim_str = f"(Where the val loss per dimension is tumor_volume val loss: {generation_dict['code_dict']['loss_per_dim_dict']['tumor_volume_val_loss']:.3g}, chemotherapy_drug_concentration val loss: {generation_dict['code_dict']['loss_per_dim_dict']['chemo_drug_concentration_val_loss']:.3g})"
        elif self.env.env_name == 'Dataset-3DLV':
            val_loss_per_dim_str = f"(Where the val loss per dimension is prey_population val loss: {generation_dict['code_dict']['loss_per_dim_dict']['prey_population']:.3g}, intermediate_population val loss: {generation_dict['code_dict']['loss_per_dim_dict']['intermediate_population']:.3g}, top_predators_population val loss: {generation_dict['code_dict']['loss_per_dim_dict']['top_predators_population']:.3g})"
        elif self.env.env_name == 'Dataset-HL':
            val_loss_per_dim_str = f"(Where the val loss per dimension is hare_population val loss: {generation_dict['code_dict']['loss_per_dim_dict']['hare_population']:.3g}, lynx_population val loss: {generation_dict['code_dict']['loss_per_dim_dict']['lynx_population']:.3g}"
        elif self.env.env_name == 'COVID':
            val_loss_per_dim_str = f"(Where the val loss per dimension is susceptible val loss: {generation_dict['code_dict']['loss_per_dim_dict']['susceptible']:.3g}, exposed val loss: {generation_dict['code_dict']['loss_per_dim_dict']['exposed']:.3g}, infected val loss: {generation_dict['code_dict']['loss_per_dim_dict']['infected']:.3g}, recovered val loss: {generation_dict['code_dict']['loss_per_dim_dict']['recovered']:.3g})"
        else:
            raise ValueError(f'Unknown env name: {self.env.env_name}')
        completion = f"""
Val Loss: {generation_dict['code_dict']['val_loss']:.3g} {val_loss_per_dim_str} Iteration: {generation_dict['iteration']}
###
```
{generation_dict['code_dict']['StateDifferential_code']}
```
optimized_parameters = {generation_dict['code_dict']['optimized_parameters']}
###

"""
        return completion

    def generate_reflection_prompt_with_group(self, generation_dicts, history_generation, iteration, history_best_generation):
            generation_dicts = sorted(generation_dicts, key=lambda d: d['fitness'], reverse=True)
            # history_completions = []
            # for past_completion_idx in range(1,min(len(history_generation), self.config.run.reflection_history) + 1):
            #     past_completion = history_generation[-past_completion_idx]
            #     history_completions.append(self.generate_reflection_competition_for_generation_dict(past_completion))
            # history_completions = reversed(history_completions)
            # history_completions_str = '\n'.join(history_completions)
            history_best_completions = []

            for idx, past_completion in enumerate(history_best_generation):
                history_best_completions.append(f"Iteration {idx}. Best Val Loss: {past_completion['code_dict']['val_loss']}. Model description: {past_completion['code_dict']['model_description']}")
            history_best_completions_str = '\n'.join(history_best_completions)

            completions = []
            for generation_dict in generation_dicts:
                completions.append(self.generate_reflection_competition_for_generation_dict(generation_dict))
            completions = '\n'.join(completions)
            if self.name == 'NSDT' or self.name == 'NSDT-no-memory':
                return f"""
You generated the following code completions, which then had their parameters optimized to the training dataset. Please reflect on how you can improve the code to minimize the validation loss to 1e-6 or less. The code examples are delineated by ###.

Here are your previous iterations the best programs generated. Use it to see if you have exhausted white box models, i.e. when a white box model repeats with the same val loss and then only add black box models to the white box models:```
{history_best_completions_str}
```

Here are the top code completions so far that you have generated, sorted for the lowest validation loss last:```
{completions}
```

Please reflect on how you can improve the code to fit the dataset as accurately as possible, and be interpretable. Think step-by-step. Provide only actionable feedback, that has direct changes to the code. Do not write out the code, only describe how it can be improved. Where applicable use the values of the optimized parameters to reason how the code can be improved to fit the dataset as accurately as possible. This is for generating new code for the next iteration {iteration} out of {self.config.run.generations}.
"""
            elif self.name == 'NSDT-no-critic':
                return f"""
You generated the following code completions, which then had their parameters optimized to the training dataset. Improve the code to minimize the validation loss to 1e-6 or less. The code examples are delineated by ###.

Here are your previous iterations the best programs generated. Use it to see if you have exhausted white box models, i.e. when a white box model repeats with the same val loss and then only add black box models to the white box models:```
{history_best_completions_str}
```

Here are the top code completions so far that you have generated, sorted for the lowest validation loss last:```
{completions}
```

This is for generating new code for the next iteration {iteration} out of {self.config.run.generations}.
"""
            else:
                raise NotImplementedError

    def _run(self, state=''):
        self.messages = [self.system_message]
        initial_prompt = prompts.first_task_prompt(env_name=self.env.env_name, generations=self.config.run.generations)
        print(initial_prompt)
        print('')
        logs = []
        generation_id = 0
        n = 1
        self.logger.info(f'[Running generation 0] {self.name} | {self.env.env_name} | {self.env.seed} | Sampling n={n} keep_top_samples')

        self.messages.append({"role": "user", "content": initial_prompt, 'function_call': {"name": 'complete_StateDifferential_code'}})
        self.save_agent_state(self.messages)
        self.max_re_tries = 30
        self.re_tries = 0
        response_messages = self.get_llm_response_with_retries(self.messages, n=n) # Code message
        generation_dicts = []
        history_generation = []
        history_best_generation = []
        computed_funcs = set()
        for response_message in response_messages:
            code_json = json.dumps(response_message["function_call"])
            if code_json not in computed_funcs:
                # try:
                function_return_message, code_dict, has_success = process_function_call_and_return_message(response_message["function_call"], self.simulator_code_dict, env=self.env, functions=self.functions, config=self.config, logger=self.logger)
                generation_dict = {'function_return_message': deepcopy(function_return_message), 'iteration': generation_id, 'code_dict': deepcopy(code_dict), 'has_success': deepcopy(has_success), 'fitness': deepcopy(code_dict['val_loss']), 'code_string': code_json}
                generation_dicts.append(generation_dict)
                history_generation.append(generation_dict)
                computed_funcs.add(json.dumps(response_message["function_call"]))
                # except Exception as e:
                #     self.logger.info(f'[Invalid code, skipping evaluation of program] {e}')
            else:
                generation_dict = deepcopy([generation_dict for generation_dict in history_generation if generation_dict['code_string'] == code_json][0])
                generation_dict['iteration'] = generation_id
                history_generation.append(generation_dict)


        generation_dicts = sorted(generation_dicts, key=lambda d: d['fitness'])
        mean_fitness = np.mean([generation_dict['fitness'] for generation_dict in generation_dicts])
        fitnesses = [generation_dict['fitness'] for generation_dict in generation_dicts]
        num_programs = len(generation_dicts)
        top_fitness = generation_dicts[0]['fitness']
        history_best_generation.append(deepcopy(generation_dicts[0]))
        self.logger.info(f"[{self.name} | {self.env.env_name} | {self.env.seed}][Generation {generation_id}] | Top Fitness: {top_fitness} | Num Programs: {num_programs} | Mean Fitness: {mean_fitness} | Fitnesses: {fitnesses} | Current Gen Val Loss: {generation_dict['fitness']}")
        result = {'method': self.name, 'env_name': self.env.env_name, 'seed': self.env.seed, 'generation': generation_id, 'top_fitness': top_fitness, 'num_programs': num_programs, 'mean_fitness': mean_fitness, 'fitnesses': fitnesses, 'current_gen_val_loss': generation_dict['fitness']}
        self.logger.info(f"[{self.name} | {self.env.env_name} | {self.env.seed}][GEN RESULT] {result}")
            # self.messages.append(response_message)
        

        best_fitness = float('inf')  # Initialize with a very high value
        patience_counter = 0  # Counter for tracking patience

        for generation_id in range(1, self.config.run.generations):
            generation_dicts = sorted(generation_dicts, key=lambda d: d['fitness'])
            num_programs = len(generation_dicts)
            # sub_group_programs = num_programs // 4
            # if sub_group_programs <= 2:
            #     sub_group_programs = 2
            sub_group_programs = n
            # groups = sample_all_groups(generation_dicts, n=sub_group_programs)
            # for group in groups:
            reflection_prompt = self.generate_reflection_prompt_with_group(generation_dicts=generation_dicts, history_generation=history_generation, iteration=generation_id, history_best_generation=history_best_generation)
            messages_i = deepcopy(self.messages)
            if self.name == 'NSDT' or self.name == 'NSDT-no-memory':
                messages_i.append({"role": "user", "content": reflection_prompt, 'function_call': 'none'})
                response_message = self.get_llm_response_with_retries(messages_i, print_=False) # Reflection message
                messages_i.append(response_message)
                messages_i.append({"role": "user", "content": f"""
Please now regenerate the code function, with the aim to improve the code to achieve a lower validation error. Use the feedback where applicable. You are generating code for iteration {generation_id} out of {self.config.run.generations} total iterations. When generating code if you are unsure about something, take your best guess. You have to generate code, and cannot give an empty string answer.

Please always only fill in the following code skeleton:```
{prompts.get_skeleton_code(self.env.env_name)}
```
You cannot change the code skeleton, or input variables.
""", 'function_call': {"name": 'complete_StateDifferential_code'}})
            elif self.name == 'NSDT-no-critic':
                messages_i = [self.system_message, {"role": "user", "content": prompts.first_task_prompt(env_name=self.env.env_name, generations=self.config.run.generations, current_iteration=generation_id), 'function_call': {"name": 'complete_StateDifferential_code'}}]
            response_messages = self.get_llm_response_with_retries(messages_i, n=sub_group_programs) # Code message
            for response_message in response_messages:
                code_json = json.dumps(response_message["function_call"])
                if code_json not in computed_funcs:
                    try:
                        function_return_message, code_dict, has_success = process_function_call_and_return_message(response_message["function_call"], self.simulator_code_dict, env=self.env, functions=self.functions, config=self.config, logger=self.logger)
                        generation_dict = {'function_return_message': deepcopy(function_return_message), 'iteration': generation_id, 'code_dict': deepcopy(code_dict), 'has_success': deepcopy(has_success), 'fitness': deepcopy(code_dict['val_loss']), 'code_string': code_json}
                        if self.name == 'NSDT-no-memory':
                            generation_dicts = []
                            history_generation = []
                            history_best_generation = []
                            computed_funcs = set()

                        generation_dicts.append(generation_dict)
                        history_generation.append(generation_dict)
                        computed_funcs.add(json.dumps(response_message["function_call"]))
                        # if has_success:
                        #         break
                    except Exception as e:
                        # self.logger.info(f'[Invalid code, skipping evaluation of program] {e}')
                        self.logger.info(f'[Invalid code, attempting to fix the code] {e}')
                        # Try to fix the code -- Give the explicit code skeleton!. Can even do this above perhaps to avoid this from happening?!
                else:
                    generation_dict = deepcopy([generation_dict for generation_dict in history_generation if generation_dict['code_string'] == code_json][0])
                    generation_dict['iteration'] = generation_id
                    history_generation.append(generation_dict)

            generation_dicts = sorted(generation_dicts, key=lambda d: d['fitness'])
            # Remove nan's
            generation_dicts = [generation_dict for generation_dict in generation_dicts if not np.isnan(generation_dict['fitness'])]
            # Cull poor fitnesses
            generation_dicts = generation_dicts[:self.config.run.keep_top_samples]
            mean_fitness = np.mean([generation_dict['fitness'] for generation_dict in generation_dicts])
            fitnesses = [generation_dict['fitness'] for generation_dict in generation_dicts]
            num_programs = len(generation_dicts)
            top_fitness = generation_dicts[0]['fitness']
            history_best_generation.append(deepcopy(generation_dicts[0]))
            # self.logger.info(f'[{self.name} | {self.env.env_name} | {self.env.seed}][Generation {generation_id}] | Top Fitness: {top_fitness:.3g} | Num Programs: {num_programs} | Mean Fitness: {mean_fitness} | Fitnesses: {fitnesses}')
            self.logger.info(f"[{self.name} | {self.env.env_name} | {self.env.seed}][Generation {generation_id}] | Top Fitness: {top_fitness} | Num Programs: {num_programs} | Mean Fitness: {mean_fitness} | Fitnesses: {fitnesses} | Current Gen Val Loss: {generation_dict['fitness']}")
            result = {'method': self.name, 'env_name': self.env.env_name, 'seed': self.env.seed, 'generation': generation_id, 'top_fitness': top_fitness, 'num_programs': num_programs, 'mean_fitness': mean_fitness, 'fitnesses': fitnesses, 'current_gen_val_loss': generation_dict['fitness']}
            self.logger.info(f"[{self.name} | {self.env.env_name} | {self.env.seed}][GEN RESULT] {result}")

            # Early stopping check
            if top_fitness < best_fitness:
                best_fitness = top_fitness
                patience_counter = 0  # Reset counter on improvement
            else:
                patience_counter += 1  # Increment counter if no improvement
            if patience_counter >= self.config.run.nsdt_patience:
                self.logger.info(f"Early stopping triggered at generation {generation_id}")
                break  # Exit the loop if no improvement for 'patience' generations
        
        # Evaluate best code now
        best_program = generation_dicts[0]['code_dict']
        test_loss = best_program['test_loss']
        # test_loss = self.env.evaluate_simulator_code_on_test_dataset(best_program['StateDifferential_code'], config=self.config, logger=self.logger)
        self.logger.info(f'[{self.env.env_name} {self.name} {self.env.seed}][Test Run completed successfully] MSE TEST LOSS {test_loss:.4f}')
        return test_loss
        # self.logger.info(f'[Run completed successfully] MSE VAL LOSS {self.simulator_code_dict["val_loss"]:.4f} | Iteration {iteration}/{self.max_iterations}')
        # self.logger.info('')
        # write_files_from_dict(self.simulator_code_dict, base_dir=f'{self.folder_path}/{self.name}')
        # self.save_agent_state(self.sub_messages)
        # return f'{self.folder_path}/{self.name}'

class ZeroShot(NSDT):
    def __init__(self, env, config, logger, rate_limiter, optimize_params=False, name=''):
        super().__init__(env, config, logger, rate_limiter)
        self.name = 'ZeroZhot'
        self.folder_path = f"{self.config.run.log_path.split('.txt')[0]}/{self.env.env_name}/{self.env.seed}/"
        Path(self.folder_path).mkdir(parents=True, exist_ok=True)
        self.max_tokens = get_model_max_tokens(config)
        self.functions = function_definition_list_factory()
        self.config.run.optimize_params = optimize_params
        print('Optimize params:', self.config.run.optimize_params)
        print('')


    def _run(self, state=''):
        self.messages = [self.system_message]
        initial_prompt = prompts.first_task_prompt(env_name=self.env.env_name, generations=self.config.run.generations)
        print(initial_prompt)
        print('')
        logs = []
        generation_id = 0
        # n = self.config.run.keep_top_samples // 2
        n = 1
        self.logger.info(f'[Running generation 0] {self.name} | {self.env.env_name} | {self.env.seed} | Sampling n={n} keep_top_samples')

        self.messages.append({"role": "user", "content": initial_prompt, 'function_call': {"name": 'complete_StateDifferential_code'}})
        self.save_agent_state(self.messages)
        self.max_re_tries = 30
        self.re_tries = 0
        response_messages = self.get_llm_response_with_retries(self.messages, n=n) # Code message
        generation_dicts = []
        computed_funcs = set()
        for response_message in response_messages:
            if json.dumps(response_message["function_call"]) not in computed_funcs:
                # try:
                function_return_message, code_dict, has_success = process_function_call_and_return_message(response_message["function_call"], self.simulator_code_dict, env=self.env, functions=self.functions, config=self.config, logger=self.logger)
                generation_dicts.append({'function_return_message': deepcopy(function_return_message), 'code_dict': deepcopy(code_dict), 'has_success': deepcopy(has_success), 'fitness': deepcopy(code_dict['val_loss'])})
                computed_funcs.add(json.dumps(response_message["function_call"]))
                # except Exception as e:
                #     self.logger.info(f'[Invalid code, skipping evaluation of program] {e}')


        generation_dicts = sorted(generation_dicts, key=lambda d: d['fitness'])
        best_program = generation_dicts[0]['code_dict']
        test_loss = best_program['test_loss']
        # test_loss = self.env.evaluate_simulator_code_on_test_dataset(best_program['StateDifferential_code'], config=self.config, logger=self.logger)
        self.logger.info(f'[{self.env.env_name} {self.name} {self.env.seed}][Test Run completed successfully] MSE TEST LOSS {test_loss:.4f}')
        return test_loss
                                   

class DyNODEAgent(Agent):
    def __init__(self, env, config, logger, rate_limiter):
        super().__init__(env, config, logger, rate_limiter)
        self.name = 'DyNODE'

        self.folder_path = f"{self.config.run.log_path.split('.txt')[0]}/{self.env.env_name}/{self.env.seed}/"
        Path(self.folder_path).mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if config.setup.cuda else "cpu"
        if self.env.train_data[1] is None:
            self.action_dim = None
        else:
            self.action_dim = self.env.train_data[1].shape[-1]
        self.states_dim = self.env.train_data[0].shape[-1]

        states_train, actions_train = self.env.train_data
        states_train = torch.tensor(states_train, dtype=torch.float32, device=self.device)
        if actions_train is not None:
            actions_train = torch.tensor(actions_train, dtype=torch.float32, device=self.device)
            states_actions_train = torch.cat([states_train, actions_train], dim=-1)
        else:
            states_actions_train = states_train
        
        self.f_model = DyNODEModel(self.states_dim, self.action_dim, states_actions_train)
        self.f_model.to(self.device)
        self.model_path = f'{self.config.run.dynode_saved_models_folder}/dynode_model_{self.env.env_name}_0_trajs_{self.config.run.trajectories}.pt'

    def run(self, state=''):
        # try:
        return self._run(state)
        # except Exception as e:
        #     self.logger.error('Error in LLMatic.run()')
        #     self.logger.error(e)
        #     self.logger.error(traceback.format_exc())
        #     self.save_agent_state(self.sub_messages)
        #     write_files_from_dict(self.file_dict, base_dir=f'{self.folder_path}/{self.name}')
        #     raise e
    
    def _run(self, state=''):
        if self.config.run.dynode_retrain_model:
            self.train_model()
        else:
            # Attempt to load model if exists, if not exists then train model
            if Path(self.model_path).exists():
                self.f_model.load_state_dict(torch.load(self.model_path))
                self.f_model.to(self.device)
                self.f_model.eval()
            else:
                self.logger.info(f'[WARNING] No saved model found at {self.model_path}, training from scratch...')
                self.train_model()
        
        states_test, actions_test = self.env.test_data
        states_test = torch.tensor(states_test, dtype=torch.float32, device=self.device)
        if actions_test is not None:
            actions_test = torch.tensor(actions_test, dtype=torch.float32, device=self.device)
            states_actions_test = torch.cat([states_test, actions_test], dim=-1)
        else:
            states_actions_test = states_test


        MSE = torch.nn.MSELoss()
        # Collect test loss
        self.f_model.eval()
        with torch.no_grad():
            pred_states = []
            pred_state = states_actions_test[:,0,:self.states_dim]
            for t in range(states_actions_test.shape[1]):
                pred_states.append(pred_state)
                dx_dt = self.f_model(states_actions_test[:,t])
                pred_state = states_actions_test[:,t,:self.states_dim] + dx_dt
            pred_states = torch.stack(pred_states, dim=1)
            test_loss = MSE(pred_states, states_actions_test[:,:,:self.states_dim]).item()
        self.logger.info(f'[{self.env.env_name} {self.name} {self.env.seed}][Test Run completed successfully] MSE TEST LOSS {test_loss:.4f}')

        if self.env.env_name == 'COVID-insight':
            print('')

            with torch.no_grad():

                (states_val, actions_val) = self.env.val_data
                (states_test, actions_test) = self.env.test_data

                noint_states, noint_actions = states_val, actions_val
                int_states, int_actions = states_test, actions_test

                pred_states = []
                pred_state = states_actions_test[:,19,:self.states_dim]
                for t in range(19, states_actions_test.shape[1]):
                    pred_states.append(pred_state)
                    dx_dt = self.f_model(states_actions_test[:,t])
                    pred_state = pred_state + dx_dt
                pred_states = torch.stack(pred_states, dim=1)

                predicted_dynode_states = torch.concat((states_actions_test[:,:19,:self.states_dim],pred_states),dim=1).detach()

                predicted_dynode_states = predicted_dynode_states.cpu().numpy()

                np.savez('COVID-intervention-dynode.npz',
                    predicted_sindy_states=predicted_dynode_states,
                )

                import matplotlib.pyplot as plt
                # 6, 7?, 9!; 20# with rejig
                # traj_idx = 20
                traj_idx = 9
                # Create a figure and a set of subplots
                fig, axs = plt.subplots(4, 1, sharex=True)
                # Set the overall title for all subplots
                fig.suptitle(f"SEIRD for traj {traj_idx}")
                # Plotting each data in a separate subplot
                axs[0].plot(noint_states[traj_idx,:,0], label="Not Int")
                axs[0].plot(int_states[traj_idx,:,0], label="Int")
                axs[0].plot(predicted_dynode_states[traj_idx,:,0], label="DYNODE")
                # axs[0].set_ylim((0.0, 1.0))
                # axs[0].plot(predicted_full_states[traj_idx,:,0], label="NSDT")
                axs[0].legend()
                axs[1].plot(noint_states[traj_idx,:,1], label="Not Int")
                axs[1].plot(int_states[traj_idx,:,1], label="Int")
                axs[1].plot(predicted_dynode_states[traj_idx,:,1], label="DYNODE")
                # axs[1].plot(predicted_full_states[traj_idx,:,1], label="NSDT")
                axs[1].legend()
                axs[2].plot(noint_states[traj_idx,:,2], label="Not Int")
                axs[2].plot(int_states[traj_idx,:,2], label="Int")
                axs[2].plot(predicted_dynode_states[traj_idx,:,2], label="DYNODE")
                # axs[2].plot(predicted_full_states[traj_idx,:,2], label="NSDT")
                axs[2].legend()
                axs[3].plot(noint_states[traj_idx,:,3], label="Not Int")
                axs[3].plot(int_states[traj_idx,:,3], label="Int")
                axs[3].plot(predicted_dynode_states[traj_idx,:,3], label="DYNODE")
                # axs[3].plot(predicted_full_states[traj_idx,:,3].cpu(), label="NSDT")
                axs[3].legend()
                # axs[4].plot(states[traj_idx,:,4], label="Dead")
                # axs[4].legend()
                # axs[4].plot(actions[traj_idx,:,0], label="Intervention")
                # axs[4].legend()
                # Adjust the layout
                plt.tight_layout()
                # Save the figure
                plt.savefig("test.png")
                # plt.clf()
                print('')


        return test_loss
    
    def train_model(self):
        # Train model
        self.f_model.train()
        states_train, actions_train = self.env.train_data
        states_train = torch.tensor(states_train, dtype=torch.float32, device=self.device)
        if actions_train is not None:
            actions_train = torch.tensor(actions_train, dtype=torch.float32, device=self.device)
            states_actions_train = torch.cat([states_train, actions_train], dim=-1)
        else:
            states_actions_train = states_train

        states_val, actions_val = self.env.val_data
        states_val = torch.tensor(states_val, dtype=torch.float32, device=self.device)
        if actions_val is not None:
            actions_val = torch.tensor(actions_val, dtype=torch.float32, device=self.device)
            states_actions_val = torch.cat([states_val, actions_val], dim=-1)
        else:
            states_actions_val = states_val

        batch_size = min(self.config.run.pytorch_as_optimizer.batch_size, states_actions_train.shape[0])
        MSE = torch.nn.MSELoss()
        optimizer = optim.Adam(self.f_model.parameters(), lr=self.config.run.dynode_learning_rate, weight_decay=self.config.run.pytorch_as_optimizer.weight_decay)
        # clip_grad_norm = self.config.run.pytorch_as_optimizer.clip_grad_norm if self.config.run.pytorch_as_optimizer.clip_grad_norm > 0 else None

        def train(model, states_actions_train_batch_i):
            optimizer.zero_grad(True)
            pred_states = []
            pred_state = states_actions_train_batch_i[:,0,:self.states_dim]
            for t in range(states_actions_train_batch_i.shape[1]):
                pred_states.append(pred_state)
                dx_dt = model(states_actions_train_batch_i[:,t])
                pred_state = states_actions_train_batch_i[:,t,:self.states_dim] + dx_dt
            pred_states = torch.stack(pred_states, dim=1)
            loss = MSE(pred_states, states_actions_train_batch_i[:,:,:self.states_dim])
            loss.backward()
            # if clip_grad_norm:
            #     torch.nn.utils.clip_grad_norm_(self.f_model.parameters(), clip_grad_norm)
            optimizer.step()
            return loss.item()
        
        # train_opt = torch.compile(train)
        train_opt = train

        best_val_loss = float('inf')  # Initialize with a very high value
        patience_counter = 0  # Counter for tracking patience

                
        for epoch in range(self.config.run.pytorch_as_optimizer.epochs):
            iters = 0 
            cum_loss = 0
            t0 = time.perf_counter()
            permutation = torch.randperm(states_actions_train.shape[0]) # Need to change to arange for causal for camera ready
            for iter_i in range(int(permutation.shape[0]/batch_size)):
                indices = permutation[iter_i*batch_size:iter_i*batch_size+batch_size]
                states_actions_train_batch = states_actions_train[indices]
                cum_loss += train_opt(self.f_model, states_actions_train_batch)
                iters += 1
            time_taken = time.perf_counter() - t0
            if epoch % self.config.run.pytorch_as_optimizer.log_interval == 0:
                # Collect validation loss
                self.f_model.eval()
                with torch.no_grad():
                    pred_states = []
                    pred_state = states_actions_val[:,0,:self.states_dim]
                    for t in range(states_actions_val.shape[1]):
                        pred_states.append(pred_state)
                        dx_dt = self.f_model(states_actions_val[:,t])
                        pred_state = states_actions_val[:,t,:self.states_dim] + dx_dt
                    pred_states = torch.stack(pred_states, dim=1)
                    val_loss = MSE(pred_states, states_actions_val[:,:,:self.states_dim]).item()
                self.f_model.train()
                self.logger.info(f'[{self.name}][EPOCH {epoch} COMPLETE] MSE TRAIN LOSS {cum_loss/iters:.4f} | MSE VAL LOSS {val_loss:.4f} | s/epoch: {time_taken:.2f}s')
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0  # Reset counter on improvement
                else:
                    patience_counter += 1  # Increment counter if no improvement
                if patience_counter >= self.config.run.optimization.patience:
                    self.logger.info(f"Early stopping triggered at epoch {epoch}")
                    break  # Exit the loop if no improvement for 'patience' generations

        # Save model after training
        self.f_model.eval()
        # torch.save(self.f_model.state_dict(), f'{self.folder_path}dynode_model_{self.env.env_name}_0_trajs_{self.config.run.pytorch_as_optimizer.trajectories}.pt')
        torch.save(self.f_model.state_dict(), self.model_path)
        self.logger.info(f'[Train Run completed successfully] MSE VAL LOSS {val_loss:.4f}')
        self.logger.info('')

class RNNAgent(Agent):
    def __init__(self, env, config, logger, rate_limiter):
        super().__init__(env, config, logger, rate_limiter)
        self.name = 'RNN'

        self.folder_path = f"{self.config.run.log_path.split('.txt')[0]}/{self.env.env_name}/{self.env.seed}/"
        Path(self.folder_path).mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if config.setup.cuda else "cpu"
        if self.env.train_data[1] is None:
            self.action_dim = None
        else:
            self.action_dim = self.env.train_data[1].shape[-1]
        self.states_dim = self.env.train_data[0].shape[-1]

        states_train, actions_train = self.env.train_data
        states_train = torch.tensor(states_train, dtype=torch.float32, device=self.device)
        if actions_train is not None:
            actions_train = torch.tensor(actions_train, dtype=torch.float32, device=self.device)
            states_actions_train = torch.cat([states_train, actions_train], dim=-1)
        else:
            states_actions_train = states_train
        
        self.f_model = RNNModel(self.states_dim, self.action_dim, states_actions_train)
        self.f_model.to(self.device)
        self.model_path = f'{self.config.run.dynode_saved_models_folder}/rnn_model_{self.env.env_name}_0_trajs_{self.config.run.trajectories}.pt'

    def run(self, state=''):
        # try:
        return self._run(state)
        # except Exception as e:
        #     self.logger.error('Error in LLMatic.run()')
        #     self.logger.error(e)
        #     self.logger.error(traceback.format_exc())
        #     self.save_agent_state(self.sub_messages)
        #     write_files_from_dict(self.file_dict, base_dir=f'{self.folder_path}/{self.name}')
        #     raise e
    
    def _run(self, state=''):
        if self.config.run.dynode_retrain_model:
            self.train_model()
        else:
            # Attempt to load model if exists, if not exists then train model
            if Path(self.model_path).exists():
                self.f_model.load_state_dict(torch.load(self.model_path))
                self.f_model.to(self.device)
                self.f_model.eval()
            else:
                self.logger.info(f'[WARNING] No saved model found at {self.model_path}, training from scratch...')
                self.train_model()
        
        states_test, actions_test = self.env.test_data
        states_test = torch.tensor(states_test, dtype=torch.float32, device=self.device)
        if actions_test is not None:
            actions_test = torch.tensor(actions_test, dtype=torch.float32, device=self.device)
            states_actions_test = torch.cat([states_test, actions_test], dim=-1)
        else:
            states_actions_test = states_test

        windows_test, labels_test = self.process_dataset_to_rnn_format(states_actions_test)

        MSE = torch.nn.MSELoss()
        self.f_model.eval()
        with torch.no_grad():
            test_loss = MSE(self.f_model(windows_test), labels_test).item()
        self.logger.info(f'[{self.env.env_name} {self.name} {self.env.seed}][Test Run completed successfully] MSE TEST LOSS {test_loss:.4f}')
        return test_loss

        MSE = torch.nn.MSELoss()
        # Collect test loss
        self.f_model.eval()
        with torch.no_grad():
            pred_states = []
            pred_state = states_actions_test[:,0,:self.states_dim]
            for t in range(states_actions_test.shape[1]):
                pred_states.append(pred_state)
                dx_dt = self.f_model(states_actions_test[:,t])
                pred_state = states_actions_test[:,t,:self.states_dim] + dx_dt
            pred_states = torch.stack(pred_states, dim=1)
            test_loss = MSE(pred_states, states_actions_test[:,:,:self.states_dim]).item()
        self.logger.info(f'[{self.env.env_name} {self.name} {self.env.seed}][Test Run completed successfully] MSE TEST LOSS {test_loss:.4f}')
        return test_loss
    
    def process_dataset_to_rnn_format(self, states_actions, window_length=20):
        # Prepare the windows and labels

        if states_actions.shape[1] <= window_length:
            return states_actions, states_actions[:, -1, :self.states_dim]

        windows = []
        labels = []
        for i in range(states_actions.shape[1] - window_length):
            windows.append(states_actions[:, i:i+window_length, :])
            labels.append(states_actions[:, i+window_length, :self.states_dim])

        # Convert lists to tensors
        windows = torch.stack(windows, dim=1).view(-1, window_length, states_actions.shape[-1])
        labels = torch.stack(labels, dim=0).view(-1, self.states_dim)
        return windows, labels
    
    def train_model(self):
        # Train model
        self.f_model.train()
        states_train, actions_train = self.env.train_data
        states_train = torch.tensor(states_train, dtype=torch.float32, device=self.device)
        if actions_train is not None:
            actions_train = torch.tensor(actions_train, dtype=torch.float32, device=self.device)
            states_actions_train = torch.cat([states_train, actions_train], dim=-1)
        else:
            states_actions_train = states_train

        states_val, actions_val = self.env.val_data
        states_val = torch.tensor(states_val, dtype=torch.float32, device=self.device)
        if actions_val is not None:
            actions_val = torch.tensor(actions_val, dtype=torch.float32, device=self.device)
            states_actions_val = torch.cat([states_val, actions_val], dim=-1)
        else:
            states_actions_val = states_val

        windows_train, labels_train = self.process_dataset_to_rnn_format(states_actions_train)
        windows_val, labels_val = self.process_dataset_to_rnn_format(states_actions_val)


        batch_size = min(self.config.run.pytorch_as_optimizer.batch_size, states_actions_train.shape[0])
        # batch_size = 10000
        # batch_size = 32
        # batch_size = 1
        # batch_size = 32
        MSE = torch.nn.MSELoss()
        # optimizer = optim.Adam(self.f_model.parameters(), lr=self.config.run.rnn_learning_rate, weight_decay=self.config.run.pytorch_as_optimizer.weight_decay)
        lr = 0.00005
        optimizer = optim.Adam(self.f_model.parameters(), lr=self.config.run.rnn_learning_rate)
        # optimizer = optim.AdamW(self.f_model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)


        # batch_size = min(self.config.run.pytorch_as_optimizer.batch_size, states_actions_train.shape[0])
        # # batch_size = 1
        # # batch_size = 32
        # MSE = torch.nn.MSELoss()
        # # optimizer = optim.Adam(self.f_model.parameters(), lr=self.config.run.rnn_learning_rate, weight_decay=self.config.run.pytorch_as_optimizer.weight_decay)
        # optimizer = optim.Adam(self.f_model.parameters(), lr=self.config.run.rnn_learning_rate)
        # # clip_grad_norm = self.config.run.pytorch_as_optimizer.clip_grad_norm if self.config.run.pytorch_as_optimizer.clip_grad_norm > 0 else None

        def train(model, windows_batch, labels_batch):
            optimizer.zero_grad(True)
            pred = model(windows_batch)
            loss = MSE(pred, labels_batch)
            loss.backward()
            # if clip_grad_norm:
            #     torch.nn.utils.clip_grad_norm_(self.f_model.parameters(), clip_grad_norm)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
            optimizer.step()
            return loss.item()
        
        # train_opt = torch.compile(train)
        train_opt = train

        best_val_loss = float('inf')  # Initialize with a very high value
        patience_counter = 0  # Counter for tracking patience

                
        for epoch in range(self.config.run.pytorch_as_optimizer.epochs):
            iters = 0 
            cum_loss = 0
            t0 = time.perf_counter()
            permutation = torch.randperm(windows_train.shape[0])
            # permutation = torch.arange(states_actions_train.shape[0])
            for iter_i in range(int(permutation.shape[0]/batch_size)):
                indices = permutation[iter_i*batch_size:iter_i*batch_size+batch_size]
                windows_train_batch, labels_train_batch = windows_train[indices], labels_train[indices] 
                cum_loss += train_opt(self.f_model, windows_train_batch, labels_train_batch)
                iters += 1
            time_taken = time.perf_counter() - t0
            if epoch % self.config.run.pytorch_as_optimizer.log_interval == 0:
                # Collect validation loss
                self.f_model.eval()
                with torch.no_grad():
                    windows_val, labels_val
                    val_loss = MSE(self.f_model(windows_val), labels_val).item()
                self.f_model.train()
                self.logger.info(f'[{self.name}][EPOCH {epoch} COMPLETE] MSE TRAIN LOSS {cum_loss/iters:.4f} | MSE VAL LOSS {val_loss:.4f} | s/epoch: {time_taken:.2f}s')
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0  # Reset counter on improvement
                else:
                    patience_counter += 1  # Increment counter if no improvement
                if patience_counter >= self.config.run.optimization.patience:
                    self.logger.info(f"Early stopping triggered at epoch {epoch}")
                    break  # Exit the loop if no improvement for 'patience' generations

        # Save model after training
        self.f_model.eval()
        # torch.save(self.f_model.state_dict(), f'{self.folder_path}dynode_model_{self.env.env_name}_0_trajs_{self.config.run.pytorch_as_optimizer.trajectories}.pt')
        torch.save(self.f_model.state_dict(), self.model_path)
        self.logger.info(f'[Train Run completed successfully] MSE VAL LOSS {val_loss:.4f}')
        self.logger.info('')



class TransformerAgent(Agent):
    def __init__(self, env, config, logger, rate_limiter):
        super().__init__(env, config, logger, rate_limiter)
        self.name = 'Transformer'

        self.folder_path = f"{self.config.run.log_path.split('.txt')[0]}/{self.env.env_name}/{self.env.seed}/"
        Path(self.folder_path).mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if config.setup.cuda else "cpu"
        if self.env.train_data[1] is None:
            self.action_dim = None
        else:
            self.action_dim = self.env.train_data[1].shape[-1]
        self.states_dim = self.env.train_data[0].shape[-1]

        states_train, actions_train = self.env.train_data
        states_train = torch.tensor(states_train, dtype=torch.float32, device=self.device)
        if actions_train is not None:
            actions_train = torch.tensor(actions_train, dtype=torch.float32, device=self.device)
            states_actions_train = torch.cat([states_train, actions_train], dim=-1)
        else:
            states_actions_train = states_train
        
        self.f_model = TransformerModel(self.states_dim, self.action_dim, states_actions_train)
        self.f_model.to(self.device)
        self.model_path = f'{self.config.run.dynode_saved_models_folder}/rnn_model_{self.env.env_name}_0_trajs_{self.config.run.trajectories}.pt'

    def run(self, state=''):
        # try:
        return self._run(state)
        # except Exception as e:
        #     self.logger.error('Error in LLMatic.run()')
        #     self.logger.error(e)
        #     self.logger.error(traceback.format_exc())
        #     self.save_agent_state(self.sub_messages)
        #     write_files_from_dict(self.file_dict, base_dir=f'{self.folder_path}/{self.name}')
        #     raise e
    
    def _run(self, state=''):
        if self.config.run.dynode_retrain_model:
            self.train_model()
        else:
            # Attempt to load model if exists, if not exists then train model
            if Path(self.model_path).exists():
                self.f_model.load_state_dict(torch.load(self.model_path))
                self.f_model.to(self.device)
                self.f_model.eval()
            else:
                self.logger.info(f'[WARNING] No saved model found at {self.model_path}, training from scratch...')
                self.train_model()
        
        states_test, actions_test = self.env.test_data
        states_test = torch.tensor(states_test, dtype=torch.float32, device=self.device)
        if actions_test is not None:
            actions_test = torch.tensor(actions_test, dtype=torch.float32, device=self.device)
            states_actions_test = torch.cat([states_test, actions_test], dim=-1)
        else:
            states_actions_test = states_test

        windows_test, labels_test = self.process_dataset_to_transformer_format(states_actions_test)
        windows_test, labels_test = windows_test, labels_test

        MSE = torch.nn.MSELoss()
        self.f_model.eval()
        with torch.no_grad():
            test_loss = MSE(self.f_model(windows_test), labels_test).item()
        self.logger.info(f'[{self.env.env_name} {self.name} {self.env.seed}][Test Run completed successfully] MSE TEST LOSS {test_loss:.4f}')
        return test_loss


        MSE = torch.nn.MSELoss()
        # Collect test loss
        self.f_model.eval()
        with torch.no_grad():
            pred_states = []
            pred_state = states_actions_test[:,0,:self.states_dim]
            for t in range(states_actions_test.shape[1]):
                pred_states.append(pred_state)
                dx_dt = self.f_model(states_actions_test[:,t])
                pred_state = states_actions_test[:,t,:self.states_dim] + dx_dt
            pred_states = torch.stack(pred_states, dim=1)
            test_loss = MSE(pred_states, states_actions_test[:,:,:self.states_dim]).item()
        self.logger.info(f'[{self.env.env_name} {self.name} {self.env.seed}][Test Run completed successfully] MSE TEST LOSS {test_loss:.4f}')
        return test_loss
    
    def process_dataset_to_transformer_format(self, states_actions, window_length=10, output_window=1):
        # Prepare the windows and labels

        windows = []
        labels = []
        for i in range(states_actions.shape[1] - window_length):
            windows.append(states_actions[:, i:i+window_length, :])
            labels.append(states_actions[:, i+output_window:i+window_length+output_window, :self.states_dim])

        # Convert lists to tensors
        windows = torch.cat(windows, dim=0)
        labels = torch.cat(labels, dim=0)
        return windows, labels
    
    def train_model(self):
        # Train model
        self.f_model.train()
        states_train, actions_train = self.env.train_data
        states_train = torch.tensor(states_train, dtype=torch.float32, device=self.device)
        if actions_train is not None:
            actions_train = torch.tensor(actions_train, dtype=torch.float32, device=self.device)
            states_actions_train = torch.cat([states_train, actions_train], dim=-1)
        else:
            states_actions_train = states_train

        states_val, actions_val = self.env.val_data
        states_val = torch.tensor(states_val, dtype=torch.float32, device=self.device)
        if actions_val is not None:
            actions_val = torch.tensor(actions_val, dtype=torch.float32, device=self.device)
            states_actions_val = torch.cat([states_val, actions_val], dim=-1)
        else:
            states_actions_val = states_val

        windows_train, labels_train = self.process_dataset_to_transformer_format(states_actions_train)
        windows_val, labels_val = self.process_dataset_to_transformer_format(states_actions_val)
        # windows_val, labels_val = windows_val[:100], labels_val[:100]


        batch_size = min(self.config.run.pytorch_as_optimizer.batch_size, states_actions_train.shape[0])
        # batch_size = 10000
        # batch_size = 32
        # batch_size = 1
        # batch_size = 32
        MSE = torch.nn.MSELoss()
        # optimizer = optim.Adam(self.f_model.parameters(), lr=self.config.run.rnn_learning_rate, weight_decay=self.config.run.pytorch_as_optimizer.weight_decay)
        lr = 0.00005
        # optimizer = optim.Adam(self.f_model.parameters(), lr=self.config.run.rnn_learning_rate)
        optimizer = optim.AdamW(self.f_model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
        # clip_grad_norm = self.config.run.pytorch_as_optimizer.clip_grad_norm if self.config.run.pytorch_as_optimizer.clip_grad_norm > 0 else None

        def train(model, windows_batch, labels_batch):
            optimizer.zero_grad(True)
            pred = model(windows_batch)
            loss = MSE(pred, labels_batch)
            loss.backward()
            # if clip_grad_norm:
            #     torch.nn.utils.clip_grad_norm_(self.f_model.parameters(), clip_grad_norm)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
            optimizer.step()
            return loss.item()
        
        # train_opt = torch.compile(train)
        train_opt = train

        best_val_loss = float('inf')  # Initialize with a very high value
        patience_counter = 0  # Counter for tracking patience

                
        for epoch in range(self.config.run.pytorch_as_optimizer.epochs):
            iters = 0 
            cum_loss = 0
            t0 = time.perf_counter()
            permutation = torch.randperm(windows_train.shape[0])
            # permutation = torch.arange(states_actions_train.shape[0])
            for iter_i in range(int(permutation.shape[0]/batch_size)):
                indices = permutation[iter_i*batch_size:iter_i*batch_size+batch_size]
                windows_train_batch, labels_train_batch = windows_train[indices], labels_train[indices] 
                cum_loss += train_opt(self.f_model, windows_train_batch, labels_train_batch)
                iters += 1
            time_taken = time.perf_counter() - t0
            if epoch % self.config.run.pytorch_as_optimizer.log_interval == 0:
                # Collect validation loss
                self.f_model.eval()
                with torch.no_grad():
                    val_loss = MSE(self.f_model(windows_val), labels_val).item()
                self.f_model.train()
                self.logger.info(f'[{self.name}][EPOCH {epoch} COMPLETE] MSE TRAIN LOSS {cum_loss/iters:.4f} | MSE VAL LOSS {val_loss:.4f} | s/epoch: {time_taken:.2f}s')
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0  # Reset counter on improvement
                else:
                    patience_counter += 1  # Increment counter if no improvement
                if patience_counter >= self.config.run.optimization.patience:
                    self.logger.info(f"Early stopping triggered at epoch {epoch}")
                    break  # Exit the loop if no improvement for 'patience' generations

        # Save model after training
        self.f_model.eval()
        # torch.save(self.f_model.state_dict(), f'{self.folder_path}dynode_model_{self.env.env_name}_0_trajs_{self.config.run.pytorch_as_optimizer.trajectories}.pt')
        torch.save(self.f_model.state_dict(), self.model_path)
        self.logger.info(f'[Train Run completed successfully] MSE VAL LOSS {val_loss:.4f}')
        self.logger.info('')


class GPAgent(Agent):
    def __init__(self, env, config, logger, rate_limiter):
        super().__init__(env, config, logger, rate_limiter)
        self.name = 'GP'

        self.folder_path = f"{self.config.run.log_path.split('.txt')[0]}/{self.env.env_name}/{self.env.seed}/"
        Path(self.folder_path).mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if config.setup.cuda else "cpu"
        if self.env.train_data[1] is None:
            self.action_dim = None
        else:
            self.action_dim = self.env.train_data[1].shape[-1]
        self.states_dim = self.env.train_data[0].shape[-1]

        states_train, actions_train = self.env.train_data
        states_train = torch.tensor(states_train, dtype=torch.float32, device=self.device)
        self.state_dim = states_train.shape[-1]
        if actions_train is not None:
            actions_train = torch.tensor(actions_train, dtype=torch.float32, device=self.device)
            states_actions_train = torch.cat([states_train, actions_train], dim=-1)
        else:
            states_actions_train = states_train
        
        self.model_path = f'{self.config.run.dynode_saved_models_folder}/rnn_model_{self.env.env_name}_0_trajs_{self.config.run.trajectories}.pt'

    def run(self, state=''):
        # try:
        return self._run(state)
        # except Exception as e:
        #     self.logger.error('Error in LLMatic.run()')
        #     self.logger.error(e)
        #     self.logger.error(traceback.format_exc())
        #     self.save_agent_state(self.sub_messages)
        #     write_files_from_dict(self.file_dict, base_dir=f'{self.folder_path}/{self.name}')
        #     raise e
    
    def _run(self, state=''):
        # Train data
        states_train, actions_train = self.env.train_data
        states_train = torch.tensor(states_train, dtype=torch.float32, device=self.device)
        if actions_train is not None:
            actions_train = torch.tensor(actions_train, dtype=torch.float32, device=self.device)
            states_actions_train = torch.cat([states_train, actions_train], dim=-1)
        else:
            states_actions_train = states_train

        states_val, actions_val = self.env.val_data
        states_val = torch.tensor(states_val, dtype=torch.float32, device=self.device)
        if actions_val is not None:
            actions_val = torch.tensor(actions_val, dtype=torch.float32, device=self.device)
            states_actions_val = torch.cat([states_val, actions_val], dim=-1)
        else:
            states_actions_val = states_val


        # Test data
        states_test, actions_test = self.env.test_data
        states_test = torch.tensor(states_test, dtype=torch.float32, device=self.device)
        if actions_test is not None:
            actions_test = torch.tensor(actions_test, dtype=torch.float32, device=self.device)
            states_actions_test = torch.cat([states_test, actions_test], dim=-1)
        else:
            states_actions_test = states_test

        test_loss = gp_method.discover_multivariate_equations_and_evaluate(states_actions_train.cpu().numpy(), states_actions_test.cpu().numpy(), state_dim=self.state_dim, logger=self.logger, env=self.env, config=self.config)

        self.logger.info(f'[{self.env.env_name} {self.name} {self.env.seed}][Test Run completed successfully] MSE TEST LOSS {test_loss:.4f}')
        return test_loss


class SINDYAgent(Agent):
    def __init__(self, env, config, logger, rate_limiter):
        super().__init__(env, config, logger, rate_limiter)
        self.name = 'SINDY'
        self.PolynomialLibrary_kw = dict(degree=3, interaction_only=True)
        self.sindy_threshold = 0.02
        self.sindy_alpha = 0.5
        
        self.folder_path = f"{self.config.run.log_path.split('.txt')[0]}/{self.env.env_name}/{self.env.seed}/"
        Path(self.folder_path).mkdir(parents=True, exist_ok=True)
        self.compute_val_loss = False

    def run(self, state=''):
        # try:
        return self._run(state)
        # except Exception as e:
        #     self.logger.error('Error in LLMatic.run()')
        #     self.logger.error(e)
        #     self.logger.error(traceback.format_exc())
        #     self.save_agent_state(self.sub_messages)
        #     write_files_from_dict(self.file_dict, base_dir=f'{self.folder_path}/{self.name}')
        #     raise e
    
    def _run(self, state=''):
        # Load model
        self.train_model()
        states_test, actions_test = self.env.test_data

        MSE = torch.nn.MSELoss()
        # Calculate test loss
        pred_states = []
        pred_state = states_test[:,0]
        for t in tqdm(range(states_test.shape[1])):
            pred_states.append(pred_state)
            if actions_test is None:
                dx_dt = self.model.predict([*states_test[:,t:t+1]], multiple_trajectories=True)
            else:
                dx_dt = self.model.predict([*states_test[:,t:t+1]], u=[*actions_test[:,t:t+1]], multiple_trajectories=True)
            dx_dt = np.stack([np.array(dt.data)[0] for dt in dx_dt])
            pred_state = states_test[:,t] + dx_dt
        pred_states = np.stack(pred_states, axis=1)
        test_loss = MSE(torch.tensor(pred_states), torch.tensor(states_test)).item()
        # test_loss = np.mean((pred_states - states_test)**2)
        self.logger.info(f'[{self.env.env_name} {self.name} {self.env.seed}][Test Run completed successfully] MSE TEST LOSS {test_loss:.4f}')

        if self.env.env_name == 'COVID-insight':
            print('')

            (states_val, actions_val) = self.env.val_data
            (states_test, actions_test) = self.env.test_data

            noint_states, noint_actions = states_val, actions_val
            int_states, int_actions = states_test, actions_test

            pred_states = []
            pred_state = int_states[:,19]
            for t in tqdm(range(19, int_states.shape[1])):
                pred_states.append(pred_state)
                if actions_test is None:
                    dx_dt = self.model.predict([*int_states[:,t:t+1]], multiple_trajectories=True)
                else:
                    dx_dt = self.model.predict([*int_states[:,t:t+1]], u=[*actions_test[:,t:t+1]], multiple_trajectories=True)
                dx_dt = np.stack([np.array(dt.data)[0] for dt in dx_dt])
                pred_state = pred_state + dx_dt
            pred_states = np.stack(pred_states, axis=1)

            predicted_sindy_states = np.concatenate((int_states[:,:19,:],pred_states),axis=1)

            np.savez('COVID-intervention-sindy.npz',
                predicted_sindy_states=predicted_sindy_states,
            )

            import matplotlib.pyplot as plt
            # 6, 7?, 9!; 20# with rejig
            # traj_idx = 20
            traj_idx = 9
            # Create a figure and a set of subplots
            fig, axs = plt.subplots(4, 1, sharex=True)
            # Set the overall title for all subplots
            fig.suptitle(f"SEIRD for traj {traj_idx}")
            # Plotting each data in a separate subplot
            axs[0].plot(noint_states[traj_idx,:,0], label="Not Int")
            axs[0].plot(int_states[traj_idx,:,0], label="Int")
            axs[0].plot(predicted_sindy_states[traj_idx,:,0], label="SINDY")
            # axs[0].set_ylim((0.0, 1.0))
            # axs[0].plot(predicted_full_states[traj_idx,:,0], label="NSDT")
            axs[0].legend()
            axs[1].plot(noint_states[traj_idx,:,1], label="Not Int")
            axs[1].plot(int_states[traj_idx,:,1], label="Int")
            axs[1].plot(predicted_sindy_states[traj_idx,:,1], label="SINDY")
            # axs[1].plot(predicted_full_states[traj_idx,:,1], label="NSDT")
            axs[1].legend()
            axs[2].plot(noint_states[traj_idx,:,2], label="Not Int")
            axs[2].plot(int_states[traj_idx,:,2], label="Int")
            axs[2].plot(predicted_sindy_states[traj_idx,:,2], label="SINDY")
            # axs[2].plot(predicted_full_states[traj_idx,:,2], label="NSDT")
            axs[2].legend()
            axs[3].plot(noint_states[traj_idx,:,3], label="Not Int")
            axs[3].plot(int_states[traj_idx,:,3], label="Int")
            axs[3].plot(predicted_sindy_states[traj_idx,:,3], label="SINDY")
            # axs[3].plot(predicted_full_states[traj_idx,:,3].cpu(), label="NSDT")
            axs[3].legend()
            # axs[4].plot(states[traj_idx,:,4], label="Dead")
            # axs[4].legend()
            # axs[4].plot(actions[traj_idx,:,0], label="Intervention")
            # axs[4].legend()
            # Adjust the layout
            plt.tight_layout()
            # Save the figure
            plt.savefig("test.png")
            # plt.clf()
            print('')
            

        # def state_diff(state, action):
        #     # (x0)' = 0.0566147711811175874396795393295178655535 x0 + 0.0310870825182381523010821666730407741852 x1 + -0.0896386953849055700960235526508768089116 u1 + -0.0292010490227662330986468930404953425750 x0 x1 + -0.0331159317075416584352787197076395386830 x0 u1 + 0.0571204561973386315543166347197256982327 x1 u1
        #     # (x1)' = -0.7956578799696311099509671294072177261114 1 + 0.5124591653866369389547230639436747878790 u0
        #     x0, x1 = state[:,0], state[:,1]
        #     u0, u1 = action[:,0], action[:,1]
        #     dx0_dt = 0.0566147711811175874396795393295178655535 * x0 + 0.0310870825182381523010821666730407741852 * x1 + -0.0896386953849055700960235526508768089116 * u1 + -0.0292010490227662330986468930404953425750 * x0 * x1 + -0.0331159317075416584352787197076395386830 * x0 * u1 + 0.0571204561973386315543166347197256982327 * x1 * u1
        #     dx1_dt = -0.7956578799696311099509671294072177261114 * x1 + 0.5124591653866369389547230639436747878790 * u0
        #     return np.stack([dx0_dt, dx1_dt], axis=-1)

        # # Calculate loss with SINDy model saved parameters
        # pred_states = []
        # pred_state = states_test[:,0]
        # for t in tqdm(range(states_test.shape[1])):
        #     pred_states.append(pred_state)
        #     dx_dt = state_diff(states_test[:,t], actions_test[:,t])
        #     pred_state = pred_state + dx_dt
        # pred_states = np.stack(pred_states, axis=1)
        # test_loss = MSE(torch.tensor(pred_states), torch.tensor(states_test)).item()
        # # test_loss = np.mean((pred_states - states_test)**2)
        # self.logger.info(f'[{self.env.env_name} {self.name} {self.env.seed}][Test Run completed successfully] MSE TEST LOSS {test_loss:.4f}')
        return test_loss

    def train_model(self):
        # Train model
        states_train, actions_train = self.env.train_data        
        states_val, actions_val = self.env.val_data
        if 'COVID' in self.env.env_name:
            self.sindy_threshold = 1e-5

        # self.model = ps.SINDy(optimizer=STLSQ(threshold=sindy_threshold, alpha=sindy_alpha, max_iter=100, ridge_kw={'tol': 1e-6}),  differentiation_method=SmoothedFiniteDifference(smoother_kws=smoother_kws, is_uniform=True, order=4), feature_library=PolynomialLibrary(**PolynomialLibrary_kw))
        self.model = ps.SINDy(optimizer=STLSQ(threshold=self.sindy_threshold, alpha=self.sindy_alpha, max_iter=100, ridge_kw={'tol': 1e-6}), feature_library=PolynomialLibrary(**self.PolynomialLibrary_kw))
        
        # self.model = ps.SINDy()
        if actions_train is None:
            self.model.fit([*states_train], multiple_trajectories=True)
        else:
            self.model.fit([*states_train], u=[*actions_train], multiple_trajectories=True)
        # self.model.print()
        model_str = return_sindy_string(self.model)
        self.logger.info(f'[{self.env.env_name} {self.name} {self.env.seed}][Model] {model_str}')

        if self.compute_val_loss:
            # Calculate validation loss
            pred_states = []
            pred_state = states_val[:,0]
            for t in tqdm(range(states_val.shape[1])):
                pred_states.append(pred_state)
                dx_dt = self.model.predict([*states_val[:,t:t+1]], u=[*actions_val[:,t:t+1]], multiple_trajectories=True)
                dx_dt = np.stack([np.array(dt.data)[0] for dt in dx_dt])
                pred_state = states_val[:,t] + dx_dt
            pred_states = np.stack(pred_states, axis=1)
            val_loss = np.mean((pred_states - states_val)**2)
            self.logger.info(f'[{self.name}][COMPLETE] MSE VAL LOSS {val_loss:.4f}')


   

# def return_sindy_string(model, lhs=None, precision=3):
def return_sindy_string(model, lhs=None, precision=40):
    str_out = '\n'
    try:  # Waiting on PEP 690 to lazy import CVXPY
        from pysindy.optimizers import SINDyPI

        sindy_pi_flag = True
    except ImportError:
        sindy_pi_flag = False
    eqns = model.equations(precision)
    if sindy_pi_flag and isinstance(model.optimizer, SINDyPI):
        feature_names = model.get_feature_names()
    else:
        feature_names = model.feature_names
    for i, eqn in enumerate(eqns):
        if model.discrete_time:
            names = "(" + feature_names[i] + ")"
            str_out += names + "[k+1] = " + eqn + '\n'
        elif lhs is None:
            if not sindy_pi_flag or not isinstance(model.optimizer, SINDyPI):
                names = "(" + feature_names[i] + ")"
                str_out += names + "' = " + eqn + '\n'
            else:
                names = feature_names[i]
                str_out += names + " = " + eqn + '\n'
        else:
            str_out += lhs[i] + " = " + eqn + '\n'
    return str_out

def sample_all_groups(lst, n):
    """
    Sample randomly without replacement groups of n from the list until all objects are sampled.
    The last group can be less than n.

    :param lst: The list from which to sample.
    :param n: The size of each group.
    :return: A list of sampled groups.
    """
    random.shuffle(lst)
    groups = [lst[i:i + n] for i in range(0, len(lst), n)]
    return groups