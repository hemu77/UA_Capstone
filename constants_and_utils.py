"""
Shared utilities for the whole project.

Read this file as the "infrastructure layer":
- it defines where outputs are stored
- it knows how to talk to the model provider
- it saves graphs and renders graph images

The network-generation logic lives elsewhere. This file exists so the rest of
the code can reuse one consistent set of helpers.
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
from openai import OpenAI
import random
import json
from PIL import Image
import os
import sys
import time

import plotting

PATH_TO_FOLDER = '.'
PATH_TO_TEXT_FILES = PATH_TO_FOLDER + '/text-files'  # folder holding text files, typically GPT output
PATH_TO_STATS_FILES = PATH_TO_FOLDER + '/stats'  # folder holding stats files, eg, proportion of nodes in giant component
DEFAULT_TEMPERATURE = 0.8
SHOW_PLOTS = False
OPENAI_MODEL_PREFIXES = ('gpt-', 'o1', 'o3', 'o4')

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')


def _read_api_key_file(path='api-key.txt'):
    """
    Read provider keys from api-key.txt if it exists.
    The first line is treated as an OpenAI key and the optional second line as a Llama key.
    """
    if not os.path.exists(path):
        return None, None

    with open(path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    openai_key = lines[0] if len(lines) >= 1 else None
    llama_key = lines[1] if len(lines) >= 2 else None
    return openai_key, llama_key


def load_api_keys():
    """
    Prefer environment-based OpenAI auth and keep api-key.txt as a backward-compatible fallback.
    """
    # Step 1 changed the project so OpenAI-only use is enough to run the repo.
    file_openai_key, file_llama_key = _read_api_key_file()
    openai_key = os.getenv("OPENAI_API_KEY") or file_openai_key
    llama_key = os.getenv("LLAMA_API_KEY") or file_llama_key
    return openai_key, llama_key


def is_openai_model(model):
    return model.startswith(OPENAI_MODEL_PREFIXES)


openai_key, llama_key = load_api_keys()

##########################################
# functions to draw and save networks
##########################################
def draw_and_save_network_plot(G, save_prefix):
    """
    Draw network, save figure.
    """
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=0, k=2*1/np.sqrt(len(G.nodes()))))
    plt.axis("off")  # turn off axis
    axis = plt.gca()
    axis.set_xlim([1.1*x for x in axis.get_xlim()])  # add padding so that node labels aren't cut off
    axis.set_ylim([1.1*y for y in axis.get_ylim()])
    plt.tight_layout()
    fig_path = os.path.join(plotting.PATH_TO_SAVED_PLOTS, f'{save_prefix}.png')
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    print('Saving network drawing in ', fig_path)
    plt.savefig(fig_path)
    plt.close()

def draw_and_save_network_plot_no_labels(G, save_prefix):
    """
    Draw network, save figure.
    """
    # draw network without node labels
    # set small node size
    # set small line width
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=0, k=2*1/np.sqrt(len(G.nodes()))), with_labels=False, node_size=15, width=0.1)
    plt.axis("off")
    fig_path = os.path.join(plotting.PATH_TO_SAVED_PLOTS, f'{save_prefix}.png')
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path)
    plt.close()

def save_network(G, save_prefix):
    """
    Save network as adjlist.
    """
    graph_path = os.path.join(PATH_TO_TEXT_FILES, f'{save_prefix}.adj')
    os.makedirs(os.path.dirname(graph_path), exist_ok=True)
    print('Saving adjlist in ', graph_path)
    nx.write_adjlist(G, graph_path)

def get_node_from_string(s):
    """
    If it is a persona of the form "<name> - <description>", get name; else, assume to be name.
    Replace spaces in name with hyphens, so that we can save to and read from nx adjlist.
    """
    if ' - ' in s:  # seems to be persona
        s = s.split(' - ', 1)[0]
    node = s.replace(' ', '-')
    return node

def prop_nodes_in_giant_component(G):
    """
    Get proportion of nodes in largest conneced component.
    """
    largest_cc = max(nx.connected_components(G.to_undirected()), key=len)
    return len(largest_cc) / len(G.nodes())

def shuffle_dict(dict):
    keys = list(dict.keys())
    random.shuffle(keys)
    shuffled_dict = {}
    for item in keys:
        shuffled_dict[item] = dict[item]
    return shuffled_dict

def combine_plots(folders, plot_names):
    for j, plot_name in enumerate(plot_names):
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        for i, folder in enumerate(folders):
            img_path = os.path.join(folder, plot_name)
            img = Image.open(img_path)
            pairs = [(0, 0), (0, 1), (1, 0), (1, 1)]
            axs[pairs[i]].imshow(img)
            axs[pairs[i]].axis('off')

        plt.tight_layout()
        # save combined plot
        fig_path = os.path.join(os.path.join(plotting.PATH_TO_SAVED_PLOTS), f'{plot_name}_combined_plot.png')
        # save plot
        print('Saving combined plot in ', fig_path)
        plt.savefig(fig_path)
        # close figure
        plt.close()


def load_and_draw_network(path_prefix, nr_networks):


    nr_edges = []
    for i in range(nr_networks):
        G = nx.read_adjlist(f'{path_prefix}-{i}.adj')
        nr_edges.append(len(G.edges()))

        network_name = path_prefix.split('/')[1]
        # make os path
        if not os.path.exists(os.path.join(plotting.PATH_TO_SAVED_PLOTS, f'{network_name}/drawn')):
            os.makedirs(os.path.join(plotting.PATH_TO_SAVED_PLOTS, f'{network_name}/drawn/'))
        draw_and_save_network_plot_no_labels(G, f'{network_name}/drawn/{i}')
    plotting.plot_nr_edges(nr_edges, f'{network_name}')

def draw_list_of_networks(list_of_G, network_name):
    nr_edges = []
    for i in range(len(list_of_G)):
        G = list_of_G[i]
        nr_edges.append(len(G.edges()))

        # make os path
        if not os.path.exists(os.path.join(plotting.PATH_TO_SAVED_PLOTS, f'{network_name}/drawn')):
            os.makedirs(os.path.join(plotting.PATH_TO_SAVED_PLOTS, f'{network_name}/drawn/'))
        draw_and_save_network_plot_no_labels(G, f'{network_name}/drawn/{i}')
    plotting.plot_nr_edges(nr_edges, f'{network_name}')
    
##########################################
# functions to interact with LLMs
##########################################
def get_llm_response(model, messages, savename=None, temp=DEFAULT_TEMPERATURE, verbose=False):
    """
    Call OpenAI API, check for finish reason; if all looks good, return response.
    """
    # Pick the provider path based on the requested model family.
    if is_openai_model(model):
        if not openai_key or len(openai_key) < 10:
            raise ValueError('Missing OpenAI API key. Set OPENAI_API_KEY or add it as the first line of api-key.txt.')
        client = OpenAI(api_key=openai_key)
    else:
        if not llama_key or len(llama_key) < 10:
            raise ValueError('Missing Llama API key. Set LLAMA_API_KEY or add it as the second line of api-key.txt.')
        client = OpenAI(api_key=llama_key, base_url="https://api.llama-api.com")
 
    response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temp)
    
    if savename is not None:
        # Token counts are accumulated in a small JSON file so experiments can
        # be compared later without re-running everything.
        # read json in savename
        # if file exists
        if os.path.exists(savename):
            with open(savename) as f:
                data = json.load(f)
        else:
            data = {"prompt_tokens": 0, "completion_tokens": 0}

        data["prompt_tokens"] += response.usage.prompt_tokens
        data["completion_tokens"] += response.usage.completion_tokens

        # save to savename
        with open(savename, 'w') as f:
            json.dump(data, f)

    response = response.choices[0]
    finish_reason = response.finish_reason
    if finish_reason != 'stop':
        if is_openai_model(model):
            raise Exception(f'Finish reason: {finish_reason}\nResponse: {response.message.content}')
        else:  # for some reason Llama produces max_token a lot even though the full answer is coming out
            print(f'Warning: finish reason was {finish_reason}\nResponse: {response.message.content}')
        
    if verbose:
        for m in messages:
            print(m['role'].upper())
            print(m['content'])
            print()
        print('\nRESPONSE')
        print(response.message.content)
    return response.message.content
        

def repeat_prompt_until_parsed(model, system_prompt, user_prompt, parse_method,
                               parse_args, max_tries=10, temp=DEFAULT_TEMPERATURE, verbose=False):
    """
    Helper function to repeat API call and parsing until it works.
    Works with any generic parse_method, where 'response' must be one of its args,
    and additional parse_args.
    """
    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    assert user_prompt is not None
    messages.append({"role": "user", "content": user_prompt})
    
    num_tries = 1
    while num_tries <= max_tries:
        try:
            response = get_llm_response(model, messages, temp=temp, verbose=verbose)
            try:
                parse_args['response'] = response
                parse_out = parse_method(**parse_args)
                return parse_out, response, num_tries
            except Exception as e:
                # Bad format is treated as a recoverable error: show the model
                # what went wrong and ask again.
                print('Failed to parse response:', e)
                for m in messages:
                    print(m['role'].upper())
                    print(m['content'])
                    print()
                print('\nRESPONSE:')
                print(response)
                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": f"Invalid response: {e}! Re-answer using only the provided persona IDs. Do not use ages, counts, or any other numbers as IDs.",
                })
        except Exception as e:
            print('Failed to get response:', e)
        num_tries += 1
        time.sleep(1)
    raise Exception(f'Exceed max tries of {max_tries}')
       

def compute_token_cost(savepath, nr_networks, model='gpt-3.5-turbo'):

    prompt_tokens = []
    completion_tokens = []
    for i in range(nr_networks):
        with open(f'{savepath}-{i}.json') as f:
            data = json.load(f)
            prompt_tokens.append(data['prompt_tokens'])
            completion_tokens.append(data['completion_tokens'])

    # print averages and std
    print(f'Files in {savepath}: {nr_networks}')
    print(f'Prompt tokens: {np.mean(prompt_tokens)} +- {np.std(prompt_tokens)}')
    print(f'Completion tokens: {np.mean(completion_tokens)} +- {np.std(completion_tokens)}')

    # pricing
    if model == 'gpt-3.5-turbo':
        prompt_cost = 0.0005/1000
        completion_cost = 0.0015/1000
        costs = [prompt_cost*pt + completion_cost*ct for pt, ct in zip(prompt_tokens, completion_tokens)]
        print(f'Cost in dollars: {np.mean(costs)} +- {np.std(costs)}')

    else:
        print("Model cost unknown")

if __name__ == '__main__':

    compute_token_cost('costs/cost_all-at-once-for_us_50-gpt-3.5-turbo', 15)
    compute_token_cost('costs/cost_llm-as-agent-for_us_50-gpt-3.5-turbo', 15)
    compute_token_cost('costs/cost_one-by-one-for_us_50-gpt-3.5-turbo', 15)

    combine_plots(['plots/all-at-once-for_us_50-gpt-3.5-turbo', 'plots/llm-as-agent-for_us_50-gpt-3.5-turbo', 'plots/one-by-one-for_us_50-gpt-3.5-turbo', 'plots/real'],
                  ['betweenness_centrality_hist.png', 'degree_centrality_hist.png', 'closeness_centrality_hist.png'])


    load_and_draw_network('text-files/all-at-once-for_us_50-gpt-3.5-turbo', 15)
    load_and_draw_network('text-files/llm-as-agent-for_us_50-gpt-3.5-turbo', 15)
    load_and_draw_network('text-files/one-by-one-for_us_50-gpt-3.5-turbo', 15)
