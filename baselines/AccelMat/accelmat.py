import os,sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import json
from openai import OpenAI
from typing import List
import re
import requests
from experiment_agent import ExperimentAgent
from optimizer import run_mcts, compute_average_exploration_distance

client_gpt = OpenAI(
    base_url="OpenAI_BASE_URL",
    api_key="OpenAI_API_KEY",
)

client_deepseek = OpenAI(
    base_url="DeepSeek_BASE_URL",
    api_key="DeepSeek_API_KEY",
)

client_qwen = OpenAI(
    base_url="Qwen_BASE_URL",
    api_key="Qwen_API_KEY"
)

semantic_api = "Semantic_API_KEY"

goal_statement = """
    Find the structural parameters corresponding to the strongest chirality (g-factor characteristics) in the nanohelix material system.
"""

constraint_list = """
    Explicitly show the underlying physicochemical principles regarding the structure and property relationships.
"""

def get_semantic_scholar_data(query, api_key, offset=0, limit=10):
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    headers = {"x-api-key": api_key} if api_key else {}
    params = {
        "query": query,
        "offset": offset,
        "limit": limit,
        "fields": "title,abstract,year,citationCount,authors,url,publicationTypes"  # you can change this fields.
    }

    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error of RequestException: {str(e)}")
        return None

def search_literature(goal, constraints):
    query_prompt = [
        {"role": "system",
         "content": "Now you are an expert in generating search keywords for scientific database queries. Your task is to use refined research goals and research constraints to create precise and effective search queries for the Semantic Scholar API in the required format."},
        {"role": "user", "content": """Here, I need you to generate search queries for a literature review. Please follow these REQUIREMENTS:
            1. Use the provided clarified research goal and constraints to identify relevant search queries.
            2. Ensure the output is formatted as a single search string separated by commas, suitable for the Semantic Scholar API.
            3. Maintain brevity and precision, using domain-specific terms.
            4. Ensure the search terms cover both the research goal and constraints effectively.
            5. The query words you suppose should be AS LESS AS POSSIBLE, as Semantic Scholar may cannot find some enough literatures with too many constraints."""},
        {"role": "user", "content": f"""Clarified Research Goal: {goal}
            Clarified Research Constraints: {constraints}"""}
    ]

    query = client_gpt.chat.completions.create(
        model="gpt-4o",
        messages=query_prompt,
        max_tokens=250,
        temperature=0,
    ).choices[0].message.content.strip()

    semantic_results = get_semantic_scholar_data(query, api_key=semantic_api)
    if semantic_results:
        prompt = [
            {"role": "system",
             "content": "You are an expert in summarizing literature review results from scientific database searches. Your task is to process and summarize results retrieved from the Semantic Scholar API, focusing on the **mechanisms** for hypothesis generation."},
            {"role": "user",
             "content": """Here, I would like to summarize the search results from a literature review. The summaries should focus on the **mechanisms** and their **impact**. Please adhere to the following REQUIREMENTS:
                        1. Include the article title, authors, and publication year.
                        2. Provide a 1-2 sentence summary of the article's focus on **mechanisms**, specifically how different factors or processes affect related material properties.
                        3. Use precise scientific language to ensure clarity and relevance.
                        4. Avoid including unrelated details; prioritize findings directly tied to the effects on mechanisms.
                        5. Format the summaries for easy reference and further exploration."""},
            {"role": "user",
             "content": f"""Search Results Provided: {semantic_results}"""}
        ]
        response = client_gpt.chat.completions.create(
            model="gpt-4o",
            messages=prompt,
            max_tokens=1000,
            temperature=0
        ).choices[0].message.content.strip()
        print(f"Literature review summary results: {response}")
        return response
    else:
        return None

def construct_prompt_for_hypotheses_generator(goal_statement:str, constraint_list:str, literature_response:str):
    prompt = f"""{goal_statement} \n\n Constraints:- \n{constraint_list}. Literature Insights: {literature_response}\n
        Provide me 20 innovative suggestions that will help achieve the above goal while satisfying all of the above mentioned constraints strictly. 
        Parameter space:
          angle: [0.123160654, 1.009814211]
          curl: [0.628318531, 8.078381109]
          fiber_radius: [20, 60]
          height: [43.32551229, 954.9296586]
          helix_radius: [20, 90]
          n_turns: [3, 10]
          pitch: [60, 200]
          total_fiber_length: [303.7757835, 1127.781297]
          total_length: [300, 650]

        Provide reason for each suggestion. The suggestions must be in the below mentioned format in a JSON object. For example:\n
        {{Suggestion_1: 
            Materials: 
            Methods_to_develop_the_materials_suggested: 
            Reasoning:
            ,
        Suggestion_20: 
            Materials: 
            Methods_to_develop_the_materials_suggested: 
            Reasoning: }}
    """
    return prompt

def construct_critic_prompt(goal_statement:str, constraint_list:str, chat_history:str):
    critic_prompt = f"""{goal_statement}\n\nConstraints:-\n{constraint_list}\n\nSuggestions:\n{chat_history}Given the above goal statement, constraints and suggestions about materials design and discovery, evaluate each suggestion and generate detailed feedback which will help the suggestion generation process to generate suggestions such that they help achieve goal statement and satisfy all the constraints strictly. The detailed feedback should be in the below JSON format strictly:
        {{"Feedback_for_suggestion_1":
        Meets_the_goal_statement_and_satisfies_all_constraints_strictly: "YES/NO"
        Reasoning:" ",
        "Feedback_for_suggestion_20":
        Meets_the_goal_statement_and_satisfies_all_constraints_strictly: "YES/NO"
        Reasoning:" ",
        "Overall_Feedback_for_improvement": " " ]]
        }}
    """
    return critic_prompt

def construct_feedback_prompt(feedback):
    feedback_prompt = f"""Below provided is the feedback you gave for each of the initial suggestions generated and an overall feedback for the improvement of future suggestion generations\n{feedback}.Refine your suggestions based on the feedback accordingly to meet the goal statement and satisfy all the constraints strictly. The suggestions must be in the below mentioned format in a JSON object. 
    For example:\n
    {{Suggestion_1: 
        Materials: 
        Methods_to_develop_the_materials_suggested: 
        Reasoning:
        ,
    Suggestion_20: 
        Materials: 
        Methods_to_develop_the_materials_suggested: 
        Reasoning:}}"""
    return feedback_prompt

def construct_feedback_prompt_for_refined_hypotheses(feedback_history, chat_history):
    feedback_prompt = f"""Below provided is the feedback you gave for the initial suggestions\n{feedback_history}. Below are the refined suggestions based on the feedback\n{chat_history}. Now evaluate each refined suggestion and provide detailed feedback which will help the suggestion generation process to generate suggestions such that they help achieve goal statement and satisfy all the constraints strictly.
    parameter_space:
          angle: [0.123160654, 1.009814211]
          curl: [0.628318531, 8.078381109]
          fiber_radius: [20, 60]
          height: [43.32551229, 954.9296586]
          helix_radius: [20, 90]
          n_turns: [3, 10]
          pitch: [60, 200]
          total_fiber_length: [303.7757835, 1127.781297]
          total_length: [300, 650]

     The detailed feedback should be in the below JSON format strictly:
        {{"Feedback_for_suggestion_1":
        Meets_the_goal_statement_and_satisfies_all_constraints_strictly: "YES/NO"
        Reasoning:" ",
        "Feedback_for_suggestion_20":
        Meets_the_goal_statement_and_satisfies_all_constraints_strictly: "YES/NO"
        Reasoning:" ",
        "Overall_Feedback_for_improvement": " " ]]
        }}
    """
    return feedback_prompt

def expert_list_generator(goal_statement):
    completion = client_gpt.chat.completions.create(
        model = 'gpt-4o',
        temperature = 0.7,
        messages = [
            {
                'role': 'system',
                'content': f'You are an helpful assistant'
            },
            {
                'role': 'user',
                'content': f'Generate a list of experts required to achieve the below mentioned goal:\n{goal_statement}. Just list the top 5 experts in the format "Expert_1, Expert_2, Expert_3, Expert_4, Expert_5"'
                }
              ]
    )
    return completion.choices[0].message.content

def hypothesis_generator(expert_list, prompt, feedback=None, chat_history=None):
    if feedback == None and chat_history == None:
        completion = client_gpt.chat.completions.create(
            model = 'gpt-4o',
            temperature = 0.7,
            messages = [
                {
                    'role': 'system',
                    'content': f'You are an innovative {expert_list} capable of doing impactful materials discovery and design'
                },
                {
                    'role': 'user',
                    'content': prompt
                },
            ],
            response_format = {"type": "json_object"}
        )
    else:
        completion = client_gpt.chat.completions.create(
            model = 'gpt-4o',
            temperature = 0.7,
            messages = [
                {
                    'role': 'system',
                    'content': f'You are an innovative {expert_list} capable of doing impactful materials discovery and design'
                },
                {
                    'role': 'user',
                    'content': prompt
                },
                {
                    'role': 'assistant',
                    'content': chat_history
                },
                {
                    'role': 'user',
                    'content': feedback
                }
            ],
            response_format = {"type": "json_object"}
        )
    return completion.choices[0].message.content

def critic_1(expert_list,critic_prompt,feedback_history,refined_feedback_prompt):
    if feedback_history==None and refined_feedback_prompt==None:
        completion = client_gpt.chat.completions.create(
            model = 'gpt-4o',
            temperature = 0.7,
            messages = [
                {
                    'role': 'system',
                    'content': f'You are an expert {expert_list} capable of doing impactful materials discovery and design. Given a goal statement, additional constraints, and a list of suggestions about materials design and discovery, your task is to evaluate each suggestion such that it meets the goal statement and satisfies all the constraints strictly. '
                },
                {
                    'role': 'user',
                    'content': critic_prompt
                }
            ],
            response_format = {"type": "json_object"}
        )
    else:
        completion = client_gpt.chat.completions.create(
        model = 'gpt-4o',
        temperature = 0.7,
        messages = [
            {
                'role': 'system',
                'content': f'You are an expert {expert_list} capable of doing impactful materials discovery and design. Given a goal statement, additional constraints, and a list of suggestions about materials design and discovery, your task is to evaluate each suggestion such that it meets the goal statement and satisfies all the constraints strictly.'
            },
            {
                'role': 'user',
                'content': critic_prompt
            },
            {
                'role': 'assistant',
                'content': feedback_history
            },
            {
                'role': 'user',
                'content': refined_feedback_prompt
            }
        ],
        response_format = {"type": "json_object"}
    )
    return completion.choices[0].message.content

def critic_2(expert_list,critic_prompt,feedback_history,refined_feedback_prompt):
    if feedback_history==None and refined_feedback_prompt==None:
        completion = client_deepseek.chat.completions.create(
            model = 'deepseek-chat',
            temperature = 0.7,
            messages = [
                {
                    'role': 'system',
                    'content': f'You are an expert {expert_list} capable of doing impactful materials discovery and design. Given a goal statement, additional constraints, and a list of suggestions about materials design and discovery, your task is to evaluate each suggestion such that it meets the goal statement and satisfies all the constraints strictly. '
                },
                {
                    'role': 'user',
                    'content': critic_prompt
                }
            ],
            response_format = {"type": "json_object"}
        )
    else:
        completion = client_deepseek.chat.completions.create(
        model = 'deepseek-chat',
        temperature = 0.7,
        messages = [
            {
                'role': 'system',
                'content': f'You are an expert {expert_list} capable of doing impactful materials discovery and design. Given a goal statement, additional constraints, and a list of suggestions about materials design and discovery, your task is to evaluate each suggestion such that it meets the goal statement and satisfies all the constraints strictly.'
            },
            {
                'role': 'user',
                'content': critic_prompt
            },
            {
                'role': 'assistant',
                'content': feedback_history
            },
            {
                'role': 'user',
                'content': refined_feedback_prompt
            }
        ],
        response_format = {"type": "json_object"}
    )
    return completion.choices[0].message.content

def critic_3(expert_list, critic_prompt, feedback_history, refined_feedback_prompt):
    if feedback_history==None and refined_feedback_prompt==None:
        completion = client_qwen.chat.completions.create(
            model = 'qwen2.5-14b-instruct-1m',
            temperature = 0.7,
            messages = [
                {
                    'role': 'system',
                    'content': f'You are an expert {expert_list} capable of doing impactful materials discovery and design. Given a goal statement, additional constraints, and a list of suggestions about materials design and discovery, your task is to evaluate each suggestion such that it meets the goal statement and satisfies all the constraints strictly. '
                },
                {
                    'role': 'user',
                    'content': critic_prompt
                }
            ],
            response_format={"type": "json_object"}
        )
    else:
        completion = client_qwen.chat.completions.create(
            model = 'qwen2.5-14b-instruct-1m',
            temperature = 0.7,
            messages = [
                {
                    'role': 'system',
                    'content': f'You are an expert {expert_list} capable of doing impactful materials discovery and design. Given a goal statement, additional constraints, and a list of suggestions about materials design and discovery, your task is to evaluate each suggestion such that it meets the goal statement and satisfies all the constraints strictly.'
                },
                {
                    'role': 'user',
                    'content': critic_prompt
                },
                {
                    'role': 'assistant',
                    'content': feedback_history
                },
                {
                    'role': 'user',
                    'content': refined_feedback_prompt
                }
            ],
            response_format={"type": "json_object"}
        )
    return completion.choices[0].message.content

def json_to_text(json_obj):
    output_text = ""
    for key, value in json_obj.items():
        suggestion_details = (
            f"{key.replace('_', ' ')}:\n"
            f"Materials:{''.join(value['Materials'])}\n"
            f"Methods_to_develop_the_materials_suggested:{''.join(value['Methods_to_develop_the_materials_suggested'])}\n"
            f"Reasoning:{value[f'Reasoning']}\n\n"
        )
        output_text += suggestion_details
    return output_text

def process_feedback_extract_final_answer(feedback):
    suggestions_with_no = 0
    processed_feedback = ""
    final_answer = "Yes"
    for key, value in feedback.items():
        if key.startswith("Feedback_for_suggestion"):
            suggestion_num = key.split('_')[-1]
            Meets_the_goal_statement_and_satisfies_all_constraints_strictly = value.get('Meets_the_goal_statement_and_satisfies_all_constraints_strictly', 'N/A')
            if Meets_the_goal_statement_and_satisfies_all_constraints_strictly == 'NO':
                suggestions_with_no += 1
                final_answer = "NO"
            Reasoning = value.get('Reasoning', 'N/A')
            processed_feedback += f"Feedback_for_suggestion_{suggestion_num}:\nMeets_the_goal_statement_and_satisfies_all_constraints_strictly:{Meets_the_goal_statement_and_satisfies_all_constraints_strictly}.\nReasoning: {Reasoning}\n\n"
        elif key.startswith("Overall_Feedback_for_improvement"):
            processed_feedback += f"Overall Feedback_for_future_suggestion_improvement: {value}\n"
    return processed_feedback, final_answer, suggestions_with_no

exp_agent = ExperimentAgent()
experiments_data = []
chat_history = []
feedback_history = []
initial_feedback = None
final_answer = ""
feedback_prompt = None
refined_feedback_prompt = None
text_from_feedback = None
suggestions_no_list = []
expert_list = expert_list_generator(goal_statement)
literature_results = search_literature(goal_statement, constraint_list)

prompt = construct_prompt_for_hypotheses_generator(goal_statement, constraint_list, literature_results)
print(f'prompt----->')
generated_hypotheses = hypothesis_generator(expert_list,prompt, feedback_prompt)
generated_hypotheses = json.loads(generated_hypotheses)
print(f'generated_hypotheses----->')

exp_agent.initialize(generated_hypotheses)
run_mcts(exp_agent, iterations=100)
avg_distance = compute_average_exploration_distance(exp_agent.param_history)
print("Average Exploration Distance:", avg_distance)
print("Experiment Result:", max(exp_agent.g_factor))
experiments_data.append({
    "Hypothesis": generated_hypotheses,
    "ParameterHistory": exp_agent.param_history[:],
    "GFactorHistory": exp_agent.g_factor[:],
    "BestGFactorSoFar": max(exp_agent.g_factor),
    "AverageExplorationDistance": avg_distance
})

if len(generated_hypotheses.keys())==20:
    generated_hypotheses = json_to_text(generated_hypotheses)
    chat_history.append(generated_hypotheses)
    critic_prompt = construct_critic_prompt(goal_statement, constraint_list, chat_history[-1])
    feedback_from_critic_1 = critic_1(expert_list,critic_prompt, initial_feedback, refined_feedback_prompt)
    feedback_from_critic_1 = json.loads(feedback_from_critic_1)
    feedback_from_critic_1, final_answer_1, suggestions_with_no_1 = process_feedback_extract_final_answer(feedback_from_critic_1)
    feedback_from_critic_2 = critic_2(expert_list,critic_prompt, initial_feedback, refined_feedback_prompt)
    feedback_from_critic_2 = json.loads(feedback_from_critic_2)
    feedback_from_critic_2, final_answer_2, suggestions_with_no_2 = process_feedback_extract_final_answer(feedback_from_critic_2)
    feedback_from_critic_3 = critic_3(expert_list,critic_prompt, initial_feedback, refined_feedback_prompt)
    feedback_from_critic_3 = json.loads(feedback_from_critic_3)
    feedback_from_critic_3, final_answer_3, suggestions_with_no_3 = process_feedback_extract_final_answer(feedback_from_critic_3)
    feedback_history.append(feedback_from_critic_1)
    feedback_history.append(feedback_from_critic_2)
    feedback_history.append(feedback_from_critic_3)
    print('suggestions_with_no----->',suggestions_with_no_1, suggestions_with_no_2)
    print(f'final_answer-----> {final_answer}')
    attempts = 0
    while not (final_answer_1 == "Yes" and final_answer_2 == "Yes" and final_answer_3 == "Yes") and attempts<5:
        feedback_prompt = construct_feedback_prompt(feedback_from_critic_1)
        print('==============>Constructing feedback prompt for refined hypotheses')
        refined_hypotheses = hypothesis_generator(expert_list, prompt, feedback_prompt, chat_history[-1])
        refined_hypotheses = json.loads(refined_hypotheses)
        print('===================>refined hypothesis generated')
        chat_history.append(json_to_text(refined_hypotheses))

        exp_agent.initialize(refined_hypotheses)
        run_mcts(exp_agent, iterations=100)
        avg_distance = compute_average_exploration_distance(exp_agent.param_history)
        print("Average Exploration Distance:", avg_distance)
        print("Experiment Result:", max(exp_agent.g_factor))
        experiments_data.append({
            "Hypothesis": refined_hypotheses,
            "ParameterHistory": exp_agent.param_history[:],
            "GFactorHistory": exp_agent.g_factor[:],
            "BestGFactorSoFar": max(exp_agent.g_factor),
            "AverageExplorationDistance": avg_distance
        })

        feedback_prompt_for_refined_hypothesis = construct_feedback_prompt_for_refined_hypotheses(feedback_history[-1], chat_history[-1])
        print('===================>Constructing feedback prompt for refined hypotheses')
        refined_feedback = critic_1(expert_list, critic_prompt, feedback_history[-1], feedback_prompt_for_refined_hypothesis)
        refined_feedback = json.loads(refined_feedback)
        print('===================>feedback received for refined hypotheses')
        feedback_from_critic_1, final_answer_1, suggestions_with_no_1 = process_feedback_extract_final_answer(refined_feedback)
        feedback_history.append(feedback_from_critic_1)
        suggestions_no_list.append(suggestions_with_no_1)
        print(f'suggestions_with_no-----> {suggestions_with_no_1}')
        refined_feedback = critic_2(expert_list, critic_prompt, feedback_history[-2], feedback_prompt_for_refined_hypothesis)
        refined_feedback = json.loads(refined_feedback)
        print('===================>feedback received for refined hypotheses')
        feedback_from_critic_2, final_answer_2, suggestions_with_no_2 = process_feedback_extract_final_answer(refined_feedback)
        feedback_history.append(feedback_from_critic_2)
        suggestions_no_list.append(suggestions_with_no_2)
        print(f'suggestions_with_no-----> {suggestions_with_no_2}')
        refined_feedback = critic_3(expert_list, critic_prompt, feedback_history[-3], feedback_prompt_for_refined_hypothesis)
        refined_feedback = json.loads(refined_feedback)
        print('===================>feedback received for refined hypotheses')
        feedback_from_critic_3, final_answer_3, suggestions_with_no_3 = process_feedback_extract_final_answer(refined_feedback)
        feedback_history.append(feedback_from_critic_3)
        suggestions_no_list.append(suggestions_with_no_3)
        print(f'suggestions_with_no-----> {suggestions_with_no_3}')
        attempts += 1
        print(f'attempts-----> {attempts}')
    if final_answer=="Yes":
        print("Suggestions are generated properly")
else:
    print("Suggestions are not generated properly. Please try again")

with open('accelmat_chat.json', 'w') as f:
    json.dump(chat_history, f)
with open('accelmat_feedback.json', 'w') as f:
    json.dump(feedback_history, f)

with open('accelmat_experiments.json', 'w') as f:
    json.dump(experiments_data, f, indent=2)