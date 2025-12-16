import time
import json
from datetime import datetime
from preprocessing import classify_intent, extract_entities
from rag import retrieve
from llm import chat, LLM_MODELS, ANSWER_QUERY



MODELS_TO_COMPARE = ['qwen', 'openai', 'meta']

TEST_CASES = [
    {
        'id': 1,
        'query': "Which flights from LAX to IAX have the most delays?",
        'expected_intent': 'most_delayed_flights',
        'expected_entities': {'origin': 'LAX', 'destination': 'IAX', 'flight_number': None, 'code': None, 'limit': 5},
        'description': 'Flight delay analysis on a specific route'
    },
    {
        'id': 2,
        'query': "What are the least crowded flights from ORX to IAX?",
        'expected_intent': 'least_crowded_flights',
        'expected_entities': {'origin': 'ORX', 'destination': 'IAX', 'flight_number': None, 'code': None, 'limit': 5},
        'description': 'Passenger crowd analysis on a route'
    },
    {
        'id': 3,
        'query': "Show me the top 3 routes with the worst food",
        'expected_intent': 'worst_food_routes',
        'expected_entities': {'origin': None, 'destination': None, 'flight_number': None, 'code': None, 'limit': 3},
        'description': 'Food satisfaction ranking'
    },
    {
        'id': 4,
        'query': "What are the most popular airports?",
        'expected_intent': 'popular_airports',
        'expected_entities': {'origin': None, 'destination': None, 'flight_number': None, 'code': None, 'limit': 5},
        'description': 'Airport popularity ranking'
    },
    {
        'id': 5,
        'query': "What classes are available on flight 2411?",
        'expected_intent': 'flight_classes_offered',
        'expected_entities': {'origin': None, 'destination': None, 'flight_number': 2411, 'code': None, 'limit': 5},
        'description': 'Flight class information lookup'
    },
    # {
    #     'id': 6,
    #     'query': "What is the flight distance from DFX to ORX?",
    #     'expected_intent': 'route_distance',
    #     'expected_entities': {'origin': 'DFX', 'destination': 'ORX', 'flight_number': None, 'code': None, 'limit': 5},
    #     'description': 'Route distance calculation'
    # },
    # {
    #     'id': 7,
    #     'query': "What is the dominant passenger generation at LAX?",
    #     'expected_intent': 'dominant_generation_airport',
    #     'expected_entities': {'origin': None, 'destination': None, 'flight_number': None, 'code': 'LAX', 'limit': 5},
    #     'description': 'Passenger generation demographics at airport'
    # },
    # {
    #     'id': 8,
    #     'query': "Can I go from DEX to LAX directly?",
    #     'expected_intent': 'direct_flight',
    #     'expected_entities': {'origin': 'DEX', 'destination': 'LAX', 'flight_number': None, 'code': None, 'limit': 5},
    #     'description': 'Direct flight availability check'
    # },
    # {
    #     'id': 9,
    #     'query': "Which aircraft will I be flying on flight 924?",
    #     'expected_intent': 'flight_aircraft',
    #     'expected_entities': {'origin': None, 'destination': None, 'flight_number': 924, 'code': None, 'limit': 5},
    #     'description': 'Aircraft type lookup'
    # },
    # {
    #     'id': 10,
    #     'query': "Do frequent flyers use the route from EWX to IAX?",
    #     'expected_intent': 'frequent_flyers_route',
    #     'expected_entities': {'origin': 'EWX', 'destination': 'IAX', 'flight_number': None, 'code': None, 'limit': 5},
    #     'description': 'Frequent flyer route usage'
    # }
]



def estimate_tokens(text: str) -> int:
    return len(text) // 4

def check_entities_match(extracted: dict, expected: dict) -> bool:
    if not extracted or not expected: return False
    
    for key in ['origin', 'destination', 'flight_number', 'code']:
        extracted_val = extracted.get(key)
        expected_val = expected.get(key)
        
        if extracted_val in [None, 'null', 'NULL', '']: extracted_val = None
        if expected_val in [None, 'null', 'NULL', '']: expected_val = None
            
        if extracted_val != expected_val: return False
    
    return True

def run_single_test(model: str, query: str) -> dict:
    metrics = {
        'model': model,
        'query': query,
        'success': False,
        'error': None,
        'intent': None,
        'entities': None,
        'intent_time': 0,
        'entities_time': 0,
        'retrieval_time': 0,
        'answer_time': 0,
        'total_time': 0,
        'answer': None,
        'answer_length': 0,
        'estimated_tokens': 0
    }
    
    total_start = time.time()
    
    try:
        # Step 1: Classify intent
        intent_start = time.time()
        intent = classify_intent(query, model=model)
        metrics['intent_time'] = time.time() - intent_start
        metrics['intent'] = intent
        
        # Step 2: Extract entities
        entities_start = time.time()
        entities = extract_entities(intent, query, model=model)
        metrics['entities_time'] = time.time() - entities_start
        metrics['entities'] = entities
        
        # Step 3: Retrieve context
        retrieval_start = time.time()
        results = retrieve(intent=intent, entities=entities, query=query, embedder=None)
        metrics['retrieval_time'] = time.time() - retrieval_start
        
        baseline_results = results.get('baseline', [])
        baseline_context = ""
        if baseline_results and isinstance(baseline_results, list):
            valid_baseline = [x for x in baseline_results if not ('error' in x or 'message' in x)]
            if valid_baseline:
                baseline_context = '\n'.join([str(x) for x in valid_baseline])
            else:
                baseline_context = str(baseline_results[0]) if baseline_results else "No baseline results."
        else:
            baseline_context = "No baseline results."
        
        embedding_context = "No embedding results."
        
        answer_start = time.time()
        answer = chat(
            ANSWER_QUERY.format(baseline_context=baseline_context, embedding_context=embedding_context, query=query),
            model=model,
            temperature=0.5
        )

        metrics['answer_time'] = time.time() - answer_start
        metrics['answer'] = answer
        metrics['answer_length'] = len(answer)
        metrics['estimated_tokens'] = estimate_tokens(answer)
        metrics['success'] = True
        
    except Exception as e:
        metrics['error'] = str(e)
        metrics['success'] = False
    
    metrics['total_time'] = time.time() - total_start
    return metrics

def run_comparison():
    print("=" * 80)
    print("AIRLINE GRAPH-RAG MODEL COMPARISON")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Models: {', '.join(MODELS_TO_COMPARE)}")
    print(f"Test Cases: {len(TEST_CASES)}")
    print("=" * 80)
    
    all_results = []
    
    for test_case in TEST_CASES:
        print(f"\n{'─' * 80}")
        print(f"TEST CASE {test_case['id']}: {test_case['description']}")
        print(f"Query: \"{test_case['query']}\"")
        print(f"Expected Intent: {test_case['expected_intent']}")
        print("─" * 80)
        
        case_results = {'test_case': test_case, 'model_results': []}
        
        for model in MODELS_TO_COMPARE:
            print(f"\n  Running with {model.upper()}...")
            metrics = run_single_test(model, test_case['query'])
            case_results['model_results'].append(metrics)
            
            if metrics['success']:
                print(f"    ✓ Success")
                print(f"    Intent: {metrics['intent']} (correct: {metrics['intent'] == test_case['expected_intent']})")
                entities_correct = check_entities_match(metrics['entities'], test_case['expected_entities'])
                print(f"    Entities: {metrics['entities']} (correct: {entities_correct})")
                print(f"    Times: Intent={metrics['intent_time']:.2f}s, Entities={metrics['entities_time']:.2f}s, "
                      f"Retrieval={metrics['retrieval_time']:.2f}s, Answer={metrics['answer_time']:.2f}s")
                print(f"    Total Time: {metrics['total_time']:.2f}s")
                print(f"    Answer Length: {metrics['answer_length']} chars (~{metrics['estimated_tokens']} tokens)")
            else:
                print(f"    ✗ Failed: {metrics['error']}")
            
            print(f"    Waiting 15 seconds before next model...\n")
            time.sleep(15)
        
        all_results.append(case_results)
    
    return all_results


def print_summary(all_results: list):
    print("\n" + "=" * 80)
    print("QUANTITATIVE SUMMARY")
    print("=" * 80)
    
    # Aggregate metrics per model
    model_aggregates = {model: {
        'total_time': 0,
        'intent_time': 0,
        'entities_time': 0,
        'answer_time': 0,
        'answer_length': 0,
        'estimated_tokens': 0,
        'success_count': 0,
        'intent_correct': 0,
        'entities_correct': 0,
        'test_count': 0
    } for model in MODELS_TO_COMPARE}
    
    for case_result in all_results:
        expected_intent = case_result['test_case']['expected_intent']
        expected_entities = case_result['test_case']['expected_entities']
        for model_result in case_result['model_results']:
            model = model_result['model']
            model_aggregates[model]['test_count'] += 1
            if model_result['success']:
                model_aggregates[model]['success_count'] += 1
                model_aggregates[model]['total_time'] += model_result['total_time']
                model_aggregates[model]['intent_time'] += model_result['intent_time']
                model_aggregates[model]['entities_time'] += model_result['entities_time']
                model_aggregates[model]['answer_time'] += model_result['answer_time']
                model_aggregates[model]['answer_length'] += model_result['answer_length']
                model_aggregates[model]['estimated_tokens'] += model_result['estimated_tokens']
                if model_result['intent'] == expected_intent:
                    model_aggregates[model]['intent_correct'] += 1
                if check_entities_match(model_result['entities'], expected_entities):
                    model_aggregates[model]['entities_correct'] += 1
    
    # Print summary table
    print(f"\n{'Model':<12} {'Success':<10} {'Intent Acc':<12} {'Entity Acc':<12} {'Avg Time':<12} {'Avg Tokens':<12}")
    print("-" * 72)
    
    for model in MODELS_TO_COMPARE:
        agg = model_aggregates[model]
        success_rate = f"{agg['success_count']}/{agg['test_count']}"
        intent_acc = f"{agg['intent_correct']}/{agg['test_count']}" if agg['success_count'] > 0 else "N/A"
        entity_acc = f"{agg['entities_correct']}/{agg['test_count']}" if agg['success_count'] > 0 else "N/A"
        avg_time = f"{agg['total_time']/agg['success_count']:.2f}s" if agg['success_count'] > 0 else "N/A"
        avg_tokens = f"{agg['estimated_tokens']//agg['success_count']}" if agg['success_count'] > 0 else "N/A"
        
        print(f"{model:<12} {success_rate:<10} {intent_acc:<12} {entity_acc:<12} {avg_time:<12} {avg_tokens:<12}")
    
    print("\n" + "=" * 80)
    print("DETAILED TIMING BREAKDOWN")
    print("=" * 80)
    print(f"\n{'Model':<12} {'Intent':<12} {'Entities':<12} {'Answer':<12} {'Total':<12}")
    print("-" * 60)
    
    for model in MODELS_TO_COMPARE:
        agg = model_aggregates[model]
        if agg['success_count'] > 0:
            n = agg['success_count']
            print(f"{model:<12} {agg['intent_time']/n:.2f}s{'':<6} {agg['entities_time']/n:.2f}s{'':<6} "
                  f"{agg['answer_time']/n:.2f}s{'':<6} {agg['total_time']/n:.2f}s")
        else:
            print(f"{model:<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}")


def print_answers_for_qualitative_review(all_results: list):
    print("\n" + "=" * 80)
    print("ANSWERS FOR QUALITATIVE EVALUATION")
    print("=" * 80)
    
    for case_result in all_results:
        test_case = case_result['test_case']
        print(f"\n{'─' * 80}")
        print(f"TEST CASE {test_case['id']}: {test_case['query']}")
        print("─" * 80)
        
        for model_result in case_result['model_results']:
            print(f"\n[{model_result['model'].upper()}]")
            if model_result['success']:
                print(f"Answer: {model_result['answer']}")
            else:
                print(f"ERROR: {model_result['error']}")
            print()



if __name__ == '__main__':
    print("\nStarting model comparison...\n")
    results = run_comparison()
    print_summary(results)
    print_answers_for_qualitative_review(results)
