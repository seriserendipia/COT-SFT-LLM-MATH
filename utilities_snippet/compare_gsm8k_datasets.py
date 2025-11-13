"""
Compare answers between openai/gsm8k and ankner/gsm8k-CoT datasets
Iterate through first 100 questions from CoT dataset and find matches in OpenAI dataset
Output real-time accuracy, speed, and save mismatches to JSON
"""

from datasets import load_dataset
import json
import time
from datetime import datetime

def extract_numeric_answer(answer_text):
    """Extract the numeric answer from answer text"""
    if '####' in answer_text:
        # Format: "explanation #### answer"
        return answer_text.split('####')[-1].strip()
    return answer_text.strip()

def find_matching_question(cot_question, openai_dataset):
    """Find matching question in OpenAI dataset"""
    cot_q_stripped = cot_question.strip()
    for idx, openai_data in enumerate(openai_dataset):
        if openai_data['question'].strip() == cot_q_stripped:
            return idx, openai_data
    return None, None

def main():
    print("Loading datasets...")
    start_time = time.time()
    
    # Load openai/gsm8k dataset (train split)
    gsm8k_openai = load_dataset("openai/gsm8k", "main", split="train")
    
    # Load ankner/gsm8k-CoT dataset (train split)
    gsm8k_cot = load_dataset("ankner/gsm8k-CoT", split="train")
    
    load_time = time.time() - start_time
    print(f"OpenAI GSM8K train size: {len(gsm8k_openai)}")
    print(f"Ankner GSM8K-CoT train size: {len(gsm8k_cot)}")
    print(f"Loading time: {load_time:.2f}s\n")
    
    print("="*80)
    print("Processing first 100 questions from CoT dataset...")
    print("="*80 + "\n")
    
    mismatches = []
    matched_count = 0
    answer_matched_count = 0
    not_found_count = 0
    
    process_start = time.time()
    
    for i in range(min(100, len(gsm8k_cot))):
        iter_start = time.time()
        
        cot_data = gsm8k_cot[i]
        cot_question = cot_data['question'].strip()
        cot_answer = cot_data['answer'].strip()
        cot_numeric = extract_numeric_answer(cot_answer)
        
        # Find matching question in OpenAI dataset
        openai_idx, openai_data = find_matching_question(cot_question, gsm8k_openai)
        
        iter_time = time.time() - iter_start
        
        if openai_data is None:
            # Question not found in OpenAI dataset
            not_found_count += 1
            mismatches.append({
                'cot_index': i,
                'openai_index': None,
                'status': 'question_not_found',
                'cot_question': cot_question,
                'cot_answer': cot_answer,
                'cot_numeric_answer': cot_numeric,
                'openai_question': None,
                'openai_answer': None,
                'openai_numeric_answer': None
            })
            
            # Real-time output
            elapsed = time.time() - process_start
            accuracy = (matched_count / (i + 1)) * 100
            answer_accuracy = (answer_matched_count / (i + 1)) * 100
            not_found_rate = (not_found_count / (i + 1)) * 100
            avg_speed = (i + 1) / elapsed
            
            print(f"[{i+1}/100] ❌ NOT FOUND | "
                  f"Accuracy: {accuracy:.1f}% | "
                  f"Answer Match: {answer_accuracy:.1f}% | "
                  f"Not Found: {not_found_rate:.1f}% | "
                  f"Speed: {avg_speed:.2f} q/s | "
                  f"Time: {iter_time:.3f}s")
        else:
            # Question found, check answer
            openai_question = openai_data['question'].strip()
            openai_answer = openai_data['answer'].strip()
            openai_numeric = extract_numeric_answer(openai_answer)
            
            answers_match = (cot_numeric == openai_numeric)
            
            if answers_match:
                matched_count += 1
                answer_matched_count += 1
            else:
                # Answer mismatch
                mismatches.append({
                    'cot_index': i,
                    'openai_index': openai_idx,
                    'status': 'answer_mismatch',
                    'cot_question': cot_question,
                    'cot_answer': cot_answer,
                    'cot_numeric_answer': cot_numeric,
                    'openai_question': openai_question,
                    'openai_answer': openai_answer,
                    'openai_numeric_answer': openai_numeric
                })
                matched_count += 1  # Question matched but answer didn't
            
            # Real-time output
            elapsed = time.time() - process_start
            accuracy = (matched_count / (i + 1)) * 100
            answer_accuracy = (answer_matched_count / (i + 1)) * 100
            not_found_rate = (not_found_count / (i + 1)) * 100
            avg_speed = (i + 1) / elapsed
            
            status = "✓" if answers_match else "⚠ ANSWER DIFF"
            print(f"[{i+1}/100] {status} | "
                  f"Q-Match: {accuracy:.1f}% | "
                  f"A-Match: {answer_accuracy:.1f}% | "
                  f"Not Found: {not_found_rate:.1f}% | "
                  f"Speed: {avg_speed:.2f} q/s | "
                  f"Time: {iter_time:.3f}s")
    
    total_time = time.time() - process_start
    
    # Final statistics
    print("\n" + "="*80)
    print("FINAL STATISTICS")
    print("="*80)
    print(f"Total processed: 100")
    print(f"Questions matched: {matched_count} ({matched_count}%)")
    print(f"Answers matched: {answer_matched_count} ({answer_matched_count}%)")
    print(f"Questions not found: {not_found_count} ({not_found_count}%)")
    print(f"Total mismatches: {len(mismatches)}")
    print(f"Total processing time: {total_time:.2f}s")
    print(f"Average speed: {100/total_time:.2f} questions/second")
    
    # Save mismatches to JSON
    output_file = "gsm8k_comparison_mismatches.json"
    output_data = {
        'metadata': {
            'comparison_date': datetime.now().isoformat(),
            'total_compared': 100,
            'questions_matched': matched_count,
            'answers_matched': answer_matched_count,
            'questions_not_found': not_found_count,
            'total_mismatches': len(mismatches),
            'processing_time_seconds': total_time
        },
        'mismatches': mismatches
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nMismatches saved to: {output_file}")
    
    if mismatches:
        print("\n" + "="*80)
        print("MISMATCH DETAILS")
        print("="*80 + "\n")
        
        for mismatch in mismatches:
            print(f"CoT Index: {mismatch['cot_index']}")
            print(f"OpenAI Index: {mismatch['openai_index']}")
            print(f"Status: {mismatch['status']}")
            
            if mismatch['status'] == 'question_not_found':
                print(f"\n--- CoT Question (NOT FOUND IN OPENAI) ---")
                print(f"Question: {mismatch['cot_question'][:200]}...")
                print(f"Answer: {mismatch['cot_numeric_answer']}")
            else:
                print(f"\n--- CoT Dataset ---")
                print(f"Question: {mismatch['cot_question'][:150]}...")
                print(f"Numeric Answer: {mismatch['cot_numeric_answer']}")
                
                print(f"\n--- OpenAI Dataset ---")
                print(f"Question: {mismatch['openai_question'][:150]}...")
                print(f"Numeric Answer: {mismatch['openai_numeric_answer']}")
            
            print("\n" + "-"*80 + "\n")
    else:
        print("\n✓ All 100 questions and answers match perfectly!")

if __name__ == "__main__":
    main()
