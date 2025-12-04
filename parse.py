import csv
import json

def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

full_results = read_json('ry7_careful_full_results.json')

for result in full_results["results"]:
    # list of to and from relationships
    exp = result["testParams"]["expectations"]

    # list of to and from relationships
    res = result["generatedResponse"]["model"]["relationships"]

    # Find relationships that are in both exp and res
    common_relationships = []
    
    # Find common relationships (matching to, from, and polarity)
    for exp_rel in exp:
        for res_rel in res:
            if (exp_rel.get("to") == res_rel.get("to") and 
                exp_rel.get("from") == res_rel.get("from") and
                exp_rel.get("polarity") == res_rel.get("polarity")):
                common_relationships.append(exp_rel)
                break

    just_exp = []
    just_res = []
    
    # Find relationships only in exp
    for exp_rel in exp:
        found_in_res = False
        for res_rel in res:
            if (exp_rel.get("to") == res_rel.get("to") and 
                exp_rel.get("from") == res_rel.get("from") and
                exp_rel.get("polarity") == res_rel.get("polarity")):
                found_in_res = True
                break
        if not found_in_res:
            just_exp.append(exp_rel)
    
    # Find relationships only in res
    for res_rel in res:
        found_in_exp = False
        for exp_rel in exp:
            if (res_rel.get("to") == exp_rel.get("to") and 
                res_rel.get("from") == exp_rel.get("from") and
                exp_rel.get("polarity") == res_rel.get("polarity")):
                found_in_exp = True
                break
        if not found_in_exp:
            just_res.append(res_rel)


    chunks = result['testParams']['additionalParameters']['backgroundKnowledge']
    # Split by periods, commas, and newlines while keeping the delimiter at the end
    import re
    
    # Find the first opening parenthesis
    first_paren = chunks.find('(')
    
    if first_paren != -1:
        # Split only the part before the first (
        before_paren = chunks[:first_paren]
        split_chunks = re.split('([.,\n])', before_paren)
        
        # Recombine to keep delimiters at the end of each chunk
        formatted_chunks = []
        for i in range(0, len(split_chunks) - 1, 2):
            if i + 1 < len(split_chunks):
                formatted_chunks.append(split_chunks[i] + split_chunks[i + 1])
            else:
                formatted_chunks.append(split_chunks[i])
        
        # Print each chunk before the parenthesis
        for chunk in formatted_chunks:
            if chunk.strip():  # Only print non-empty chunks
                print(f"{chunk}")
        
        # Extract everything inside parentheses and put on one line
        paren_content = chunks[first_paren:]
        if paren_content:
            print(f"{paren_content}")
    else:
        # No parentheses found, use original chunking logic
        split_chunks = re.split('([.,\n])', chunks)
        formatted_chunks = []
        for i in range(0, len(split_chunks) - 1, 2):
            if i + 1 < len(split_chunks):
                formatted_chunks.append(split_chunks[i] + split_chunks[i + 1])
            else:
                formatted_chunks.append(split_chunks[i])
        
        for chunk in formatted_chunks:
            if chunk.strip():
                print(f"{chunk}")

    for r in common_relationships:
        # print in csv format to console
        print("\t".join(["both", r["from"], f"{r['polarity']}>", r["to"]]))

    for r in just_exp:
        print("\t".join(["was missing", r["from"], f"{r['polarity']}>", r["to"]]))

    for r in just_res:
        print("\t".join(["made up", r["from"], f"{r['polarity']}>", r["to"]]))

    print("")