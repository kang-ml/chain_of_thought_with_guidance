demos = [
    {'context': "The cow is blue. The cow is round. The cow likes the lion. The cow visits the tiger. The lion is cold. The lion is nice. The lion likes the squirrel. The squirrel is round. The squirrel sees the lion. The squirrel visits the cow. The tiger likes the cow. The tiger likes the squirrel. If something is cold then it visits the tiger. If something visits the tiger then it is nice. If something sees the tiger and it is young then it is blue. If something is nice then it sees the tiger. If something likes the squirrel and it likes the cow then it visits the tiger. If something is nice and it sees the tiger then it is young. If the cow is cold and the cow visits the lion then the lion sees the squirrel.",
     'question': "Based on the above information, is the following statement true, false, or unknown? The tiger is not young.",
     'reason': "The tiger likes the cow. The tiger likes the squirrel. If something likes the squirrel and it likes the cow, then it visits the tiger. So the tiger visits the tiger. If something visits the tiger then it is nice. So the tiger is nice. If something is nice and it sees the tiger then it is young. So the tiger is young.",
     'ans': "False"},
    {'context': "The dog sees the rabbit. The dog sees the squirrel. The dog sees the tiger. The rabbit eats the dog. The rabbit does not eat the tiger. The rabbit does not like the tiger. The squirrel does not see the rabbit. The tiger does not eat the rabbit. The tiger is not kind. The tiger likes the dog. The tiger sees the dog. If something is cold then it likes the rabbit. If something eats the tiger and it is nice then it likes the rabbit. If something likes the squirrel then the squirrel likes the rabbit. If something likes the rabbit and the rabbit is kind then it sees the tiger. If something likes the tiger then the tiger is young. If something is young and it eats the rabbit then it likes the tiger. If something sees the rabbit then the rabbit is cold. If something likes the rabbit then it likes the squirrel. If something likes the squirrel then the squirrel is cold.",
     'question': "Question: Based on the above information, is the following statement true, false, or unknown? The rabbit is cold.",
     'reason': "The dog sees the rabbit. If something sees the rabbit then the rabbit is cold. So the rabbit is cold.",
     'ans': "True"},
    {'context': "Anne is not blue. Anne is cold. Anne is round. Fiona is blue. Fiona is furry. Gary is furry. Gary is quiet. Gary is smart. Harry is cold. Harry is quiet. If something is furry and not blue then it is nice. If Anne is furry then Anne is nice. Smart, furry things are round.",
     'question': "Based on the above information, is the following statement true, false, or unknown? Anne is quiet.",
     'reason': "Anne is not blue. Anne is cold. Anne is round. Gary is quiet. Not related to Anne is quiet. So it is unknown if Anne is quiet.",
     'ans': "Unknown"}
]